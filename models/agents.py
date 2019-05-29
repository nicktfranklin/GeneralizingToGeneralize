import numpy as np
import pandas as pd
from misc import sample_cmf, augment_assignments
from grid_world import GridWorld
from scipy.misc import logsumexp
from cython_library import GoalHypothesis, MappingHypothesis, policy_iteration, value_iteration

""" these agents differ from the generative agents I typically use in that I need to pass a transition
function (and possibly a reward function) to the agent for each trial. """


def make_q_primitive(q_abstract, mapping):
    q_primitive = np.zeros(8)
    n, m = np.shape(mapping)
    for aa in range(m):
        for a in range(n):
            q_primitive[a] += q_abstract[aa] * mapping[a, aa]
    return q_primitive


def kl_divergence(q, p):
    d = 0
    for q_ii, p_ii in zip(q, p):
        if p_ii > 0:
            d += p_ii * np.log2(p_ii/q_ii)
    return d


class MultiStepAgent(object):

    def __init__(self, task):
        self.task = task
        # assert type(self.task) is Task
        self.current_trial = 0
        self.results = []
        self.is_mixture = False

    def get_action_pmf(self, location):
        assert type(location) is tuple
        return np.ones(self.task.n_primitive_actions, dtype=float) / self.task.n_primitive_actions

    def get_primitive_q(self, location):
        return np.ones(self.task.n_primitive_actions)

    def get_action_cmf(self, location):
        return np.cumsum(self.get_action_pmf(location))

    def get_goal_probability(self, context):
        return np.ones(self.task.n_goals, dtype=np.float) / self.task.n_goals

    def get_mapping_function(self, context, aa):
        return np.ones(self.task.n_primitive_actions, dtype=np.float) / self.task.n_primitive_actions

    def select_action(self, location):
        return sample_cmf(self.get_action_cmf(location))

    def update_mapping(self, c, a, aa):
        pass

    def update_goal_values(self, c, goal, r):
        pass

    def prune_hypothesis_space(self, threshold=50.):
        pass

    def augment_assignments(self, context):
        pass

    def count_hypotheses(self):
        return 1

    def get_joint_probability(self):
        return None

    def get_responsibilities(self):
        return None, None

    def get_responsibilities_derivative(self):
        return None, None

    def generate(self, evaluate=True, debug=False, pruning_threshold=None):

        # initialize variables
        step_counter = 0
        times_seen_ctx = np.zeros(self.task.n_ctx)
        new_trial = True
        c = None
        t = None
        goal_locations = None
        kl_goal_pmf = None
        kl_map_pmf = None
        ii = 0

        while True:

            if new_trial:
                c = self.task.get_current_context()  # current context
                t = self.task.get_trial_number()  # current trial number
                goal_locations = self.task.get_goal_locations()
                new_trial = False
                times_seen_ctx[c] += 1
                step_counter = 0

                self.prune_hypothesis_space(threshold=pruning_threshold)
                if times_seen_ctx[c] == 1:
                    self.augment_assignments(c)

                if evaluate:
                    # compare the difference in goal probabilities
                    agent_goal_probability = self.get_goal_probability(c)
                    true_goal_probability = self.task.get_goal_values()
                    kl_goal_pmf = kl_divergence(agent_goal_probability, true_goal_probability)

            if evaluate:
                # compare the mapping probabilities for the greedy policy:
                kl_map_pmf = 0
                for aa0 in range(self.task.n_abstract_actions):
                    agent_mapping = self.get_mapping_function(c, aa0)
                    true_mapping = self.task.get_mapping_function(aa0)
                    kl_map_pmf += kl_divergence(agent_mapping, true_mapping)

            step_counter += 1

            # select an action
            start_location = self.task.get_location()
            action = self.select_action(start_location)

            # save for data output
            action_map = self.task.get_action_map()
            walls = self.task.get_walls()

            if debug:
                pmf = self.get_action_pmf(start_location)
                p = pmf[action]
            else:
                p = None

            # take an action
            aa, end_location, goal_id, r = self.task.move(action)

            # update mapping
            self.update_mapping(c, action, aa)

            # End condition is a goal check
            if goal_id is not None:
                self.update_goal_values(c, goal_id, r)
                new_trial = True

            results_dict = {
                'Context': c,
                'Start Location': [start_location],
                'Key-press': action,
                'End Location': [end_location],
                'Action Map': [action_map],
                'Walls': [walls],
                'Action': aa,  # the cardinal movement, in words
                'Reward': r,
                'In Goal': goal_id is not None,
                'Chosen Goal': goal_id,
                'Steps Taken': step_counter,
                'Goal Locations': [goal_locations],
                'Trial Number': t,
                'Times Seen Context': times_seen_ctx[c],
                'Action Probability': p,
            }
            if evaluate:
                results_dict['Goal KL Divergence'] = kl_goal_pmf
                results_dict['Map KL Divergence'] = kl_map_pmf

            if self.is_mixture:
                results_dict['Joint Probability'] = self.get_joint_probability()
                ind, joint = self.get_responsibilities()
                results_dict["Ind Weight"] = ind
                results_dict["Joint Weight"] = joint
                d_ind, d_joint = self.get_responsibilities_derivative()
                results_dict["Ind dWeight"] = d_ind
                results_dict["Joint dWeight"] = d_joint

            self.results.append(results_dict)
            ii += 1

            # evaluate stop condition
            if self.task.end_check():
                break

            # stop criterion
            if step_counter > 1000:
                return None

        return self.get_results()

    def get_results(self):
        return pd.DataFrame(self.results, index=range(len(self.results)))


class FlatAgent(MultiStepAgent):

    def __init__(self, task, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001):
        super(FlatAgent, self).__init__(task)

        self.gamma = gamma
        self.inv_temp = inv_temp
        self.stop_criterion = stop_criterion

        # initialize the hypothesis space
        self.goal_hypotheses = [GoalHypothesis(self.task.n_goals, 1.0, goal_prior)]
        self.mapping_hypotheses = [
            MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
                              1.0, mapping_prior)
        ]

        # initialize the belief spaces
        self.log_belief_goal = np.ones(1, dtype=float)
        self.log_belief_map = np.ones(1, dtype=float)

    def count_hypotheses(self):
        return len(self.log_belief_map)

    def augment_assignments(self, context):

        h_g = self.goal_hypotheses[0]
        h_m = self.mapping_hypotheses[0]

        assert type(h_g) is GoalHypothesis
        assert type(h_m) is MappingHypothesis

        h_g.add_new_context_assignment(context, context)
        h_m.add_new_context_assignment(context, context)

        # don't need to update the belief for the flat agent

    def update_goal_values(self, c, goal, r):
        goal_idx_num = self.task.get_goal_index(goal)
        for h_goal in self.goal_hypotheses:
            assert type(h_goal) is GoalHypothesis
            h_goal.update(c, goal_idx_num, r)

        # update the posterior of the goal hypotheses
        log_belief = np.zeros(len(self.goal_hypotheses))
        for ii, h_goal in enumerate(self.goal_hypotheses):
            assert type(h_goal) is GoalHypothesis
            log_belief[ii] = h_goal.get_log_posterior()

        self.log_belief_goal = log_belief

    def update_mapping(self, c, a, aa):
        for h_m in self.mapping_hypotheses:
            assert type(h_m) is MappingHypothesis
            h_m.update_mapping(c, a, self.task.abstract_action_key[aa])

        # update the posterior of the mapping hypothesis
        log_belief = np.zeros(len(self.mapping_hypotheses))
        for ii, h_m in enumerate(self.mapping_hypotheses):
            assert type(h_m) is MappingHypothesis
            log_belief[ii] = h_m.get_log_posterior()

        self.log_belief_map = log_belief

    def get_goal_probability(self, context):
        # get the value of the goals for the MAP cluster
        ii = np.argmax(self.log_belief_goal)
        h_goal = self.goal_hypotheses[ii]
        assert type(h_goal) is GoalHypothesis

        goal_pmf = h_goal.get_goal_probability(context)

        return goal_pmf

    def get_mapping_function(self, context, aa):
        # used to calculate cross-entropy
        ii = np.argmax(self.log_belief_map)

        h_map = self.mapping_hypotheses[ii]
        assert type(h_map) is MappingHypothesis

        mapping_pmf = np.zeros(self.task.n_primitive_actions, dtype=float)
        for a0 in range(self.task.n_primitive_actions):
            mapping_pmf[a0] = h_map.get_mapping_probability(context, a0, aa)

        return mapping_pmf

    def convert_goal_values_to_reward(self, goal_pmf):
        grid = self.task.get_current_gridworld()
        assert type(grid) is GridWorld

        reward_function = np.zeros(len(grid.state_location_key))
        for location, goal in grid.goal_locations.iteritems():
            goal_state = grid.state_location_key[location]
            p = goal_pmf[self.task.get_goal_index(goal)]
            reward_function[goal_state] = p - 0.1 * (1 - p)

        return reward_function

    def get_abstract_action_q(self, location):
        c = self.task.get_current_context()
        t = self.task.get_transition_function()

        r = self.convert_goal_values_to_reward(self.get_goal_probability(c))
        v = value_iteration(t, r, self.gamma, self.stop_criterion)
        s = self.task.state_location_key[location]

        # use the belman equation to get q-values
        q = np.zeros(self.task.n_abstract_actions)
        for aa in range(self.task.n_abstract_actions):
            q[aa] = np.sum(t[s, aa, :] * (r[:] + self.gamma * v[:]))

        return q

    def get_primitive_q(self, location):
        q_aa = self.get_abstract_action_q(location)
        c = self.task.get_current_context()

        # use the mapping distribution to get the q-values for the primitive actiosn
        q_a = np.zeros(self.task.n_primitive_actions)
        for aa0 in np.arange(self.task.n_abstract_actions, dtype=np.int32):
            ii = np.argmax(self.log_belief_map)
            h_map = self.mapping_hypotheses[ii]

            _mapping_pmf = np.zeros(self.task.n_primitive_actions)
            for a0 in np.arange(self.task.n_primitive_actions, dtype=np.int32):
                _mapping_pmf[a0] = h_map.get_mapping_probability(c, a0, aa0)
            q_a += _mapping_pmf * q_aa[aa0]
        return q_a

    def get_action_pmf(self, location):
        c = self.task.get_current_context()

        q = self.get_abstract_action_q(location)

        # use softmax to convert to probability function
        p_aa = np.exp(self.inv_temp * q) / np.sum(np.exp(self.inv_temp * q))

        # use the distribution P(A=A*) to get P(a=a*) by integration
        # P(a=a*) = Sum[ P(a=A) x P(A=A*) ]
        pmf = np.zeros(self.task.n_primitive_actions)
        for aa0 in np.arange(self.task.n_abstract_actions, dtype=np.int32):

            # loop through the mapping hypotheses to get a mapping distribution
            # for each abstract action
            ii = np.argmax(self.log_belief_map)
            h_map = self.mapping_hypotheses[ii]

            _mapping_pmf = np.zeros(self.task.n_primitive_actions)
            for a0 in np.arange(self.task.n_primitive_actions, dtype=np.int32):
                _mapping_pmf[a0] = h_map.get_mapping_probability(c, a0, aa0)
            pmf += _mapping_pmf * p_aa[aa0]

        # because we omit low probability goals from planning,
        # sometimes the pmf does not sum to one.
        # therefore, we need to re-normalize
        pmf /= pmf.sum()
        return pmf


class IndependentClusterAgent(FlatAgent):

    def __init__(self, task, alpha=1.0, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001):
        super(FlatAgent, self).__init__(task)

        self.gamma = gamma
        self.inv_temp = inv_temp
        self.stop_criterion = stop_criterion
        self.alpha = alpha

        # initialize the hypothesis space with a single hypothesis that can be augmented
        # as new contexts are encountered
        self.goal_hypotheses = [GoalHypothesis(self.task.n_goals, alpha, goal_prior)]
        self.mapping_hypotheses = [
            MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
                              alpha, mapping_prior)
        ]

        # initialize the belief spaces
        self.log_belief_goal = np.ones(1, dtype=float)
        self.log_belief_map = np.ones(1, dtype=float)

    def count_hypotheses(self):
        return len(self.log_belief_map) + len(self.log_belief_goal)

    def augment_assignments(self, context):
        _goal_hypotheses = list()
        _mapping_hypotheses = list()
        _goal_log_belief = list()
        _mapping_log_belief = list()

        for h_g in self.goal_hypotheses:
            assert type(h_g) is GoalHypothesis

            old_assignments = h_g.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            # create a list of the new clusters to add
            for assignment in new_assignments:
                k = assignment[context]
                h_r0 = h_g.deep_copy()
                h_r0.add_new_context_assignment(context, k)

                _goal_hypotheses.append(h_r0)
                _goal_log_belief.append(h_r0.get_log_posterior())

        for h_m in self.mapping_hypotheses:
            assert type(h_m) is MappingHypothesis

            old_assignments = h_m.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            for assignment in new_assignments:
                k = assignment[context]
                h_m0 = h_m.deep_copy()
                h_m0.add_new_context_assignment(context, k)

                _mapping_hypotheses.append(h_m0)
                _mapping_log_belief.append(h_m0.get_log_posterior())

        self.mapping_hypotheses = _mapping_hypotheses
        self.goal_hypotheses = _goal_hypotheses
        self.log_belief_map = _mapping_log_belief
        self.log_belief_goal = _goal_log_belief

    def prune_hypothesis_space(self, threshold=50.):
        if threshold is not None:
            _log_belief_goal = []
            _log_belief_map = []
            _goal_hypotheses = []
            _mapping_hypotheses = []

            log_threshold = np.log(threshold)

            max_belief = np.max(self.log_belief_goal)
            for ii, log_b in enumerate(self.log_belief_goal):
                if max_belief - log_b < log_threshold:
                    _log_belief_goal.append(log_b)
                    _goal_hypotheses.append(self.goal_hypotheses[ii])

            max_belief = np.max(self.log_belief_map)
            for ii, log_b in enumerate(self.log_belief_map):
                if max_belief - log_b < log_threshold:
                    _log_belief_map.append(log_b)
                    _mapping_hypotheses.append(self.mapping_hypotheses[ii])

            self.log_belief_goal = _log_belief_goal
            self.goal_hypotheses = _goal_hypotheses
            self.log_belief_map = _log_belief_map
            self.mapping_hypotheses = _mapping_hypotheses

    def get_goal_prior_over_new_contexts(self):
        from cython_library.core import get_prior_log_probability
        log_prior_pmf = []
        goal_pmfs = []

        for h_goal in self.goal_hypotheses:

            set_assignment = h_goal.get_set_assignments()
            n_k = np.max(set_assignment) + 1
            ll = h_goal.get_log_likelihood()

            for ts in range(n_k):
                sa0 = np.array(np.concatenate([set_assignment, [ts]]), dtype=np.int32)
                log_prior_pmf.append(get_prior_log_probability(sa0, self.alpha) + ll)
                goal_pmfs.append(h_goal.get_set_goal_probability(ts))

            # new cluster
            sa0 = np.array(np.concatenate([set_assignment, [n_k]]), dtype=np.int32)
            log_prior_pmf.append(get_prior_log_probability(sa0, self.alpha) + ll)
            goal_pmfs.append(np.ones(self.task.n_goals, dtype=np.float32) / self.task.n_goals)

        # Normalize the prior
        log_prior_pmf = np.array(log_prior_pmf)
        log_prior_pmf -= np.max(log_prior_pmf)
        prior_pmf = np.exp(log_prior_pmf)
        prior_pmf /= np.sum(prior_pmf)

        # weight the goal probability to create a distribution over goals
        goal_pmf = np.squeeze(np.dot(np.array(goal_pmfs).T, np.array([prior_pmf]).T))

        goal_pmf = pd.DataFrame({
            'Probability': goal_pmf,
            'Goal': range(1, self.task.n_goals + 1),
            'Map': ['Combined'] * self.task.n_goals,
            'Model': ['IC'] * self.task.n_goals
        })
        return goal_pmf


class JointClusteringAgent(MultiStepAgent):

    def __init__(self, task, alpha=1.0, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001):
        super(JointClusteringAgent, self).__init__(task)

        self.gamma = gamma
        self.inv_temp = inv_temp
        self.stop_criterion = stop_criterion
        self.alpha = alpha

        # initialize the hypothesis space with a single hypothesis that can be augmented
        # as new contexts are encountered
        self.goal_hypotheses = [GoalHypothesis(self.task.n_goals, alpha, goal_prior)]
        self.mapping_hypotheses = [
            MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
                              alpha, mapping_prior)
        ]

        # initialize the belief spaces
        self.log_belief = np.zeros(1, dtype=float)
        self.map_likelihood = np.zeros(1, dtype=float)
        self.goal_likelihood = np.zeros(1, dtype=float)

    def augment_assignments(self, context):
        _goal_hypotheses = list()
        _mapping_hypotheses = list()

        for h_g, h_m in zip(self.goal_hypotheses, self.mapping_hypotheses):
            assert type(h_g) is GoalHypothesis
            assert type(h_m) is MappingHypothesis

            old_assignments = h_g.get_assignments()
            new_assignments = augment_assignments([old_assignments], context)

            for assignment in new_assignments:
                k = assignment[context]
                h_g0 = h_g.deep_copy()
                h_g0.add_new_context_assignment(context, k)

                h_m0 = h_m.deep_copy()
                h_m0.add_new_context_assignment(context, k)

                _goal_hypotheses.append(h_g0)
                _mapping_hypotheses.append(h_m0)

        self.mapping_hypotheses = _mapping_hypotheses
        self.goal_hypotheses = _goal_hypotheses

        self.update_goal_log_likelihood()
        self.update_mapping_loglikelihood()
        self.update_belief()

    def update_mapping_loglikelihood(self):
        self.map_likelihood = np.zeros(len(self.mapping_hypotheses))
        for ii, h_map in enumerate(self.mapping_hypotheses):
            self.map_likelihood[ii] = h_map.get_log_likelihood()

    def update_goal_log_likelihood(self):
        self.goal_likelihood = np.zeros(len(self.goal_hypotheses))
        for ii, h_goal in enumerate(self.goal_hypotheses):
            self.goal_likelihood[ii] = h_goal.get_log_likelihood()

    def update_belief(self):
        log_posterior = np.zeros(len(self.mapping_hypotheses))
        for ii, h_map in enumerate(self.mapping_hypotheses):
            log_posterior[ii] = h_map.get_log_prior() + self.map_likelihood[ii] + \
                                self.goal_likelihood[ii]
        self.log_belief = log_posterior

    def count_hypotheses(self):
        return len(self.log_belief)

    def update_goal_values(self, c, goal, r):
        goal_idx_num = self.task.get_goal_index(goal)
        for h_goal in self.goal_hypotheses:
            assert type(h_goal) is GoalHypothesis
            h_goal.update(c, goal_idx_num, r)

        # update the belief distribution
        self.update_goal_log_likelihood()
        self.update_belief()

    def update_mapping(self, c, a, aa):
        for h_map in self.mapping_hypotheses:
            assert type(h_map) is MappingHypothesis
            h_map.update_mapping(c, a, self.task.abstract_action_key[aa])

        # update the belief distribution
        self.update_mapping_loglikelihood()
        self.update_belief()

    def prune_hypothesis_space(self, threshold=50.):
        if threshold is not None:
            _goal_hypotheses = []
            _mapping_hypotheses = []
            _goal_log_likelihoods = []
            _mapping_log_likelihoods = []
            _log_belief = []

            max_belief = np.max(self.log_belief)
            log_threshold = np.log(threshold)

            for ii, log_b in enumerate(self.log_belief):
                if max_belief - log_b < log_threshold:
                    _log_belief.append(log_b)
                    _goal_log_likelihoods.append(self.goal_likelihood[ii])
                    _mapping_log_likelihoods.append(self.map_likelihood[ii])

                    _goal_hypotheses.append(self.goal_hypotheses[ii])
                    _mapping_hypotheses.append(self.mapping_hypotheses[ii])

            self.goal_hypotheses = _goal_hypotheses
            self.mapping_hypotheses = _mapping_hypotheses

            self.goal_likelihood = _goal_log_likelihoods
            self.map_likelihood = _mapping_log_likelihoods
            self.log_belief = _log_belief

    def get_goal_probability(self, context):

        # get the value of the goals of the MAP cluster
        ii = np.argmax(self.log_belief)
        h_goal = self.goal_hypotheses[ii]
        assert type(h_goal) is GoalHypothesis

        goal_expectation = h_goal.get_goal_probability(context)

        return goal_expectation

    def get_mapping_function(self, context, aa):
        # used to calculate cross entropy
        ii = np.argmax(self.log_belief)

        h_map = self.mapping_hypotheses[ii]
        assert type(h_map) is MappingHypothesis

        mapping_expectation = np.zeros(self.task.n_primitive_actions, dtype=float)
        for a0 in range(self.task.n_primitive_actions):
            mapping_expectation[a0] += h_map.get_mapping_probability(context, a0, aa)

        return mapping_expectation

    def convert_goal_values_to_reward(self, goal_pmf):
        grid = self.task.get_current_gridworld()
        assert type(grid) is GridWorld

        reward_function = np.zeros(len(grid.state_location_key))
        for location, goal in grid.goal_locations.iteritems():
            goal_state = grid.state_location_key[location]
            p = goal_pmf[self.task.get_goal_index(goal)]
            reward_function[goal_state] = p - 0.1 * (1 - p)

        return reward_function

    def get_abstract_action_q(self, location):
        c = self.task.get_current_context()
        t = self.task.get_transition_function()

        r = self.convert_goal_values_to_reward(self.get_goal_probability(c))
        v = value_iteration(t, r, self.gamma, self.stop_criterion)
        s = self.task.state_location_key[location]

        # use the belman equation to get q-values
        q = np.zeros(self.task.n_abstract_actions)
        for aa in range(self.task.n_abstract_actions):
            q[aa] = np.sum(t[s, aa, :] * (r[:] + self.gamma * v[:]))

        return q

    def get_primitive_q(self, location):
        q_aa = self.get_abstract_action_q(location)
        c = self.task.get_current_context()

        # use the mapping distribution to get the q-values for the primitive actiosn
        q_a = np.zeros(self.task.n_primitive_actions)
        for aa0 in np.arange(self.task.n_abstract_actions, dtype=np.int32):
            ii = np.argmax(self.log_belief)
            h_map = self.mapping_hypotheses[ii]

            _mapping_pmf = np.zeros(self.task.n_primitive_actions)
            for a0 in np.arange(self.task.n_primitive_actions, dtype=np.int32):
                _mapping_pmf[a0] = h_map.get_mapping_probability(c, a0, aa0)
            q_a += _mapping_pmf * q_aa[aa0]

        return q_a

    def get_action_pmf(self, location, threshold=0.01):

        c = self.task.get_current_context()
        q = self.get_abstract_action_q(location)

        # use softmax to convert to probability function
        p_aa = np.exp(self.inv_temp * q) / np.sum(np.exp(self.inv_temp * q))

        # use the distribution P(A=A*) to get P(a=a*) by integration
        # P(a=a*) = Sum[ P(a=A) x P(A=A*) ]
        pmf = np.zeros(self.task.n_primitive_actions)
        for aa0 in np.arange(self.task.n_abstract_actions, dtype=np.int32):

            # get a mapping probability
            # for each abstract action
            ii = np.argmax(self.log_belief)
            h_map = self.mapping_hypotheses[ii]

            _mapping_pmf = np.zeros(self.task.n_primitive_actions)
            for a0 in np.arange(self.task.n_primitive_actions, dtype=np.int32):
                _mapping_pmf[a0] = h_map.get_mapping_probability(c, a0, aa0)
            pmf += _mapping_pmf * p_aa[aa0]

        # because we omit low probability goals from planning, sometimes the pmf does not sum to one.
        # therefore, we need to re-normalize
        pmf /= pmf.sum()
        return pmf

    def get_goal_prior_over_new_contexts(self):
        from cython_library.core import get_prior_log_probability

        all_log_prior_pmfs = [list() for _ in range(len(self.task.list_action_maps))]
        goal_pmfs = []

        aa_key = self.task.abstract_action_key

        for h_goal, h_map in zip(self.goal_hypotheses, self.mapping_hypotheses):

            set_assignment = h_map.get_set_assignments()
            n_k = np.max(set_assignment) + 1

            for ts in range(n_k):
                sa0 = np.array(np.concatenate([set_assignment, [ts]]), dtype=np.int32)
                goal_pmfs.append(h_goal.get_set_goal_probability(ts))

                for m_idx, action_map in enumerate(self.task.list_action_maps):
                    ll = h_map.get_log_likelihood() + h_goal.get_log_likelihood()
                    for key, dir_ in action_map.iteritems():
                        ll += np.log(h_map.get_pr_aa_given_a_ts(ts, key, aa_key[dir_]))
                    all_log_prior_pmfs[m_idx].append(ll + get_prior_log_probability(sa0, self.alpha))

            # new cluster
            goal_pmfs.append(np.ones(self.task.n_goals, dtype=np.float32) / self.task.n_goals)
            for m_idx, action_map in enumerate(self.task.list_action_maps):
                sa0 = np.array(np.concatenate([set_assignment, [n_k]]), dtype=np.int32)
                ll = h_map.get_log_likelihood() + h_goal.get_log_likelihood()
                ll += np.log(self.task.n_abstract_actions / float(self.task.n_primitive_actions))
                all_log_prior_pmfs[m_idx].append(ll + get_prior_log_probability(sa0, self.alpha))

        # Normalize the prior
        def normalize_prior(log_prior_pmf):
            log_prior_pmf = np.array(log_prior_pmf)
            log_prior_pmf -= np.max(log_prior_pmf)
            prior_pmf = np.exp(log_prior_pmf)
            prior_pmf /= np.sum(prior_pmf)
            return prior_pmf

        results = []
        for m_idx, action_map in enumerate(self.task.list_action_maps):
            prior_pmf = normalize_prior(all_log_prior_pmfs[m_idx])

            # weight the goal probability to create a distribution over goals
            goal_pmf = np.squeeze(np.dot(np.array(goal_pmfs).T, np.array([prior_pmf]).T))
            results.append(pd.DataFrame({
                'Probability': goal_pmf,
                'Goal': range(1, self.task.n_goals + 1),
                'Map': [m_idx] * self.task.n_goals,
                'Model': ['TS'] * self.task.n_goals,
                'Action Map': [action_map] * self.task.n_goals,
            }))
        return pd.concat(results)


class MetaAgent(FlatAgent):

    def __init__(self, task, alpha=1.0, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001, mix_biases=None, update_new_c_only=False):
        if mix_biases is None:
            mix_biases = [0.0, 0.0]

        super(MetaAgent, self).__init__(task)

        self.joint_agent = JointClusteringAgent(
            task, alpha=alpha, gamma=gamma, inv_temp=inv_temp, stop_criterion=stop_criterion,
            mapping_prior=mapping_prior, goal_prior=goal_prior)
        self.independent_agent = IndependentClusterAgent(
            task, alpha=alpha, gamma=gamma, inv_temp=inv_temp, stop_criterion=stop_criterion,
            mapping_prior=mapping_prior, goal_prior=goal_prior)

        self.responsibilities = {'Ind': mix_biases[0], 'Joint': mix_biases[1]}
        self.responsibilities_derivative = {'Ind': 0, 'Joint': 0}
        self.is_mixture = True
        self.choose_operating_model()

        # debugging functions
        self.update_new_c_only = update_new_c_only
        self.visted_contexts = set()

    def augment_assignments(self, context):
        self.joint_agent.augment_assignments(context)
        self.independent_agent.augment_assignments(context)

    def prune_hypothesis_space(self, threshold=50.):
        self.joint_agent.prune_hypothesis_space(threshold)
        self.independent_agent.prune_hypothesis_space(threshold)

    def update_goal_values(self, c, goal, r):
        if not self.update_new_c_only:
            self.evaluate_mixing_agent(c, goal, r)
        else:
            if c not in self.visted_contexts:
                self.evaluate_mixing_agent(c, goal, r)
            self.visted_contexts.add(c)

        self.joint_agent.update_goal_values(c, goal, r)
        self.independent_agent.update_goal_values(c, goal, r)
        self.choose_operating_model()

    def update_mapping(self, c, a, aa):
        self.joint_agent.update_mapping(c, a, aa)
        self.independent_agent.update_mapping(c, a, aa)

    def get_goal_probability(self, context):
        return self.current_agent.get_goal_probability(context)

    def get_mapping_function(self, context, aa):
        return self.current_agent.get_mapping_function(context, aa)

    def select_action(self, location):
        return self.current_agent.select_action(location)

    def get_action_pmf(self, location):
        return self.current_agent.get_action_pmf(location)

    def evaluate_mixing_agent(self, c, goal, r):
        g = self.task.get_goal_index(goal)

        # get the predicted goal value for each model
        ii = np.argmax(self.joint_agent.log_belief)
        h_g = self.joint_agent.goal_hypotheses[ii]
        r_hat_joint = h_g.get_goal_probability(c)

        ii = np.argmax(self.independent_agent.log_belief_goal)
        h_g = self.independent_agent.goal_hypotheses[ii]
        r_hat_ind = h_g.get_goal_probability(c)

        # The map estimate is sensitive to underflow error -- this prevents this be assuming the
        # model has some probability it is wrong (here, hard coded as 1/1000) and bounding the
        # models' probability estimates of reward
        r_hat_j = np.max([0.999 * r_hat_joint[g], 0.001])
        r_hat_i = np.max([0.999 * r_hat_ind[g], 0.001])

        # what is the predicted probability of the observed output for each model? Track the log prob
        self.responsibilities['Joint'] += np.log(r * r_hat_j + (1 - r) * (1 - r_hat_j))
        self.responsibilities['Ind']   += np.log(r * r_hat_i + (1 - r) * (1 - r_hat_i))
        self.responsibilities_derivative['Joint'] = np.log(r * r_hat_j + (1 - r) * (1 - r_hat_j))
        self.responsibilities_derivative['Ind']   = np.log(r * r_hat_i + (1 - r) * (1 - r_hat_i))

    def choose_operating_model(self):
        if np.random.rand() < self.get_joint_probability():
            self.current_agent = self.joint_agent
            self.current_agent_name = 'Joint'
        else:
            self.current_agent = self.independent_agent
            self.current_agent_name = 'Ind'

    def get_joint_probability(self):
        # as an aside, this is an implicit softmax temperature of 1.0  for the RL model
        return np.exp(self.responsibilities['Joint'] - logsumexp(self.responsibilities.values()))

    def get_responsibilities(self):
        return self.responsibilities['Ind'], self.responsibilities['Joint']

    def get_responsibilities_derivative(self):
        return self.responsibilities_derivative['Ind'], self.responsibilities_derivative['Joint']


class RLMetaAgent(MetaAgent):

    def __init__(self, task, alpha=1.0, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001, mix_biases=None, update_new_c_only=False,
                 rl_rate=0.1, rl_beta=1.0):

        super(RLMetaAgent, self).__init__(
            task, alpha=alpha, gamma=gamma, inv_temp=inv_temp, stop_criterion=stop_criterion,
            mapping_prior=mapping_prior, goal_prior=goal_prior, mix_biases=mix_biases,
            update_new_c_only=update_new_c_only
        )

        self.lr = rl_rate
        self.beta = rl_beta
        self.responsibilities = {'Ind': mix_biases[0], 'Joint': mix_biases[1]}

    def evaluate_mixing_agent(self, c, goal, r):
        g = self.task.get_goal_index(goal)

        # get the predicted goal value for each model
        ii = np.argmax(self.joint_agent.log_belief)
        h_g = self.joint_agent.goal_hypotheses[ii]
        r_hat_joint = h_g.get_goal_probability(c)

        ii = np.argmax(self.independent_agent.log_belief_goal)
        h_g = self.independent_agent.goal_hypotheses[ii]
        r_hat_ind = h_g.get_goal_probability(c)

        # what is the predicted probability of the observed output for each model? Track the log prob
        self.responsibilities['Joint'] += self.lr * (r - r_hat_joint[g])
        self.responsibilities['Ind']   += self.lr * (r - r_hat_ind[g])
        self.responsibilities_derivative['Joint'] = self.lr * (r - r_hat_joint[g])
        self.responsibilities_derivative['Ind']   = self.lr * (r - r_hat_ind[g])


class RLMetaAgent_UpdateActive(MetaAgent):

    def __init__(self, task, alpha=1.0, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001, mix_biases=None, update_new_c_only=False):

        super(RLMetaAgent_UpdateActive, self).__init__(
            task, alpha=alpha, gamma=gamma, inv_temp=inv_temp, stop_criterion=stop_criterion,
            mapping_prior=mapping_prior, goal_prior=goal_prior, mix_biases=mix_biases,
            update_new_c_only=update_new_c_only
        )

        self.counts = {'Ind': 0.5, 'Joint': 0.5}
        self.responsibilities = {'Ind': np.exp(mix_biases[0]), 'Joint': np.exp(mix_biases[1])}

    def evaluate_mixing_agent(self, c, goal, r):
        g = self.task.get_goal_index(goal)

        if self.current_agent_name == "Joint":
            self.counts['Joint'] += 1.0
            lr = 1.0 / self.counts['Joint']

            # get the predicted goal value
            ii = np.argmax(self.joint_agent.log_belief)
            h_g = self.joint_agent.goal_hypotheses[ii]
            r_hat_joint = h_g.get_goal_probability(c)
            self.responsibilities['Joint'] += lr * (r - r_hat_joint[g])

            self.responsibilities_derivative['Joint'] = lr * (r - r_hat_joint[g])
            self.responsibilities_derivative['Ind'] = 0

        else:
            self.counts['Ind'] += 1.0
            lr = 1.0 / self.counts['Ind']

            ii = np.argmax(self.independent_agent.log_belief_goal)
            h_g = self.independent_agent.goal_hypotheses[ii]
            r_hat_ind = h_g.get_goal_probability(c)

            # what is the predicted probability of the observed output for each model? Track the log prob
            self.responsibilities['Ind'] += lr * (r - r_hat_ind[g])
            self.responsibilities_derivative['Joint'] = 0
            self.responsibilities_derivative['Ind']   = lr * (r - r_hat_ind[g])


class QLearningAgent(FlatAgent):

    def __init__(self, task, lr=0.1, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001):
        super(FlatAgent, self).__init__(task)

        self.lr = lr
        self.gamma = gamma
        self.inv_temp = inv_temp
        self.stop_criterion = stop_criterion


        # initialize the hypothesis space
        self.mapping_hypotheses = [
            MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
                              1.0, mapping_prior)
        ]

        # initialize the belief spaces
        # self.log_belief_goal = np.ones(1, dtype=float)
        self.log_belief_map = np.ones(1, dtype=float)

        # initialize the q-values for each of the goals as a dictionary
        self.q = dict()
        self.n_goals = self.task.n_goals

    
    def update_goal_values(self, c, goal, r):
        goal_idx_num = self.task.get_goal_index(goal)

        # pull the learned goal-values
        q = self.q[c]

        q[goal_idx_num] += self.lr * (r - q[goal_idx_num])

        # cache the updated function
        self.q[c] = q

    def get_goal_probability(self, context):

        # convert the q-values to goal selection probability
        q = self.q[context]
        goal_pmf = np.exp(q * self.inv_temp - logsumexp(q * self.inv_temp))

        return goal_pmf

    def augment_assignments(self, context):

        # initialize q-values for the context
        if context not in self.q.keys():
            self.q[context] = np.ones(self.n_goals, dtype=float) / float(self.n_goals)

        h_m = self.mapping_hypotheses[0]
        assert type(h_m) is MappingHypothesis
        h_m.add_new_context_assignment(context, context)

        # don't need to update the belief for the flat agent


class KalmanUCBAgent(FlatAgent):

    def __init__(self, task, var_i=0.1, var_e=0.1, ucb_weight=1.0, gamma=0.80, inv_temp=10.0, stop_criterion=0.001,
                 mapping_prior=0.001, goal_prior=0.001):
        super(FlatAgent, self).__init__(task)

        self.var_i = var_i  # innovation gain
        self.var_e = var_e  # 
        self.ucb_weight = ucb_weight
        self.gamma = gamma
        self.inv_temp = inv_temp
        self.stop_criterion = stop_criterion


        # initialize the hypothesis space
        self.mapping_hypotheses = [
            MappingHypothesis(self.task.n_primitive_actions, self.task.n_abstract_actions,
                              1.0, mapping_prior)
        ]

        # initialize the belief spaces
        # self.log_belief_goal = np.ones(1, dtype=float)
        self.log_belief_map = np.ones(1, dtype=float)

        # initialize the q-values for each of the goals as a dictionary
        self.mus = dict()
        self.sigmas = dict()
        self.n_goals = self.task.n_goals

    
    def update_goal_values(self, c, goal, r):
        goal_idx_num = self.task.get_goal_index(goal)

        # pull the learned mus + sigmas
        mus = self.mus[c]
        sigmas = self.sigmas[c]

        # calculate the Kalman Gain
        sigmas = sigmas + self.var_i  # this is the correct update!
        G = (sigmas[goal_idx_num] + self.var_i) / (sigmas[goal_idx_num] + self.var_e + self.var_i)

        mus[goal_idx_num] += G * (r - mus[goal_idx_num])
        sigmas[goal_idx_num] = (1 - G) * (sigmas[goal_idx_num] + self.var_i)

        # cache the updated function
        self.mus[c] = mus
        self.sigmas[c] = sigmas

    def get_goal_probability(self, context):

        # convert the mus and sigmas to q values
        q = self.mus[context] + self.sigmas[context] * self.ucb_weight

        # convert to pmf
        goal_pmf = np.exp(q * self.inv_temp - logsumexp(q * self.inv_temp))
        return goal_pmf

    def augment_assignments(self, context):

        # initialize mu + sigmas for the context
        if context not in self.mus.keys():
            self.mus[context] = np.ones(self.n_goals, dtype=float) / float(self.n_goals)
            self.sigmas[context] = np.ones(self.n_goals, dtype=float)

        h_m = self.mapping_hypotheses[0]
        assert type(h_m) is MappingHypothesis
        h_m.add_new_context_assignment(context, context)

        # don't need to update the belief for the flat agent



class MinimumPathLengthAgent(object):
    """ this agent is not full agent -- it only returns the minimum path length for a predetermined goal, using only
    abstract actions
    """
    
    def __init__(self, task, subject_data, gamma=0.8, stop_criterion=0.001):
        """

        :param task:
        :param subject_data: pandas.DataFrame
        :param gamma:
        :param stop_criterion:
        :return:
        """
        self.task = task
        self.gamma = gamma
        self.stop_criterion = stop_criterion
        self.subject_data = subject_data

        # loop through subject's DataFrame and get the goal selected for each trial
        goals_selected = {}
        for idx in subject_data.index:
            goals_selected[subject_data.loc[idx, 'Trial Number']] = subject_data.loc[idx, 'Chosen Goal']

    def get_abstract_action_policy(self, goal):
        grid = self.task.get_current_gridworld()
        assert type(grid) is GridWorld

        # convert goal to value to probability density
        goal_probability = np.zeros(self.task.n_goals)
        goal_idx = self.task.get_goal_index(goal)
        goal_probability[goal_idx] = 1.0

        # convert the goal values to the lookup table values for states
        reward_function = np.zeros(len(grid.state_location_key))
        for location, goal in grid.goal_locations.iteritems():
            goal_state = grid.state_location_key[location]
            goal_value = goal_probability[self.task.get_goal_index(goal)]
            reward_function[goal_state] = goal_value - 0.1 * (1 - goal_value)

        # use the reward function to make a policy of abstract actions
        transition_function = self.task.get_transition_function()
        pi = policy_iteration(transition_function, reward_function, self.gamma, self.stop_criterion)

        return pi

    def observe(self):
        pass

    def get_shortest_path_length(self, goal, start_location):

        current_gridworld = self.task.get_current_gridworld()
        assert type(current_gridworld) is GridWorld

        path_length = 0
        while True:
            path_length += 1
            pi = self.get_abstract_action_policy(goal)
            s = self.task.state_location_key[start_location]

            # select the best action
            aa = pi[s]

            # invert the action mapping
            inverse_action_map = {dir_: a for a, dir_ in current_gridworld.action_map.iteritems()}
            a = inverse_action_map[current_gridworld.inverse_abstract_action_key[aa]]
            self.task.move(a)
            if self.task.goal_check():
                break
        return path_length
