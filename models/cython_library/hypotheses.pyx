# cython: profile=False, linetrace=False, boundscheck=False, wraparound=False
# distutils: language=c++
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

from core import get_prior_log_probability

from libcpp.vector cimport vector

DTYPE = np.float
ctypedef np.float_t DTYPE_t

INT_DTYPE = np.int32
ctypedef np.int32_t INT_DTYPE_t

cdef extern from "math.h":
    double log(double x)

cdef class MappingCluster(object):
    cdef double [:,::1] mapping_history, mapping_mle, pr_aa_given_a, log_pr_aa_given_a
    cdef double [:] abstract_action_counts, primitive_action_counts
    cdef int n_primitive_actions, n_abstract_actions
    cdef double mapping_prior

    def __init__(self, int n_primitive_actions, int n_abstract_actions, float mapping_prior):

        cdef double[:, ::1] mapping_history, mapping_mle, pr_aa_given_a, log_pr_aa_given_a
        cdef double[:] abstract_action_counts, primitive_action_counts
        cdef int a, aa

        mapping_history = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE)
        abstract_action_counts = np.ones(n_abstract_actions+1, dtype=float)
        mapping_mle = np.ones((n_primitive_actions, n_abstract_actions + 1),  dtype=DTYPE)

        primitive_action_counts = np.ones(n_primitive_actions, dtype=DTYPE)
        pr_aa_given_a = np.ones((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE)
        log_pr_aa_given_a = np.zeros((n_primitive_actions, n_abstract_actions + 1), dtype=DTYPE)


        cdef double inv_n_a = 1.0/n_primitive_actions
        cdef double inv_n_aa = 1.0/n_abstract_actions
        for a in range(n_primitive_actions):
            for aa in range(n_abstract_actions + 1):
                mapping_history[a, aa] = mapping_prior
                mapping_mle[a, aa] = inv_n_a
                pr_aa_given_a[a, aa] = inv_n_aa

        cdef double mp_X_naa = mapping_prior * n_abstract_actions
        for a in range(n_primitive_actions):
            primitive_action_counts[a] = mp_X_naa

        cdef double mp_X_na = mapping_prior * n_primitive_actions
        for aa in range(n_abstract_actions + 1):
            abstract_action_counts[aa] = mp_X_na

        self.mapping_history = mapping_history
        self.abstract_action_counts = abstract_action_counts
        self.mapping_mle = mapping_mle
        self.primitive_action_counts = primitive_action_counts
        self.pr_aa_given_a = pr_aa_given_a
        self.log_pr_aa_given_a = log_pr_aa_given_a

        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions
        self.mapping_prior = mapping_prior

    def update(self, int a, int aa):
        cdef int aa0, a0
        cdef float p, mh, aa_count

        self.mapping_history[a, aa] += 1.0
        self.abstract_action_counts[aa] += 1.0
        self.primitive_action_counts[a] += 1.0

        for aa0 in range(self.n_abstract_actions):
            aa_count = self.abstract_action_counts[aa0]

            for a0 in range(self.n_primitive_actions):
                mh = self.mapping_history[a0, aa0]
                self.mapping_mle[a0, aa0] =  mh / aa_count

                # p(A|a, k) estimator
                p = mh / self.primitive_action_counts[a0]
                self.pr_aa_given_a[a0, aa0] = p
                self.log_pr_aa_given_a[a0, aa0] = log(p)

    def get_mapping_mle(self, int a, int aa):
        return self.mapping_mle[a, aa]

    def get_likelihood(self, int a, int aa):
        return self.pr_aa_given_a[a, aa]

    def get_log_likelihood(self, int a, int aa):
        return self.log_pr_aa_given_a[a, aa]

    def get_log_likelihood_function(self):
        return self.log_pr_aa_given_a

    def deep_copy(self):
        cdef int a, aa, idx, n_aa_w

        cdef MappingCluster _cluster_copy = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                                           self.mapping_prior)

        n_aa_w = self.n_abstract_actions + 1 # include the possibility of the "wait" action

        for a in range(self.n_primitive_actions):

            _cluster_copy.primitive_action_counts[a] = self.primitive_action_counts[a]

            for aa in range(n_aa_w):
                _cluster_copy.mapping_history[a, aa] = self.mapping_history[a, aa]
                _cluster_copy.mapping_mle[a, aa] = self.mapping_mle[a, aa]
                _cluster_copy.pr_aa_given_a[a, aa] = self.pr_aa_given_a[a, aa]

        for aa in range(n_aa_w):
            _cluster_copy.abstract_action_counts[aa] = self.abstract_action_counts[aa]

        return _cluster_copy


cdef class MappingHypothesis(object):

    cdef dict cluster_assignments, clusters
    cdef double prior_log_prob, alpha, mapping_prior
    cdef vector[int] experience_k
    cdef vector[int] experience_a
    cdef vector[int] experience_aa
    cdef int n_abstract_actions, n_primitive_actions, t
    cdef list visited_clusters

    def __init__(self, int n_primitive_actions, int n_abstract_actions,
                 float alpha, float mapping_prior):

        self.cluster_assignments = dict()
        self.n_primitive_actions = n_primitive_actions
        self.n_abstract_actions = n_abstract_actions
        self.alpha = alpha
        self.mapping_prior = mapping_prior

        # initialize the clusters
        self.clusters = dict()

        # store the prior probability
        self.prior_log_prob = 0.0

        # need to store all experiences for log probability calculations
        cdef vector[int] experience_k = []
        cdef vector[int] experience_a = []
        cdef vector[int] experience_aa = []

        self.experience_k = experience_k
        self.experience_a = experience_a
        self.experience_aa = experience_aa
        self.t = 0
        self.visited_clusters = []


    def update_mapping(self, int c, int a, int aa):
        cdef int k = self.cluster_assignments[c]
        cdef MappingCluster cluster = self.clusters[k]

        cluster.update(a, aa)
        self.clusters[k] = cluster

        # need to store all experiences for log probability calculations
        self.experience_k.push_back(k)
        self.experience_a.push_back(a)
        self.experience_aa.push_back(aa)
        self.t += 1

        if k not in self.visited_clusters:
            self.visited_clusters.append(k)

    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef unsigned int k, k0, a, aa, t
        cdef MappingCluster cluster
        cdef double [:, ::1] ll_func

        #loop through experiences and get posterior
        for k in self.visited_clusters:

            # pre-cache cluster lookup b/c it is slow
            cluster = self.clusters[k]
            ll_func = cluster.get_log_likelihood_function()

            # now loop through and only pull the values for the current clusters
            t = 0
            while t < self.t:
                k0 = self.experience_k[t]
                if k == k0:
                    a = self.experience_a[t]
                    aa = self.experience_aa[t]
                    log_likelihood += ll_func[a, aa]
                t += 1

        return log_likelihood

    def get_log_posterior(self):
        return self.prior_log_prob + self.get_log_likelihood()

    def get_mapping_probability(self, int c, int a, int aa):
        cdef MappingCluster cluster = self.clusters[self.cluster_assignments[c]]
        return cluster.get_mapping_mle(a, aa)

    def get_log_prior(self):
        return self.prior_log_prob

    def deep_copy(self):
        cdef MappingHypothesis _h_copy = MappingHypothesis(
            self.n_primitive_actions, self.n_abstract_actions, self.alpha,
            self.mapping_prior
        )

        cdef int k, c
        cdef MappingCluster cluster

        _h_copy.cluster_assignments = {c: k for c, k in self.cluster_assignments.iteritems()}
        _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
        _h_copy.prior_log_prob = get_prior_log_probability(_h_copy.cluster_assignments, _h_copy.alpha)
        _h_copy.visited_clusters = [k for k in self.visited_clusters]
        _h_copy.t = self.t
        for t in range(self.t):
            _h_copy.experience_k.push_back(self.experience_k[t])
            _h_copy.experience_a.push_back(self.experience_a[t])
            _h_copy.experience_aa.push_back(self.experience_aa[t])

        return _h_copy

    def get_assignments(self):
        return self.cluster_assignments

    def add_new_context_assignment(self, int c, int k):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.cluster_assignments.values():
            # if not, add an new reward cluster
            self.clusters[k] = MappingCluster(self.n_primitive_actions, self.n_abstract_actions,
                                              self.mapping_prior)

        self.cluster_assignments[c] = k
        self.prior_log_prob = get_prior_log_probability(self.cluster_assignments, self.alpha)

cdef class GoalCluster(object):
    cdef int n_goals
    cdef double set_visits, goal_prior
    cdef double [:]  goal_rewards_received, goal_reward_probability

    def __init__(self, int n_goals, float goal_prior):
        self.n_goals = n_goals
        self.goal_prior = goal_prior

        # rewards!
        cdef double [:] goal_rewards_received = np.ones(n_goals) * goal_prior
        cdef double [:] goal_reward_probability = np.ones(n_goals) * (1.0 / n_goals)

        self.set_visits =  n_goals * goal_prior
        self.goal_rewards_received = goal_rewards_received
        self.goal_reward_probability = goal_reward_probability

    def update(self, int goal, int r):
        cdef double r0
        cdef int g0

        self.set_visits += 1.0
        self.goal_rewards_received[goal] += r

        if r == 0:
            r0 = 1.0 / (self.n_goals - 1)
            for g0 in range(self.n_goals):
                if g0 != goal:
                    self.goal_rewards_received[g0] += r0

        # update all goal probabilities
        for g0 in range(self.n_goals):
            self.goal_reward_probability[g0] = self.goal_rewards_received[g0] / self.set_visits

    def is_visited(self):
        return self.set_visits >= 1

    def get_observation_probability(self, int goal, int r):
        if r == 0:
            return 1 - self.goal_reward_probability[goal]
        return self.goal_reward_probability[goal]

    def get_goal_pmf(self):
        return self.goal_reward_probability

    def deep_copy(self):
        cdef int g

        cdef GoalCluster _cluster_copy = GoalCluster(self.n_goals, self.goal_prior)

        _cluster_copy.set_visits = self.set_visits

        for g in range(self.n_goals):
            _cluster_copy.goal_rewards_received[g] = self.goal_rewards_received[g]
            _cluster_copy.goal_reward_probability[g] = self.goal_reward_probability[g]

        return _cluster_copy


cdef class GoalHypothesis(object):
    cdef int n_goals, t
    cdef double log_prior, alpha, goal_prior
    cdef dict cluster_assignments, clusters
    cdef vector[int] experience_k
    cdef vector[int] experience_g
    cdef vector[int] experience_r

    def __init__(self, int n_goals, float alpha, float goal_prior):

        cdef dict cluster_assignments = dict()
        cdef dict clusters = dict()
        cdef list experience = list()

        self.n_goals = n_goals
        self.cluster_assignments = cluster_assignments
        self.alpha = alpha
        self.goal_prior = goal_prior

        # initialize goal clusters
        self.clusters = clusters

        # initialize posterior
        self.log_prior = 1.0

        cdef vector[int] experience_k = []
        cdef vector[int] experience_g = []
        cdef vector[int] experience_r = []

        self.experience_k = experience_k
        self.experience_g = experience_g
        self.experience_r = experience_r
        self.t = 0

    def update(self, int c, int goal, int r):
        cdef int k = self.cluster_assignments[c]
        cdef GoalCluster cluster = self.clusters[k]
        cluster.update(goal, r)
        self.clusters[k] = cluster

        self.experience_k.push_back(k)
        self.experience_g.push_back(goal)
        self.experience_r.push_back(r)
        self.t += 1

    def get_log_likelihood(self):
        cdef double log_likelihood = 0
        cdef int k, g, r
        cdef GoalCluster cluster

        #loop through experiences and get posterior
        for t in range(self.t):
            k = self.experience_k[t]
            g = self.experience_g[t]
            r = self.experience_r[t]
            cluster = self.clusters[k]
            log_likelihood += log(cluster.get_observation_probability(g, r))

        return log_likelihood

    def get_log_posterior(self):
        return self.get_log_likelihood() + self.log_prior

    def get_log_prior(self):
        return self.log_prior

    def get_goal_probability(self, int c):
        cdef int g, k

        k = self.cluster_assignments[c]
        cdef GoalCluster cluster = self.clusters[k]
        cdef np.ndarray[DTYPE_t, ndim=1] goal_probability = np.zeros(self.n_goals, dtype=DTYPE)

        cdef double [:] rew_func = cluster.get_goal_pmf()
        for g in range(self.n_goals):
            goal_probability[g] = rew_func[g]
        return goal_probability


    def deep_copy(self):
        cdef GoalHypothesis _h_copy = GoalHypothesis(self.n_goals, self.alpha, self.goal_prior)

        cdef int c, k, t
        cdef GoalCluster cluster

        _h_copy.cluster_assignments = {c: k for c, k in self.cluster_assignments.iteritems()}
        _h_copy.clusters = {k: cluster.deep_copy() for k, cluster in self.clusters.iteritems()}
        _h_copy.log_prior = get_prior_log_probability(_h_copy.cluster_assignments, _h_copy.alpha)

        _h_copy.t = self.t
        for t in range(self.t):
            _h_copy.experience_k.push_back(self.experience_k[t])
            _h_copy.experience_g.push_back(self.experience_g[t])
            _h_copy.experience_r.push_back(self.experience_r[t])

        return _h_copy

    def get_assignments(self):
        return self.cluster_assignments

    def add_new_context_assignment(self, int c, int k):

        # check if cluster "k" is already been assigned new cluster
        if k not in self.cluster_assignments.values():
            # if not, add an new reward cluster
            self.clusters[k] = GoalCluster(self.n_goals, self.goal_prior)

        self.cluster_assignments[c] = k  # note, there's no check built in here
        self.log_prior = get_prior_log_probability(self.cluster_assignments, self.alpha)

    def is_visited(self, int c):
        cdef int k = self.cluster_assignments[c]
        cdef GoalCluster cluster = self.clusters[k]
        return cluster.is_visited()