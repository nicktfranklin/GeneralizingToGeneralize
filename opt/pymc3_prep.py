import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from tqdm import tqdm


def make_group_model(proc_data, goal_key=None, importance_samples=500, exclude_train=0):

    n_goals = len(set(proc_data[u'Chosen Goal']))
    n_trials = int(max(proc_data[u'Trial Number']) + 1)
    n_contexts = len(set(proc_data.Context))

    if goal_key is None:
        # doesn't change across experiements, max is 4 anyway
        goal_key = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def make_hash(dict_):
        """ Make a hash out of a dictionary"""
        hash_ = ''
        for k, v in dict_.iteritems():
            hash_ += str(k) + str(v)
        return hash_

    def make_hash(dict_):
        hash_ = ''
        for k, v in dict_.iteritems():
            hash_ += str(k) + str(v)
        return hash_

    def make_regressors(data_df):
        df = data_df.loc[data_df['In Goal'], :]

        choice = np.zeros(n_trials)
        win_stay = np.zeros((n_trials, n_goals))
        r_counts = np.zeros((n_trials, 1)) + n_goals * 1e-5
        r_points = np.zeros((n_trials, n_goals)) + 1e-5

        correct_goals = {}
        goal_pop = np.zeros((n_trials, n_goals))

        correct_goals_map = {
            am: {} for am in set(make_hash(df.loc[idx, 'Action Map']) for idx in df.index)
        }
        map_id = {am: ii for ii, am in enumerate(correct_goals_map.keys())}
        goal_map_pop = np.zeros((n_trials, n_goals))

        r_m_counts = np.zeros(len(map_id)) + n_goals * 1e-5
        r_m_points = np.zeros((len(map_id), n_goals)) + 1e-5
        avg_map_reward = np.zeros((n_trials, n_goals))

        for ii, idx in enumerate(df.index[:-1]):
            g = goal_key[df.loc[idx, 'Chosen Goal']]
            r = df.loc[idx, u'Reward']
            c = df.loc[idx, 'Context']
            am = make_hash(df.loc[idx, 'Action Map'])
            m = map_id[am]

            choice[ii] = g

            # get win stay, loose shift
            win_stay[ii + 1, g] = (r - 5.0) / 5.0

            # get average reward over all trials!
            r_counts[ii + 1] += 1.0
            r_points[ii + 1, g] = r / 10.0

            # get average reward within over all trials within a map!
            avg_map_reward[ii, :] = r_m_points[m, :] / r_m_counts[m]
            r_m_counts[m] += 1.0
            r_m_points[m, g] = r / 10.0

            # get the goal popularity overall all contexts!
            if r > 0:
                correct_goals[c] = g
                correct_goals_map[am][c] = g

            _df_temp = correct_goals
            for g0, p in {g0: _df_temp.values().count(g0) for g0 in list(set(_df_temp.values()))}.iteritems():
                goal_pop[ii + 1, g0] = p

            _df_temp = correct_goals_map[am]
            for g0, p in {g0: _df_temp.values().count(g0) for g0 in list(set(_df_temp.values()))}.iteritems():
                goal_map_pop[ii + 1, g0] = p

        idx = df.index[-1]
        g = goal_key[df.loc[idx, 'Chosen Goal']]
        choice[-1] = g

        # actually calculate the rewards...
        avg_reward = r_points / np.tile(r_counts, (1, n_goals))

        return choice, goal_pop, goal_map_pop, win_stay, avg_reward, avg_map_reward

    # for a single subject, fit a reward model and get the expected Q-values
    def logit(x):
        return np.log(x / (1. - x))

    def inv_logit(x):
        return 1.0 / (1.0 + np.exp(-x))

    def make_q(subj_data, n_samples=importance_samples):
        contexts = subj_data.Context.values.astype('int') - 1
        rewards = subj_data.Reward.values.astype('int') / 10
        goals = [goal_key[g] for g in subj_data['Chosen Goal'].values]

        # one hot the goal choices
        goal_one_hot = np.zeros((n_trials, n_goals))
        for ii, g in enumerate(goals):
            goal_one_hot[ii, g] = 1.0

        def rl_simple(lr):
            q_c = np.ones((n_contexts, n_goals)) * 1. / n_goals  # n_trials and n_goals should be globals
            q_t = np.zeros((n_trials, n_goals))
            for ii, (c, g, r) in enumerate(zip(contexts, goals, rewards)):
                # cache predictor
                q_t[ii, :] = q_c[c, :]

                # update
                q_c[c, g] += lr * (r - q_c[c, g])

            return q_t

        def two_param_rl(beta, alpha):
            # get the values
            V = rl_simple(alpha)

            # get the log likelihood
            log_p = beta * V - np.tile(logsumexp(beta * V, axis=1), (n_goals, 1)).T

            # use one hot to select the LL for the chosen options
            ll = np.sum(log_p * goal_one_hot, axis=1)

            # return the total loglikelihood
            return logsumexp(ll)

        # parameters for the proposal distribution
        mu_a = logit(0.1)
        std_a = 1.5
        mu_b = 0.0
        std_b = 1.25

        def importance_sampler(n_samples):

            # only care about the alpha values here
            a_samples = np.zeros(n_samples)
            b_samples = np.zeros(n_samples)
            weights = np.zeros(n_samples)

            for ii in range(n_samples):
                # draw a random sample of the parameters
                a = np.random.normal(loc=mu_a, scale=std_a)
                b = np.random.normal(loc=mu_b, scale=std_b)

                # get the proposal probability
                g = norm.logpdf(a, mu_a, scale=std_a) + norm.logpdf(b, mu_b, scale=std_b)

                # get the likelihood
                f = two_param_rl(np.exp(b), inv_logit(a))

                weights[ii] = f - g
                a_samples[ii] = inv_logit(a)
                b_samples[ii] = np.exp(b)

            if any(np.isnan(weights)):
                print "Invalid importance Weight!"
                for w1, a1, b1 in zip(weights, a_samples, b_samples):
                    print w1, a1, b1
                raise(Exception)

            weights -= logsumexp(weights)

            return np.exp(weights), a_samples

        w, a = importance_sampler(n_samples)

        # get the expectation of the Q-values
        Q = np.zeros((n_trials, n_goals))
        for w0, a0 in zip(w, a):
            Q += rl_simple(a0) * w0

        return Q

    # simple WM model
    # for a single subject, fit a reward model and get the expected Q-values
    def make_wmQ(subj_data, n_samples=importance_samples):
        contexts = subj_data.Context.values.astype('int') - 1
        rewards = subj_data.Reward.values.astype('int') / 10
        goals = [goal_key[g] for g in subj_data['Chosen Goal'].values]

        # one hot the goal choices
        goal_one_hot = np.zeros((n_trials, n_goals))
        for ii, g in enumerate(goals):
            goal_one_hot[ii, g] = 1.0

        def wm_simple(wm_decay):
            q_c = np.ones((n_contexts, n_goals)) * 1. / n_goals  # n_trials and n_goals should be globals
            q_t = np.zeros((n_trials, n_goals))
            for ii, (c, g, r) in enumerate(zip(contexts, goals, rewards)):
                # cache predictor
                q_t[ii, :] = q_c[c, :]

                # update
                q_c[c, g] = r

                # decay
                q_c[c, g] *= wm_decay

            return q_t

        def two_param_wm(beta, gamma):
            # get the values
            V = wm_simple(gamma)

            # get the log likelihood
            log_p = beta * V - np.tile(logsumexp(beta * V, axis=1), (n_goals, 1)).T

            # use one hot to select the LL for the chosen options
            ll = np.sum(log_p * goal_one_hot, axis=1)

            # return the total loglikelihood
            return logsumexp(ll)

        # parameters for the proposal distribution
        mu_a = logit(0.1)
        std_a = 1.5
        mu_b = 0.0
        std_b = 1.25

        def importance_sampler(n_samples):

            # only care about the alpha values here
            a_samples = np.zeros(n_samples)
            b_samples = np.zeros(n_samples)
            weights = np.zeros(n_samples)

            for ii in range(n_samples):
                # draw a random sample of the parameters
                a = np.random.normal(loc=mu_a, scale=std_a)
                b = np.random.normal(loc=mu_b, scale=std_b)

                # get the proposal probability
                g = norm.logpdf(a, mu_a, scale=std_a) + norm.logpdf(b, mu_b, scale=std_b)

                # get the likelihood
                f = two_param_wm(np.exp(b), inv_logit(a))

                weights[ii] = f - g
                a_samples[ii] = inv_logit(a)
                b_samples[ii] = np.exp(b)

            if any(np.isnan(weights)):
                print "Invalid importance Weight!"
                for w1, a1, b1 in zip(weights, a_samples, b_samples):
                    print w1, a1, b1
                raise(Exception)
            weights -= logsumexp(weights)

            return np.exp(weights), a_samples

        w, a = importance_sampler(n_samples)

        # get the expectation of the Q-values
        Q = np.zeros((n_trials, n_goals))
        for w0, a0 in zip(w, a):
            Q += wm_simple(a0) * w0

        return Q

    # prep all of the subjects
    y = []
    X_ind = []
    X_joint = []
    X_ws = []
    X_rew = []
    X_rew_map = []
    X_Q = []
    X_wm = []
    subj_idx = []

    for sub in tqdm(set(proc_data.subj)):
        subj_data = proc_data.loc[proc_data.subj == sub, :]
        _y, _X_ind, _X_joint, _X_ws, _X_rew, _X_rew_map = make_regressors(data_df=subj_data)
        _q = make_q(subj_data)
        _qwm = make_wmQ(subj_data)

        if exclude_train > 0:
            train = subj_data.Context < exclude_train

            _y         = _y[~train]
            _X_ind     = _X_ind[~train, :]
            _X_joint   = _X_joint[~train, :]
            _X_ws      = _X_ws[~train, :]
            _X_rew     = _X_rew[~train, :]
            _X_rew_map = _X_rew_map[~train, :]
            _q         = _q[~train, :]
            _qwm       = _qwm[~train, :]

        y.append(_y)
        X_ind.append(_X_ind)
        X_joint.append(_X_joint)
        X_ws.append(_X_ws)
        X_rew.append(_X_rew)
        X_rew_map.append(_X_rew_map)
        subj_idx.append([sub] * len(_y))

        X_Q.append(_q)
        X_wm.append(_qwm)

    # all of these predictors are untransformed
    return {
        'Goal-Choice-Id': np.concatenate(y),
        'Prior-Independent': np.concatenate(X_ind),
        'Prior-Joint': np.concatenate(X_joint),
        'Win-Stay': np.concatenate(X_ws),
        'Avg-Reward-all-ctx': np.concatenate(X_rew),
        'Avg-Reward-map-ctx': np.concatenate(X_rew_map),
        'Q-Values': np.concatenate(X_Q),
        'WM-Value': np.concatenate(X_wm),
        'Subjects': np.concatenate(subj_idx)
    }