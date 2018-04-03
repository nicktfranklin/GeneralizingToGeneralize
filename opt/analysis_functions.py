import numpy as np
from tqdm import tqdm
import pandas as pd

def get_closest_goals(uids, data_frame):
    # what is the frequency the subject selected the closest goal (manhattan distance)?
    closest_goal_df = [None] * (len(uids) * (int(data_frame['Trial Number'].max()) + 1))
    p = 0
    pbar = tqdm(total=len(uids))
    for uid in uids:
        subj_df = data_frame[data_frame.subj == uid]
        n_trials = int(subj_df['Trial Number'].max() + 1)
        prev_correct = False
        prev_context = -1
        for t in range(n_trials):
            subj_trial = subj_df[subj_df['Trial Number'] == t]

            goal_locations = subj_trial.loc[subj_trial.index[0], 'Goal Locations']
            start_location = subj_trial.loc[subj_trial.index[0], 'Start Location']

            # get the manhattan distance to each goal
            manhattan_distances = dict()
            for loc, goal_id in goal_locations.iteritems():
                manhattan_distances[goal_id] = abs(loc[0] - start_location[0]) + abs(loc[1] - start_location[1])

            # did the subject select the closest goal?
            chosen_goal = subj_trial.loc[subj_trial.index[-1], 'Chosen Goal']
            chose_closet_goal = manhattan_distances[chosen_goal] == \
                                np.min(manhattan_distances.values())

            closest_goal_df[p] = pd.DataFrame(
                {
                    'Chose Closet Goal': chose_closet_goal,
                    'subj': uid,
                    'Trial Number': t,
                    'Context': subj_trial.loc[subj_trial.index[0], 'Context'],
                    'Context Repeated': subj_trial.loc[subj_trial.index[0], 'Context'] == prev_context,
                    'Previously Correct': prev_correct,
                    'Times Seen Context': subj_trial.loc[subj_trial.index[0], 'Times Seen Context'],
                    'Accuracy': subj_trial.loc[subj_trial.index[-1], 'Reward'] / 10.
                }, index=[p])
            p += 1
            prev_correct = bool(subj_trial.loc[subj_trial.index[-1], 'Reward'] / 10.0)
            prev_context = subj_trial.loc[subj_trial.index[0], 'Context']
        pbar.update()
    pbar.close()

    return pd.concat(closest_goal_df)

def generate_exclusion_list(processed_data, training_contexts=5, goal_chance=1/3.,
                            binom=True, return_measures=False):

    from scipy.stats import binom_test
    from sklearn import mixture

    closest_goal_df = get_closest_goals(set(processed_data.subj), processed_data)

    vec = (closest_goal_df['Context Repeated']) & (closest_goal_df['Context'] <= training_contexts)
    X = closest_goal_df[vec].groupby('subj').mean()
    vec = closest_goal_df['Previously Correct'] & closest_goal_df['Context Repeated']
    X_repeat_ac = closest_goal_df[vec].groupby('subj').mean()
    X['Accuracy in Repeated Trials'] = X_repeat_ac.Accuracy

    vec = ~closest_goal_df['Previously Correct'] | ~closest_goal_df['Context Repeated']
    X_nrepeat_ac = closest_goal_df[vec].groupby('subj').mean()
    X['Accuracy in Non-Repeated Trials'] = X_nrepeat_ac.Accuracy

    closest_goal_df['TrialCounter'] = [1] * len(closest_goal_df)
    vec = (~closest_goal_df['Previously Correct'] | ~closest_goal_df['Context Repeated']) & (closest_goal_df.Context < 7)
    X0 = closest_goal_df[vec].groupby('subj').sum()

    for idx in X0.index:
        n_successes = int(X0.loc[idx, 'Accuracy'])
        k_trys = int(X0.loc[idx, 'TrialCounter'])
        X.loc[idx, 'Binomial Chance Probability (Training)'] = binom_test(n_successes, k_trys, goal_chance)

    if binom:
        cluster_data = np.asarray([
            list(X['Accuracy in Repeated Trials']),
            list(X['Binomial Chance Probability (Training)']),
        ]).T
    else:
        cluster_data = np.asarray([
            list(X['Accuracy in Repeated Trials']),
            list(X['Accuracy in Non-Repeated Trials']),
        ]).T

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 3)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(cluster_data)
            bic.append(gmm.bic(cluster_data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    clst = best_gmm
    cluster_ID = clst.predict(cluster_data)
    X['Cluster'] = cluster_ID

    # get the cluster to keep
    max_clust = -1
    max_clust_members = 0
    for c in set(X.Cluster):
        if np.sum(X.Cluster == c) > max_clust_members:
            max_clust_members = np.sum(X.Cluster == c)
            max_clust = c

    X['Group'] = 'Excluded'
    X.loc[X.Cluster == max_clust, 'Group'] = "Included"

    if return_measures:
        # return X[['Accuracy in Repeated Trials', 'Accuracy in Non-Repeated Trials',
        #           'Binomial Chance Probability (Training)', 'Group']]
        return X

    return X[X.Group == 'Excluded'].index.tolist()

