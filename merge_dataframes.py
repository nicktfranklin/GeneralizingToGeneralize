import pandas as pd

def main(tag):
    # load experiment 2 goals
    sims_exp_2 = pd.concat([
        pd.read_pickle('exp_2_goals_batch_of_sims_joint{}.pkl'.format(tag)),
        pd.read_pickle('exp_2_goals_batch_of_sims_indep{}.pkl'.format(tag)),
        pd.read_pickle('exp_2_goals_batch_of_sims_meta{}.pkl'.format(tag)),
        pd.read_pickle('exp_2_goals_batch_of_sims_flat{}.pkl'.format(tag)),
    ], sort=False)
    sims_exp_2.to_pickle('exp_2_goals_batch_of_sims_{}.pkl'.format(tag))
    sims_exp_2 = None

    # load experiment 4 goals
    sims_exp_4 = pd.concat([
        pd.read_pickle('exp_4_goals_batch_of_sims_joint{}.pkl'.format(tag)),
        pd.read_pickle('exp_4_goals_batch_of_sims_indep{}.pkl'.format(tag)),
        pd.read_pickle('exp_4_goals_batch_of_sims_meta{}.pkl'.format(tag)),
        pd.read_pickle('exp_4_goals_batch_of_sims_flat{}.pkl'.format(tag)),
    ], sort=False)
    sims_exp_4.to_pickle('exp_4_goals_batch_of_sims_{}.pkl'.format(tag))
    sims_exp_4 = None


    # load experiment 3 goals
    sims_exp_3 = pd.concat([
        pd.read_pickle('exp_3_goals_batch_of_sims_joint{}.pkl'.format(tag)),
        pd.read_pickle('exp_3_goals_batch_of_sims_indep{}.pkl'.format(tag)),
        pd.read_pickle('exp_3_goals_batch_of_sims_meta{}.pkl'.format(tag)),
        pd.read_pickle('exp_3_goals_batch_of_sims_flat{}.pkl'.format(tag)),
    ], sort=False)
    sims_exp_3.to_pickle('exp_3_goals_batch_of_sims_{}.pkl'.format(tag))
    sims_exp_4 = None

if __name__ == "__main__":
    tag = '_update_all_trials_gp=1e-10_prune=100_mu=-0.5'
    main(tag)