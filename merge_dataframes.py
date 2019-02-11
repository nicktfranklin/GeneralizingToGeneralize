import pandas as pd

def main(tag):
    # # load experiment 2 goals
    print "Loading Experiment 1"
    sims_exp_2 = []
    for m in 'joint indep meta flat'.split():
        df = pd.read_pickle('exp_2_goals_batch_of_sims_{}{}.pkl'.format(m, tag))
        df.drop(['Start Location', 'End Location', 'Goal Locations', 'Action Map', 'Walls'], axis=1, inplace=True)
        df = df[df['In Goal']]
        sims_exp_2.append(df)
    df = None
    sims_exp_2 = pd.concat(sims_exp_2, sort=False)
    sims_exp_2.to_pickle('exp_2_goals_batch_of_sims{}.pkl'.format(tag))
    sims_exp_2 = None

    # load experiment 4 goals
    print "Loading Experiment 2"
    sims_exp_4 = []
    for m in 'joint indep meta flat'.split():
        df = pd.read_pickle('exp_4_goals_batch_of_sims_{}{}.pkl'.format(m, tag))
        df.drop(['Start Location', 'End Location', 'Goal Locations', 'Action Map', 'Walls'], axis=1, inplace=True)
        df = df[df['In Goal']]
        sims_exp_4.append(df)
    df = None
    sims_exp_4 = pd.concat(sims_exp_4, sort=False)
    sims_exp_4.to_pickle('exp_4_goals_batch_of_sims{}.pkl'.format(tag))
    sims_exp_4 = None


    # load experiment 3 goals
    print "Loading Experiment 3"
    sims_exp_3 = []
    for m in 'joint indep meta flat'.split():
        df = pd.read_pickle('exp_3_goals_batch_of_sims_{}{}.pkl'.format(m, tag))
        df.drop(['Start Location', 'End Location', 'Goal Locations', 'Action Map', 'Walls'], axis=1, inplace=True)
        df = df[df['In Goal']]
        sims_exp_3.append(df)
    df = None
    sims_exp_3 = pd.concat(sims_exp_3, sort=False)
    sims_exp_3.to_pickle('exp_3_goals_batch_of_sims{}.pkl'.format(tag))
    sims_exp_3 = None
    print "Done!"

if __name__ == "__main__":
    tag = '_update_all_trials_gp=1e-5_prune=100_mu=0.0_scale=1.0'
    main(tag)

    # tag = '_rl_active_only_gp=1e-10_prune=100_mu=0.0'
    # main(tag)
    #
    # tag = '_rl_active_only_gp=1e-10_prune=100_mu=-1.0'
    # main(tag)