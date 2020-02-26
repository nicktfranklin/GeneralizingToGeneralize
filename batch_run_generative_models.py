import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm

from models.grid_world import Experiment
from models.agents import IndependentClusterAgent, JointClusteringAgent, FlatAgent, MetaAgent
from models.agents import QLearningAgent, KalmanUCBAgent, NoCTX_QLearningAgent
from models.experiment_designs.experiment2 import gen_task_param as gen_task_param_exp_4_goals
from models.experiment_designs.experiment1 import gen_task_param as gen_task_param_exp_3_goals
from models.experiment_designs.experiment_3a import gen_task_param as gen_task_param_exp_2_goals_a
from models.experiment_designs.experiment_3b import gen_task_param as gen_task_param_exp_2_goals_b

def logit(x):
    return 1 / (1 + np.exp(-x))

def batch_exp_2_goals(seed=0, n_sims=1000, alpha_mu=0.0, alpha_scale=1.0, goal_prior=0.001, mapping_prior=0.001,
                      updated_new_c_only=False, pruning_threshold=10.0, tag='', beta_mu=2.0, beta_scale=1.0):

    # alpha is sample from the distribution
    # log(alpha) ~ N(alpha_mu, alpha_scale)

    evaluate = False

    # pre generate a set of tasks for consistency.
    list_tasks_a = [gen_task_param_exp_2_goals_a() for _ in range(n_sims)]
    list_tasks_b = [gen_task_param_exp_2_goals_b() for _ in range(n_sims)]

    # pre draw the alphas for consistency
    list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale))
                  for _ in range(n_sims + n_sims)]

    list_beta = [np.exp(scipy.random.normal(loc=beta_mu, scale=beta_scale))
                  for _ in range(n_sims + n_sims)]

    
    list_lr = logit(np.random.normal(-1, 1.0, n_sims*2))

    list_var_e = logit(np.random.normal(-1, 1.0, n_sims*2))
    list_var_i = logit(np.random.normal(-1, 1.0, n_sims*2))
    list_ucb_w = logit(np.random.normal(-1, 1.0, n_sims*2))

    def sim_agent(AgentClass, name='None', flat=False, meta=False):
        np.random.seed(seed)

        tt = 0

        results = []
        for task, list_task in zip(['A', 'B'], [list_tasks_a, list_tasks_b]):

            for ii, (task_args, task_kwargs) in tqdm(enumerate(list_task), total=len(list_task)):
                if not flat:
                    agent_kwargs = dict(alpha=list_alpha[tt], inv_temp=list_beta[tt],
                                        goal_prior=goal_prior, mapping_prior=mapping_prior)
                else:
                    agent_kwargs = dict(inv_temp=list_beta[tt], goal_prior=goal_prior)

                if meta:
                    p = np.random.uniform(0.0, 1.00)
                    agent_kwargs['mix_biases'] = [np.log(p), np.log(1 - p)]
                    agent_kwargs['update_new_c_only'] = updated_new_c_only

                agent = AgentClass(Experiment(*task_args, **task_kwargs), **agent_kwargs)

                _res = None
                while _res is None:
                    _res = agent.generate(evaluate=evaluate, pruning_threshold=pruning_threshold)

                _res[u'Model'] = name
                _res[u'Iteration'] = [tt] * len(_res)
                _res[u'Task'] = [task] * len(_res)
                results.append(_res)
                tt += 1

        return pd.concat(results)

    def sim_flat_agent(AgentClass, name='None', kal=False, meta=False):
        np.random.seed(seed)

        tt = 0

        results = []
        for task, list_task in zip(['A', 'B'], [list_tasks_a, list_tasks_b]):

            for ii, (task_args, task_kwargs) in tqdm(enumerate(list_task), total=len(list_task)):
                if not kal:
                    agent_kwargs = dict(lr=list_lr[tt], inv_temp=list_beta[tt],
                                        goal_prior=goal_prior, mapping_prior=mapping_prior)
                else:
                    agent_kwargs = dict(var_i=list_var_e[tt], var_e=list_var_i[tt], ucb_weight=list_ucb_w[tt],
                            inv_temp=list_beta[tt], goal_prior=goal_prior, mapping_prior=mapping_prior)


                agent = AgentClass(Experiment(*task_args, **task_kwargs), **agent_kwargs)

                _res = None
                while _res is None:
                    _res = agent.generate(evaluate=evaluate, pruning_threshold=pruning_threshold)

                _res[u'Model'] = name
                _res[u'Iteration'] = [tt] * len(_res)
                _res[u'Task'] = [task] * len(_res)
                results.append(_res)
                tt += 1

        return pd.concat(results)

    results_ic = sim_agent(IndependentClusterAgent, name='Independent')
    results_ic.to_pickle('./data/exp_2_goals_batch_of_sims_joint{}.pkl'.format(tag))
    results_ic = None

    results_jc = sim_agent(JointClusteringAgent, name='Joint')
    results_jc.to_pickle('./data/exp_2_goals_batch_of_sims_indep{}.pkl'.format(tag))
    results_jc = None

    results_fl = sim_agent(FlatAgent, name='Flat', flat=True)
    results_fl.to_pickle('./data/exp_2_goals_batch_of_sims_flat{}.pkl'.format(tag))
    results_fl = None

    results_q = sim_flat_agent(NoCTX_QLearningAgent, name='NoCTX_Q-Learner')
    results_q.to_pickle('./data/exp_2_goals_batch_of_sims_nctxq{}.pkl'.format(tag))
    results_q = None

    results_meta = sim_agent(MetaAgent, name='Meta', meta=True)
    results_meta.to_pickle('./data/exp_2_goals_batch_of_sims_meta{}.pkl'.format(tag))
    results_meta = None

    results_q = sim_flat_agent(QLearningAgent, name='Q-Learner')
    results_q.to_pickle('./data/exp_2_goals_batch_of_sims_q{}.pkl'.format(tag))
    results_q = None

    results_fl = sim_flat_agent(KalmanUCBAgent, name='KalmanUCB', kal=True)
    results_fl.to_pickle('./data/exp_2_goals_batch_of_sims_kal{}.pkl'.format(tag))
    results_fl = None


def batch_exp_3_goals(seed=0, n_sims=1000, alpha_mu=0.0, alpha_scale=1.0, goal_prior=0.001, mapping_prior=0.001,
                      updated_new_c_only=False, pruning_threshold=10.0, beta_mu=2.0, beta_scale=1.0, tag=''):

    # alpha is sample from the distribution
    # log(alpha) ~ N(alpha_mu, alpha_scale)

    evaluate = False

    # pre generate a set of tasks for consistency.
    list_tasks = [gen_task_param_exp_3_goals() for _ in range(n_sims)]

    # pre draw the alphas for consistency
    list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale))
                  for _ in range(n_sims)]

    list_beta = [np.exp(scipy.random.normal(loc=beta_mu, scale=beta_scale))
                for _ in range(n_sims)]

    list_lr = logit(np.random.normal(-1, 1.0, n_sims))

    list_var_e = logit(np.random.normal(-1, 1.0, n_sims))
    list_var_i = logit(np.random.normal(-1, 1.0, n_sims))
    list_ucb_w = logit(np.random.normal(-1, 1.0, n_sims))

    def sim_agent(AgentClass, name='None', flat=False, meta=False):
        np.random.seed(seed)

        results = []
        for ii, (task_args, task_kwargs) in tqdm(enumerate(list_tasks), total=len(list_tasks)):

            if not flat:
                agent_kwargs = dict(alpha=list_alpha[ii], inv_temp=list_beta[ii],
                                    goal_prior=goal_prior, mapping_prior=mapping_prior)
            else:
                agent_kwargs = dict(inv_temp=list_beta[ii], goal_prior=goal_prior)

            if meta:
                p = np.random.uniform(0, 1.00)
                agent_kwargs['mix_biases'] = [np.log(p), np.log(1 - p)]
                agent_kwargs['update_new_c_only'] = updated_new_c_only

            agent = AgentClass(Experiment(*task_args, **task_kwargs), **agent_kwargs)

            _res = None
            while _res is None:
                _res = agent.generate(evaluate=evaluate, pruning_threshold=pruning_threshold)
            _res[u'Model'] = name
            _res[u'Iteration'] = [ii] * len(_res)
            results.append(_res)
        return pd.concat(results)

    def sim_flat_agent(AgentClass, name='None', kal=False, meta=False):
        np.random.seed(seed)

        results = []
        for ii, (task_args, task_kwargs) in tqdm(enumerate(list_tasks), total=len(list_tasks)):

            if not kal:
                agent_kwargs = dict(lr=list_lr[ii], inv_temp=list_beta[ii],
                                    goal_prior=goal_prior, mapping_prior=mapping_prior)
            else:
                agent_kwargs = dict(var_i=list_var_e[ii], var_e=list_var_i[ii], ucb_weight=list_ucb_w[ii],
                        inv_temp=list_beta[ii], goal_prior=goal_prior, mapping_prior=mapping_prior)

            agent = AgentClass(Experiment(*task_args, **task_kwargs), **agent_kwargs)

            _res = None
            while _res is None:
                _res = agent.generate(evaluate=evaluate, pruning_threshold=pruning_threshold)
            _res[u'Model'] = name
            _res[u'Iteration'] = [ii] * len(_res)
            results.append(_res)
        return pd.concat(results)

    results_ic = sim_agent(IndependentClusterAgent, name='Independent')
    results_ic.to_pickle('./data/exp_3_goals_batch_of_sims_joint{}.pkl'.format(tag))
    results_ic = None

    results_jc = sim_agent(JointClusteringAgent, name='Joint')
    results_jc.to_pickle('./data/exp_3_goals_batch_of_sims_indep{}.pkl'.format(tag))
    results_jc = None

    results_fl = sim_agent(FlatAgent, name='Flat', flat=True)
    results_fl.to_pickle('./data/exp_3_goals_batch_of_sims_flat{}.pkl'.format(tag))
    results_fl = None

    results_q = sim_flat_agent(NoCTX_QLearningAgent, name='NoCTX_Q-Learner')
    results_q.to_pickle('./data/exp_3_goals_batch_of_sims_nctxq{}.pkl'.format(tag))
    results_q = None

    results_meta = sim_agent(MetaAgent, name='Meta', meta=True)
    results_meta.to_pickle('./data/exp_3_goals_batch_of_sims_meta{}.pkl'.format(tag))
    results_meta = None

    results_q = sim_flat_agent(QLearningAgent, name='Q-Learner')
    results_q.to_pickle('./data/exp_3_goals_batch_of_sims_q{}.pkl'.format(tag))
    results_q = None

    results_fl = sim_flat_agent(KalmanUCBAgent, name='KalmanUCB', kal=True)
    results_fl.to_pickle('./data/exp_3_goals_batch_of_sims_kal{}.pkl'.format(tag))
    results_fl = None


def batch_exp_4_goals(seed=0, n_sims=1000, alpha_mu=0.0, alpha_scale=1.0, goal_prior=0.001, mapping_prior=0.001,
                      updated_new_c_only=False, pruning_threshold=10.0,  beta_mu=2.0, beta_scale=1.0, tag=''):

    # alpha is sample from the distribution
    # log(alpha) ~ N(alpha_mu, alpha_scale)

    evaluate = False

    # pre generate a set of tasks for consistency.
    list_tasks = [gen_task_param_exp_4_goals() for _ in range(n_sims)]

    # pre draw the alphas for consistency
    list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale))
                  for _ in range(n_sims)]

    list_beta = [np.exp(scipy.random.normal(loc=beta_mu, scale=beta_scale))
                  for _ in range(n_sims)]

    list_lr = logit(np.random.normal(-1, 1.0, n_sims))

    list_var_e = logit(np.random.normal(-1, 1.0, n_sims))
    list_var_i = logit(np.random.normal(-1, 1.0, n_sims))
    list_ucb_w = logit(np.random.normal(-1, 1.0, n_sims))

    def sim_agent(AgentClass, name='None', flat=False, meta=False):
        np.random.seed(seed)

        results = []
        for ii, (task_args, task_kwargs) in tqdm(enumerate(list_tasks), total=len(list_tasks)):

            if not flat:
                agent_kwargs = dict(alpha=list_alpha[ii], inv_temp=list_beta[ii],
                                    goal_prior=goal_prior, mapping_prior=mapping_prior)
            else:
                agent_kwargs = dict(inv_temp=list_beta[ii], goal_prior=goal_prior)

            if meta:
                p = np.random.uniform(0.0, 1.00)
                agent_kwargs['mix_biases'] = [np.log(p), np.log(1 - p)]
                agent_kwargs['update_new_c_only'] = updated_new_c_only

            agent = AgentClass(Experiment(*task_args, **task_kwargs), **agent_kwargs)

            _res = None
            while _res is None:
                _res = agent.generate(evaluate=evaluate, pruning_threshold=pruning_threshold)
            _res[u'Model'] = [name] * len(_res)
            _res[u'Iteration'] = [ii] * len(_res)
            results.append(_res)
        return pd.concat(results)

    def sim_flat_agent(AgentClass, name='None', kal=False, meta=False):
        np.random.seed(seed)

        results = []
        for ii, (task_args, task_kwargs) in tqdm(enumerate(list_tasks), total=len(list_tasks)):

            if not kal:
                agent_kwargs = dict(lr=list_lr[ii], inv_temp=list_beta[ii],
                                    goal_prior=goal_prior, mapping_prior=mapping_prior)
            else:
                agent_kwargs = dict(var_i=list_var_e[ii], var_e=list_var_i[ii], ucb_weight=list_ucb_w[ii],
                        inv_temp=list_beta[ii], goal_prior=goal_prior, mapping_prior=mapping_prior)


            agent = AgentClass(Experiment(*task_args, **task_kwargs), **agent_kwargs)

            _res = None
            while _res is None:
                _res = agent.generate(evaluate=evaluate, pruning_threshold=pruning_threshold)
            _res[u'Model'] = [name] * len(_res)
            _res[u'Iteration'] = [ii] * len(_res)
            results.append(_res)
        return pd.concat(results)

    results_ic = sim_agent(IndependentClusterAgent, name='Independent')
    results_ic.to_pickle('./data/exp_4_goals_batch_of_sims_joint{}.pkl'.format(tag))
    results_ic = None

    results_jc = sim_agent(JointClusteringAgent, name='Joint')
    results_jc.to_pickle('./data/exp_4_goals_batch_of_sims_indep{}.pkl'.format(tag))
    results_jc = None

    results_fl = sim_agent(FlatAgent, name='Flat', flat=True)
    results_fl.to_pickle('./data/exp_4_goals_batch_of_sims_flat{}.pkl'.format(tag))
    results_fl = None

    results_q = sim_flat_agent(NoCTX_QLearningAgent, name='NoCTX_Q-Learner')
    results_q.to_pickle('./data/exp_4_goals_batch_of_sims_nctxq{}.pkl'.format(tag))
    results_q = None
    
    results_meta = sim_agent(MetaAgent, name='Meta', meta=True)
    results_meta.to_pickle('./data/exp_4_goals_batch_of_sims_meta{}.pkl'.format(tag))
    results_meta = None

    results_q = sim_flat_agent(QLearningAgent, name='Q-Learner')
    results_q.to_pickle('./data/exp_4_goals_batch_of_sims_q{}.pkl'.format(tag))
    results_q = None

    results_fl = sim_flat_agent(KalmanUCBAgent, name='KalmanUCB', kal=True)
    results_fl.to_pickle('./data/exp_4_goals_batch_of_sims_kal{}.pkl'.format(tag))
    results_fl = None


def merge_dfs(tag, file_path='./data/'):
    model_list = 'joint indep meta flat q nctxq kal'.split()

    # # load experiment 2 goals
    print "Loading Experiment 1"
    sims_exp_2 = []
    for m in model_list:
        df = pd.read_pickle(file_path + 'exp_2_goals_batch_of_sims_{}{}.pkl'.format(m, tag))
        df.drop(['Start Location', 'End Location', 'Goal Locations', 'Action Map', 'Walls'], axis=1, inplace=True)
        df = df[df['In Goal']]
        sims_exp_2.append(df)
    df = None
    sims_exp_2 = pd.concat(sims_exp_2, sort=False)
    sims_exp_2.to_pickle(file_path + 'exp_2_goals_batch_of_sims{}.pkl'.format(tag))
    sims_exp_2 = None

    # load experiment 4 goals
    print "Loading Experiment 2"
    sims_exp_4 = []
    for m in model_list:
        df = pd.read_pickle(file_path + 'exp_4_goals_batch_of_sims_{}{}.pkl'.format(m, tag))
        df.drop(['Start Location', 'End Location', 'Goal Locations', 'Action Map', 'Walls'], axis=1, inplace=True)
        df = df[df['In Goal']]
        sims_exp_4.append(df)
    df = None
    sims_exp_4 = pd.concat(sims_exp_4, sort=False)
    sims_exp_4.to_pickle(file_path + 'exp_4_goals_batch_of_sims{}.pkl'.format(tag))
    sims_exp_4 = None


    # load experiment 3 goals
    print "Loading Experiment 3"
    sims_exp_3 = []
    for m in model_list:
        df = pd.read_pickle(file_path + 'exp_3_goals_batch_of_sims_{}{}.pkl'.format(m, tag))
        df.drop(['Start Location', 'End Location', 'Goal Locations', 'Action Map', 'Walls'], axis=1, inplace=True)
        df = df[df['In Goal']]
        sims_exp_3.append(df)
    df = None
    sims_exp_3 = pd.concat(sims_exp_3, sort=False)
    sims_exp_3.to_pickle(file_path + 'exp_3_goals_batch_of_sims{}.pkl'.format(tag))
    sims_exp_3 = None
    print "Done!"

if __name__ == "__main__":
    kwargs = dict(
        n_sims              = 2500,
        goal_prior          = 1e-10,
        mapping_prior       = 1e-10,
        alpha_mu            = -0.5,
        alpha_scale         = 1.0,
        pruning_threshold   = 500.,
        beta_mu             = 2.0,
        beta_scale          = 0.5,
        tag                 = '_update_all_trials__gp=1e-10_prune=500_mu=-0.5_scale=1.0_invtemp_mu=2.0_invtemp_scale=0.5'
    )

    batch_exp_4_goals(**kwargs)
    batch_exp_2_goals(**kwargs)
    batch_exp_3_goals(**kwargs)

    ## combine all of the dataframes into one per experiment for simplicity
    tag = '_update_all_trials__gp=1e-10_prune=500_mu=-0.5_scale=1.0_invtemp_mu=2.0_invtemp_scale=0.5'
    merge_dfs(tag)