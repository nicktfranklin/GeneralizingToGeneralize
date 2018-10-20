import scipy
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm

from models.grid_world import Experiment
from models.agents import IndependentClusterAgent, JointClusteringAgent, FlatAgent, MetaAgent
from models.experiment_designs.experiment2 import gen_task_param as gen_task_param_exp_4_goals
from models.experiment_designs.experiment1 import gen_task_param as gen_task_param_exp_3_goals
from models.experiment_designs.experiment_3a import gen_task_param as gen_task_param_exp_2_goals_a
from models.experiment_designs.experiment_3b import gen_task_param as gen_task_param_exp_2_goals_b


def batch_exp_2_goals(seed=0):
    n_sims = 1000  # generate 1000 so we can draw multiple samples from the same size

    # alpha is sample from the distribution
    # log(alpha) ~ N(alpha_mu, alpha_scale)
    alpha_mu = 0.0
    alpha_scale = 1.0

    inv_temp = 10.
    goal_prior = 10e-20
    prunning_threshold = 10.0
    evaluate = False

    np.random.seed(seed)

    # pre generate a set of tasks for consistency.
    list_tasks_a = [gen_task_param_exp_2_goals_a() for _ in range(n_sims)]
    list_tasks_b = [gen_task_param_exp_2_goals_b() for _ in range(n_sims)]

    # pre draw the alphas for consistency
    list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale))
                  for _ in range(n_sims + n_sims)]

    def sim_agent(AgentClass, name='None', flat=False, meta=False):
        tt = 0

        results = []
        for task, list_task in zip(['A', 'B'], [list_tasks_a, list_tasks_b]):

            for ii, (task_args, task_kwargs) in tqdm(enumerate(list_task), total=len(list_task)):
                if not flat:
                    agent_kwargs = dict(alpha=list_alpha[tt], inv_temp=inv_temp,
                                        goal_prior=goal_prior)
                else:
                    agent_kwargs = dict(inv_temp=inv_temp, goal_prior=goal_prior)

                if meta:
                    p = np.random.uniform(0, 1)
                    agent_kwargs['mix_biases'] = [np.log(p), np.log(1 - p)]
                    agent_kwargs['update_new_c_only'] = True

                agent = AgentClass(Experiment(*task_args, **task_kwargs), **agent_kwargs)

                _res = agent.generate(evaluate=evaluate, prunning_threshold=prunning_threshold)
                _res[u'Model'] = name
                _res[u'Iteration'] = [tt] * len(_res)
                _res[u'Task'] = [task] * len(_res)
                results.append(_res)
                tt += 1

        return pd.concat(results)

    results_ic = sim_agent(IndependentClusterAgent, name='Independent')
    results_jc = sim_agent(JointClusteringAgent, name='Joint')
    results_fl = sim_agent(FlatAgent, name='Flat', flat=True)
    results_meta = sim_agent(MetaAgent, name='Meta', meta=True)
    results = pd.concat([results_ic, results_jc, results_fl, results_meta])

    results.to_pickle('exp_2_goals_batch_of_sims.pkl')

def batch_exp_3_goals(seed=0):
    n_sims = 1000  # generate 1000 so we can draw multiple samples from the same size

    # alpha is sample from the distribution
    # log(alpha) ~ N(alpha_mu, alpha_scale)
    alpha_mu = 0.0
    alpha_scale = 1.0

    inv_temp = 10.
    goal_prior = 10e-20
    prunning_threshold = 10.0
    evaluate = False

    np.random.seed(seed)

    # pre generate a set of tasks for consistency.
    list_tasks = [gen_task_param_exp_3_goals() for _ in range(n_sims)]

    # pre draw the alphas for consistency
    list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale))
                  for _ in range(n_sims)]

    def sim_agent(AgentClass, name='None', flat=False, meta=False):
        results = []
        for ii, (task_args, task_kwargs) in tqdm(enumerate(list_tasks), total=len(list_tasks)):

            if not flat:
                agent_kwargs = dict(alpha=list_alpha[ii], inv_temp=inv_temp,
                                    goal_prior=goal_prior)
            else:
                agent_kwargs = dict(inv_temp=inv_temp, goal_prior=goal_prior)

            if meta:
                p = np.random.uniform(0, 1)
                agent_kwargs['mix_biases'] = [np.log(p), np.log(1 - p)]
                agent_kwargs['update_new_c_only'] = True

            agent = AgentClass(Experiment(*task_args, **task_kwargs), **agent_kwargs)

            _res = agent.generate(evaluate=evaluate, prunning_threshold=prunning_threshold)
            _res[u'Model'] = name
            _res[u'Iteration'] = [ii] * len(_res)
            results.append(_res)
        return pd.concat(results)

    results_ic = sim_agent(IndependentClusterAgent, name='Independent')
    results_jc = sim_agent(JointClusteringAgent, name='Joint')
    results_fl = sim_agent(FlatAgent, name='Flat', flat=True)
    results_meta = sim_agent(MetaAgent, name='Meta', meta=True)
    results = pd.concat([results_ic, results_jc, results_fl, results_meta])

    results.to_pickle('exp_3_goals_batch_of_sims.pkl')


def batch_exp_4_goals(seed=0):
    n_sims = 1000  # generate 1000 so we can draw multiple samples from the same size

    # alpha is sample from the distribution
    # log(alpha) ~ N(alpha_mu, alpha_scale)
    alpha_mu = 0.0
    alpha_scale = 1.0

    inv_temp = 10.
    goal_prior = 10e-20
    prunning_threshold = 10.0
    evaluate = False

    np.random.seed(seed)

    # pre generate a set of tasks for consistency.
    list_tasks = [gen_task_param_exp_4_goals() for _ in range(n_sims)]

    # pre draw the alphas for consistency
    list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale))
                  for _ in range(n_sims)]

    def sim_agent(AgentClass, name='None', flat=False, meta=False):
        results = []
        for ii, (task_args, task_kwargs) in tqdm(enumerate(list_tasks), total=len(list_tasks)):

            if not flat:
                agent_kwargs = dict(alpha=list_alpha[ii], inv_temp=inv_temp,
                                    goal_prior=goal_prior)
            else:
                agent_kwargs = dict(inv_temp=inv_temp, goal_prior=goal_prior)

            if meta:
                p = np.random.uniform(0, 1)
                agent_kwargs['mix_biases'] = [np.log(p), np.log(1 - p)]
                agent_kwargs['update_new_c_only'] = True

            agent = AgentClass(Experiment(*task_args, **task_kwargs), **agent_kwargs)

            _res = agent.generate(evaluate=evaluate, prunning_threshold=prunning_threshold)
            _res[u'Model'] = name
            _res[u'Iteration'] = [ii] * len(_res)
            results.append(_res)
        return pd.concat(results)

    results_ic = sim_agent(IndependentClusterAgent, name='Independent')
    results_jc = sim_agent(JointClusteringAgent, name='Joint')
    results_fl = sim_agent(FlatAgent, name='Flat', flat=True)
    results_meta = sim_agent(MetaAgent, name='Meta', meta=True)
    results = pd.concat([results_ic, results_jc, results_fl, results_meta])

    results.to_pickle('exp_4_goals_batch_of_sims.pkl')


if __name__ == "__main__":
    batch_exp_2_goals()
    batch_exp_3_goals(32153659)
    batch_exp_4_goals()
