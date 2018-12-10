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


def batch_exp_2_goals(seed=0, n_sims=1000, alpha_mu=0.0, alpha_scale=1.0, goal_prior=0.001, mapping_prior=0.001,
                      updated_new_c_only=False, pruning_threshold=10.0, inv_temp=10., tag=''):

    # alpha is sample from the distribution
    # log(alpha) ~ N(alpha_mu, alpha_scale)

    evaluate = False

    # pre generate a set of tasks for consistency.
    list_tasks_a = [gen_task_param_exp_2_goals_a() for _ in range(n_sims)]
    list_tasks_b = [gen_task_param_exp_2_goals_b() for _ in range(n_sims)]

    # pre draw the alphas for consistency
    list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale))
                  for _ in range(n_sims + n_sims)]

    def sim_agent(AgentClass, name='None', flat=False, meta=False):
        np.random.seed(seed)

        tt = 0

        results = []
        for task, list_task in zip(['A', 'B'], [list_tasks_a, list_tasks_b]):

            for ii, (task_args, task_kwargs) in tqdm(enumerate(list_task), total=len(list_task)):
                if not flat:
                    agent_kwargs = dict(alpha=list_alpha[tt], inv_temp=inv_temp,
                                        goal_prior=goal_prior, mapping_prior=mapping_prior)
                else:
                    agent_kwargs = dict(inv_temp=inv_temp, goal_prior=goal_prior)

                if meta:
                    p = np.random.uniform(0, 1)
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

    results_ic = sim_agent(IndependentClusterAgent, name='Independent')
    results_ic.to_pickle('exp_2_goals_batch_of_sims_joint{}.pkl'.format(tag))
    results_ic = None

    results_jc = sim_agent(JointClusteringAgent, name='Joint')
    results_jc.to_pickle('exp_2_goals_batch_of_sims_indep{}.pkl'.format(tag))
    results_jc = None

    results_fl = sim_agent(FlatAgent, name='Flat', flat=True)
    results_fl.to_pickle('exp_2_goals_batch_of_sims_flat{}.pkl'.format(tag))
    results_fl = None

    results_meta = sim_agent(MetaAgent, name='Meta', meta=True)
    results_meta.to_pickle('exp_2_goals_batch_of_sims_meta{}.pkl'.format(tag))
    results_meta = None


def batch_exp_3_goals(seed=0, n_sims=1000, alpha_mu=0.0, alpha_scale=1.0, goal_prior=0.001, mapping_prior=0.001,
                      updated_new_c_only=False, pruning_threshold=10.0, inv_temp=10., tag=''):

    # alpha is sample from the distribution
    # log(alpha) ~ N(alpha_mu, alpha_scale)

    evaluate = False

    # pre generate a set of tasks for consistency.
    list_tasks = [gen_task_param_exp_3_goals() for _ in range(n_sims)]

    # pre draw the alphas for consistency
    list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale))
                  for _ in range(n_sims)]

    def sim_agent(AgentClass, name='None', flat=False, meta=False):
        np.random.seed(seed)

        results = []
        for ii, (task_args, task_kwargs) in tqdm(enumerate(list_tasks), total=len(list_tasks)):

            if not flat:
                agent_kwargs = dict(alpha=list_alpha[ii], inv_temp=inv_temp,
                                    goal_prior=goal_prior, mapping_prior=mapping_prior)
            else:
                agent_kwargs = dict(inv_temp=inv_temp, goal_prior=goal_prior)

            if meta:
                p = np.random.uniform(0, 1)
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

    results_ic = sim_agent(IndependentClusterAgent, name='Independent')
    results_ic.to_pickle('exp_3_goals_batch_of_sims_joint{}.pkl'.format(tag))
    results_ic = None

    results_jc = sim_agent(JointClusteringAgent, name='Joint')
    results_jc.to_pickle('exp_3_goals_batch_of_sims_indep{}.pkl'.format(tag))
    results_jc = None

    results_fl = sim_agent(FlatAgent, name='Flat', flat=True)
    results_fl.to_pickle('exp_3_goals_batch_of_sims_flat{}.pkl'.format(tag))
    results_fl = None

    results_meta = sim_agent(MetaAgent, name='Meta', meta=True)
    results_meta.to_pickle('exp_3_goals_batch_of_sims_meta{}.pkl'.format(tag))
    results_meta = None


def batch_exp_4_goals(seed=0, n_sims=1000, alpha_mu=0.0, alpha_scale=1.0, goal_prior=0.001, mapping_prior=0.001,
                      updated_new_c_only=False, pruning_threshold=10.0, inv_temp=10., tag=''):

    # alpha is sample from the distribution
    # log(alpha) ~ N(alpha_mu, alpha_scale)

    evaluate = False

    # pre generate a set of tasks for consistency.
    list_tasks = [gen_task_param_exp_4_goals() for _ in range(n_sims)]

    # pre draw the alphas for consistency
    list_alpha = [np.exp(scipy.random.normal(loc=alpha_mu, scale=alpha_scale))
                  for _ in range(n_sims)]

    def sim_agent(AgentClass, name='None', flat=False, meta=False):
        np.random.seed(seed)

        results = []
        for ii, (task_args, task_kwargs) in tqdm(enumerate(list_tasks), total=len(list_tasks)):

            if not flat:
                agent_kwargs = dict(alpha=list_alpha[ii], inv_temp=inv_temp,
                                    goal_prior=goal_prior, mapping_prior=mapping_prior)
            else:
                agent_kwargs = dict(inv_temp=inv_temp, goal_prior=goal_prior)

            if meta:
                p = np.random.uniform(0, 1)
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

    results_ic = sim_agent(IndependentClusterAgent, name='Independent')
    results_ic.to_pickle('exp_4_goals_batch_of_sims_joint{}.pkl'.format(tag))
    results_ic = None

    results_jc = sim_agent(JointClusteringAgent, name='Joint')
    results_jc.to_pickle('exp_4_goals_batch_of_sims_indep{}.pkl'.format(tag))
    results_jc = None

    results_fl = sim_agent(FlatAgent, name='Flat', flat=True)
    results_fl.to_pickle('exp_4_goals_batch_of_sims_flat{}.pkl'.format(tag))
    results_fl = None

    results_meta = sim_agent(MetaAgent, name='Meta', meta=True)
    results_meta.to_pickle('exp_4_goals_batch_of_sims_meta{}.pkl'.format(tag))
    results_meta = None


if __name__ == "__main__":
    kwargs = dict(
        n_sims              = 1000,
        goal_prior          = 1e-10,
        mapping_prior       = 1e-10,
        alpha_mu            = -0.5,
        alpha_scale         = 1.0,
        pruning_threshold  = 100.,
        tag                 = '_update_all_trials_gp=1e-20_prune=100_mu=-0.5_scale=1.0'
    )

    batch_exp_4_goals(**kwargs)
    batch_exp_2_goals(**kwargs)
    batch_exp_3_goals(**kwargs)
