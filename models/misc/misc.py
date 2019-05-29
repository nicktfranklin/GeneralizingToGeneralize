import numpy as np
import pandas as pd
import models.grid_world
from models.cython_library import policy_iteration


def sample_cmf(cmf):
    return int(np.sum(np.random.rand(1) > cmf))


def softmax_to_pdf(q_values, inverse_temperature):
    e = np.exp(q_values / inverse_temperature)
    return e / e.sum()


def enumerate_assignments(max_context_number):
    """
     enumerate all possible assignments of contexts to clusters for a fixed number of contexts. Has the
     hard assumption that the first context belongs to cluster #1, to remove redundant assignments that
     differ in labeling.

    :param max_context_number: int
    :return: list of lists, each a function that takes in a context id number and returns a cluster id number
    """
    cluster_assignments = [{}]  # context 0 is always in cluster 1

    for contextNumber in range(0, max_context_number):
        cluster_assignments = augment_assignments(cluster_assignments, contextNumber)

    return cluster_assignments


def augment_assignments(cluster_assignments, new_context):
    if (len(cluster_assignments) == 0) | (len(cluster_assignments[0]) == 0):
        _cluster_assignments = list()
        _cluster_assignments.append({new_context: 0})
    else:
        _cluster_assignments = list()
        for assignment in cluster_assignments:
            new_list = list()
            for k in range(0, max(assignment.values()) + 2):
                _assignment_copy = assignment.copy()
                _assignment_copy[new_context] = k
                new_list.append(_assignment_copy)
            _cluster_assignments += new_list

    return _cluster_assignments


def randomize_context_order_with_autocorrelation(balance, repeat_probability=0.25):

    contexts = [c for c, reps in enumerate(balance) for _ in range(reps)]
    order = list()

    while len(contexts) > 0:
        # randomize the order
        np.random.shuffle(contexts)

        # draw a random context:
        ctx = contexts[0]

        # figure out how many repeats
        n_rep = 1
        while np.random.rand() < repeat_probability:
            n_rep += 1
        # append the contexts to the order (up to n_rep) times and remove
        # them from the available contexts
        while (n_rep > 0) & (ctx in contexts):
            order.append(ctx)
            contexts.remove(ctx)
            n_rep -= 1
    return order


def prep_subj_for_regression(subj_df, grid_world_size=(8, 8)):

    # really, I just need the contexts, the Chosen Goal,
    # Abstract action observed and the primitive action chosen
    # nvm, I also need the walls, and the goal locations to construct a new gridworld...
    cols = ['Action', 'Action Map', 'Chosen Goal', 'Context',
            'Goal Locations', 'Key-press', 'Start Location', 'Walls', 'Trial Number',
            'Steps Taken']
    subj_df = subj_df.loc[:, cols]

    subj_task = make_subj_task_from_data(subj_df, grid_world_size)

    # assume that the final goal is always intentional and movement towards that goal is
    # always planned. So, put the reached goal as the "chosen goal" at each step
    prepped_subj_data = subj_df.copy()
    for t in set(prepped_subj_data['Trial Number']):
        vec = prepped_subj_data['Trial Number'] == t
        goal = list(prepped_subj_data.loc[vec, 'Chosen Goal'])[-1]
        prepped_subj_data.loc[vec, 'Chosen Goal'] = goal

    # keep only the needed columns
    cols = ['Action', 'Chosen Goal', 'Context', 'Key-press', 'Start Location', 'Trial Number']
    prepped_subj_data = prepped_subj_data.loc[:, cols]

    return subj_task, prepped_subj_data


def make_subj_task_from_data(subj_df, grid_world_size=(8, 8), goal_key=None):
    from models.grid_world import Experiment

    if goal_key is None:
        # the default are the values for experiment 1c, otherwise need to be specified.
        goal_key = {0: u'A', 1: u'A', 2: u'B', 3: u'A', 4: u'C', 5: u'C'}

    # really, I just need the contexts, the Chosen Goal,
    # Abstract action observed and the primitive action chosen
    # nvm, I also need the walls, and the goal locations to construct a new gridworld...
    cols = ['Action', 'Action Map', 'Chosen Goal', 'Context',
            'Goal Locations', 'Key-press', 'Start Location', 'Walls', 'Trial Number',
            'Steps Taken']
    subj_df = subj_df.loc[:, cols]

    # create a task. Requires a list of start locations, goals, contexts, action maps and walls
    list_goals = []
    set_primitive_actions = set()

    X = subj_df.loc[subj_df['Steps Taken'] == 1, :]
    list_start_locations = list(X['Start Location'])
    list_contexts = list(X['Context'])
    list_action_maps = list(X['Action Map'])
    list_walls = list(X['Walls'])

    list_goal_locations = list(X['Goal Locations'])
    list_context = list(X['Context'])

    for trial in set(subj_df.loc[:, 'Trial Number']):
        t = int(trial)
        # goal locations have to be transformed for the format of a gridworld
        goal_locations = list_goal_locations[t]
        ctx = list_contexts[t]
        goal_values = {goal: 0.0 for goal in 'A B C D'.split()}
        goal_values[goal_key[ctx]] = 1.0
        goal_dicts = {loc: (label, goal_values[label]) for loc, label in goal_locations.iteritems()}
        list_goals.append(goal_dicts)

        for k in list_action_maps[t]:
            set_primitive_actions.add(int(k))

    # use these lists to create a task
    args = [list_start_locations, list_goals, list_contexts, list_action_maps]
    kwargs = dict(grid_world_size=grid_world_size, list_walls=list_walls, primitive_actions=list(set_primitive_actions))
    return Experiment(*args, **kwargs)


def get_abstract_action_policy(task, goal, gamma=0.8, stop_criterion=0.001):
    grid = task.current_trial
    assert type(grid) is models.grid_world.GridWorld

    # convert goal to value to probability density
    goal_probability = np.zeros(task.n_goals)
    goal_idx = task.get_goal_index(goal)
    goal_probability[goal_idx] = 1.0

    # convert the goal values to the lookup table values for states
    reward_function = np.zeros(len(grid.state_location_key))
    for location, goal in grid.goal_locations.iteritems():
        goal_state = grid.state_location_key[location]
        goal_value = goal_probability[task.get_goal_index(goal)]
        reward_function[goal_state] = goal_value - 0.1 * (1 - goal_value)

    # use the reward function to make a policy of abstract actions
    transition_function = task.current_trial.transition_function
    pi = policy_iteration(transition_function, reward_function, gamma, stop_criterion)

    return pi


def get_shortest_path_length(subject_data, grid_world_size=(8, 8), goal_key=None):
    # make a task from the subjects raw data
    task = make_subj_task_from_data(subject_data, grid_world_size=grid_world_size, goal_key=goal_key)

    # get a dictionary of the selected goals
    goals_selected = {}
    for idx in subject_data.index:
        goals_selected[subject_data.loc[idx, 'Trial Number']] = subject_data.loc[idx, 'Chosen Goal']

    # initialize variables
    new_trial = True
    path_lengths = dict()
    path_length = 0
    pi = []
    inverse_action_map = dict()
    t = -1

    # loop through and get path lengths
    while True:
        if new_trial:
            t = task.current_trial_number
            pi = get_abstract_action_policy(task, goals_selected[t])
            path_length = 0

            inverse_action_map = {
                dir_: a for a, dir_ in task.current_trial.action_map.iteritems()
            }

        path_length += 1
        path_lengths[t] = path_length

        # use the policy to get the next action
        s = task.state_location_key[task.get_location()]
        aa = pi[s]
        a = inverse_action_map[task.current_trial.inverse_abstract_action_key[aa]]
        _, _, goal_id, _ = task.move(a)

        new_trial = goal_id is not None
        if task.end_check():
            return path_lengths

        if path_length > 25:
            return path_lengths


def prep_subj_model_fitting(subj_df, grid_world_size=(6, 6), goal_key=None, mapping_key=None):

    # really, I just need the contexts, the Chosen Goal,
    # Abstract action observed and the primitive action chosen
    # nvm, I also need the walls, and the goal locations to construct a new gridworld...
    cols = ['Action', 'Action Map', 'Chosen Goal', 'Context',
            'Goal Locations', 'Key-press', 'Start Location', 'Walls', 'Trial Number',
            'Steps Taken']

    subj_task = make_subj_task_from_data(subj_df.loc[:, cols], grid_world_size, goal_key=goal_key)

    prepped_subj_data = pd.DataFrame([
            subj_df['Action'],
            subj_df['Chosen Goal'],
            subj_df['Reward'] / 10,
            subj_df['Context'].astype(int),
            subj_df['Key-press'].astype(int),
            subj_df['Start Location'],
            subj_df['Trial Number'].astype(int)
        ]).T
    prepped_subj_data.index = range(len(prepped_subj_data))

    # make a vector of all of the mapping identities
    if mapping_key is not None:
        prepped_subj_data['Mapping'] = [mapping_key[c] for c in subj_df['Context'].astype(int)]

    return subj_task, prepped_subj_data


def inverse_logit(x):
    return 1. / (1. + np.exp(-x))



