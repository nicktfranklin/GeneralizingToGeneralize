import numpy as np
import copy
from models.misc import randomize_context_order_with_autocorrelation

# define the task parameters
grid_world_size = (6, 6)
action_map_0 = {4: u'up', 5: u'left', 6: u'right', 7: u'down'}
action_map_1 = {0: u'up', 1: u'left', 2: u'right', 3: u'down'}
goal_0 = {'A': 1.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
goal_1 = {'A': 0.0, 'B': 1.0, 'C': 0.0, 'D': 0.0}
# goal_2 = {'A': 0.0, 'B': 0.0, 'C': 1.0, 'D': 0.0}
# goal_3 = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 1.0}


def rand_goal_locations():
    q_size = grid_world_size[0] / 2
    q1 = (np.random.randint(q_size) + q_size, np.random.randint(q_size) + q_size)
    q2 = (np.random.randint(q_size), np.random.randint(q_size) + q_size)
    q3 = (np.random.randint(q_size), np.random.randint(q_size))
    q4 = (np.random.randint(q_size) + q_size, np.random.randint(q_size))

    goal_locations = [q1, q2, q3, q4]
    np.random.shuffle(goal_locations)

    return goal_locations


def rand_init_loc(min_manhattan_dist=3):
    t = 0
    goal_locations = rand_goal_locations()
    while True:
        loc = (np.random.randint(grid_world_size[0]), np.random.randint(grid_world_size[1]))
        min_dist = grid_world_size[0]

        for g_loc in goal_locations:
            d = np.abs(g_loc[0] - loc[0]) + np.abs(g_loc[1] - loc[1])
            min_dist = np.min([min_dist, d])

        if min_dist >= min_manhattan_dist:
            return goal_locations, loc
        t += 1

        if t > 100:
            goal_locations = rand_goal_locations()
            t = 0


def add_sections(wall_sections):
    # this function is used when making sets of walls for a trial
    walls = []
    for section in wall_sections:
        section_ = copy.copy(section)
        section_.pop(np.random.randint(len(section)))
        for wall in section_:
            walls += wall
    return walls


# def make_wall_set_a():
#     wall_sections = [
#         [
#             [[2, 5, u'up'], [2, 6, u'down']],
#             [[3, 5, u'up'], [3, 6, u'down']],
#             [[4, 5, u'up'], [4, 6, u'down']],
#             [[5, 5, u'up'], [5, 6, u'down']],
#         ],
#         [
#             [[1, 5, u'right'], [2, 5, u'left']],
#             [[1, 4, u'right'], [2, 4, u'left']],
#             [[1, 3, u'right'], [2, 3, u'left']],
#             [[1, 2, u'right'], [2, 2, u'left']],
#         ],
#         [
#             [[2, 1, u'up'], [2, 2, u'down']],
#             [[3, 1, u'up'], [3, 2, u'down']],
#             [[4, 1, u'up'], [4, 2, u'down']],
#             [[5, 1, u'up'], [5, 2, u'down']],
#         ],
#         [
#             [[5, 5, u'right'], [6, 5, u'left']],
#             [[5, 4, u'right'], [6, 4, u'left']],
#             [[5, 3, u'right'], [6, 3, u'left']],
#             [[5, 2, u'right'], [6, 2, u'left']],
#         ]
#     ]
#     ind = np.arange(len(wall_sections))
#     np.random.permutation(ind)
#     wall_sections = [wall_sections[ii] for ii in ind]
#     walls = add_sections(wall_sections[1:])
#     for wall in wall_sections[0]:
#         walls += wall
#     return walls
#
#
# def make_wall_set_b():
#     wall_sections = [
#         [
#             [[0, 3, u'up'], [0, 4, u'down']],
#             [[1, 3, u'up'], [1, 4, u'down']],
#             [[2, 3, u'up'], [2, 4, u'down']],
#             [[3, 3, u'up'], [3, 4, u'down']],
#         ],
#         [
#             [[3, 0, u'right'], [4, 0, u'left']],
#             [[3, 1, u'right'], [4, 1, u'left']],
#             [[3, 2, u'right'], [4, 2, u'left']],
#             [[3, 3, u'right'], [4, 3, u'left']],
#         ],
#         [
#             [[4, 3, u'up'], [4, 4, u'down']],
#             [[5, 3, u'up'], [5, 4, u'down']],
#             [[6, 3, u'up'], [6, 4, u'down']],
#             [[7, 3, u'up'], [7, 4, u'down']],
#         ],
#         [
#             [[3, 4, u'right'], [4, 4, u'left']],
#             [[3, 5, u'right'], [4, 5, u'left']],
#             [[3, 6, u'right'], [4, 6, u'left']],
#             [[3, 7, u'right'], [4, 7, u'left']],
#         ]
#     ]
#     return add_sections(wall_sections)
#
#
# def make_wall_set_c():
#     wall_sections = [
#         [
#             [[1, 0, u'up'], [1, 1, u'down']],
#             [[2, 0, u'up'], [2, 1, u'down']],
#             [[1, 2, u'up'], [1, 3, u'down']],
#             [[2, 2, u'up'], [2, 3, u'down']], ],
#         [
#             [[0, 1, u'right'], [1, 1, u'left']],
#             [[0, 2, u'right'], [1, 2, u'left']],
#             [[2, 1, u'right'], [3, 1, u'left']],
#             [[2, 2, u'right'], [3, 2, u'left']],
#         ],
#         [
#             [[1, 4, u'up'], [1, 5, u'down']],
#             [[2, 4, u'up'], [2, 5, u'down']],
#             [[1, 6, u'up'], [1, 7, u'down']],
#             [[2, 6, u'up'], [2, 7, u'down']],
#         ],
#         [
#             [[0, 5, u'right'], [1, 5, u'left']],
#             [[0, 6, u'right'], [1, 6, u'left']],
#             [[2, 5, u'right'], [3, 5, u'left']],
#             [[2, 6, u'right'], [3, 6, u'left']],
#         ],
#         [
#             [[5, 0, u'up'], [5, 1, u'down']],
#             [[6, 0, u'up'], [6, 1, u'down']],
#             [[5, 2, u'up'], [5, 3, u'down']],
#             [[6, 2, u'up'], [6, 3, u'down']],
#         ],
#         [
#             [[4, 1, u'right'], [5, 1, u'left']],
#             [[4, 2, u'right'], [5, 2, u'left']],
#             [[6, 1, u'right'], [7, 1, u'left']],
#             [[6, 2, u'right'], [7, 2, u'left']],
#         ],
#         [
#             [[5, 4, u'up'], [5, 5, u'down']],
#             [[6, 4, u'up'], [6, 5, u'down']],
#             [[5, 6, u'up'], [5, 7, u'down']],
#             [[6, 6, u'up'], [6, 7, u'down']],
#         ],
#         [
#             [[4, 5, u'right'], [5, 5, u'left']],
#             [[4, 6, u'right'], [5, 6, u'left']],
#             [[6, 5, u'right'], [7, 5, u'left']],
#             [[6, 6, u'right'], [7, 6, u'left']],
#         ],
#     ]
#
#     center_wall = [
#         [
#             [[3, 3, u'right'], [4, 3, u'left']],
#             [[3, 4, u'right'], [4, 4, u'left']],
#         ],
#         [
#             [[3, 3, u'up'], [3, 4, u'down']],
#             [[4, 3, u'up'], [4, 4, u'down']],
#         ]
#     ]
#
#     walls_c = add_sections(wall_sections)
#     for wall in center_wall[np.random.randint(2)]:
#         walls_c += wall
#
#     return walls_c


def make_blank_wall_set():
    return []


def make_trial(context=0):
    # select a random wall generator to generate walls
    # wall_generator = [
    #     make_wall_set_a, make_wall_set_b, make_wall_set_c,
    # ][np.random.randint(3)]
    # walls = wall_generator()

    #
    goal_locations, start_location = rand_init_loc()

    definitions_dict = {
        0: (goal_0, action_map_0),
        1: (goal_0, action_map_0),
        2: (goal_1, action_map_1),
        # test contexts
        3: (goal_0, action_map_1),
        4: (goal_0, action_map_1),
        5: (goal_1, action_map_0),
    }
    goal_set, action_map = definitions_dict[context]

    goal_dict = {
        goal_locations[0]: ('A', goal_set['A']),
        goal_locations[1]: ('B', goal_set['B']),
        # goal_locations[2]: ('C', goal_set['C']),
        # goal_locations[3]: ('D', goal_set['D']),
    }

    return start_location, goal_dict, action_map


def gen_task_param():
    balance = [4, 4, 8]
    list_start_locations = []
    list_goals = []
    list_context = []
    list_action_map = []

    contexts_a0 = randomize_context_order_with_autocorrelation(balance, repeat_probability=0.20)
    contexts_a1 = randomize_context_order_with_autocorrelation(balance, repeat_probability=0.08)
    contexts_b = [c + 3 for c, reps in enumerate([4, 4 ,8]) for _ in range(reps)]
    np.random.shuffle(contexts_b)
    contexts = list(contexts_a0) + list(contexts_a1) + list(contexts_b)

    for ii, ctx in enumerate(contexts):
        start_location, goal_dict, action_map = make_trial(ctx)

        # define the trial and add to the lists
        list_start_locations.append(start_location)
        list_goals.append(goal_dict)
        list_context.append(ctx)
        list_action_map.append(action_map)

    args = [list_start_locations, list_goals, list_context, list_action_map]
    kwargs = dict(grid_world_size=grid_world_size)
    return args, kwargs
