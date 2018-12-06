# cython: profile=False
# cython: linetrace=False
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

INT_DTYPE = np.int32
ctypedef np.int32_t INT_DTYPE_t

cdef extern from "math.h":
    double log(double x)

cdef extern from "math.h":
    double fmax(double a, double b)

cdef extern from "math.h":
    double abs(double x)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[DTYPE_t] value_iteration(
        np.ndarray[DTYPE_t, ndim=3] transition_function,
        np.ndarray[DTYPE_t, ndim=1] reward_function,
        float gamma,
        float stop_criterion):
    """
    """
    cdef double [:,:,::1] T = transition_function
    cdef double [:] R = reward_function

    cdef int n_s, n_a, a, sp, s
    n_s = transition_function.shape[0]
    n_a = transition_function.shape[1]
    cdef double [:] v = np.zeros(n_s, dtype=DTYPE)
    cdef double r, q
    cdef double [:] v_temp

    stop_criterion **= 2
    while True:
        delta = 0
        v_temp = np.zeros(n_s, dtype=DTYPE)
        for s in range(n_s):
            q_max = -10000  # initialize q_max to some very negative number
            for a in range(n_a):
                q = 0  # need a single q-value for each (s,a) pair - don't need to save the full function, only the max

                # update the q-value using the bellman equation
                for sp in range(n_s):
                    q += T[s, a, sp] * R[sp] + gamma * T[s, a, sp] * v[sp]

                q_max = fmax(q_max, q)  # compare the q-value to the current maximum value

            v_temp[s] = q_max  # store the state value

            delta = fmax(delta, (v[s] - v_temp[s]) ** 2)  # cache the largest incremental update

        v = v_temp
        if delta < stop_criterion:
            return np.array(v)


cpdef np.ndarray[INT_DTYPE_t, ndim=1] policy_iteration(
            np.ndarray[DTYPE_t, ndim=3] transition_function,
            np.ndarray[DTYPE_t, ndim=1] reward_function,
            float gamma,
            float stop_criterion):

    cdef int n_s, n_a, s, sp, b, t, a
    n_s = transition_function.shape[0]
    n_a = transition_function.shape[1]

    cdef double [:] V = np.random.rand(n_s)
    cdef int [:] pi = np.array(np.random.randint(n_a, size=n_s), dtype=INT_DTYPE)
    cdef bint policy_stable = False
    cdef double delta, v, V_temp

    cdef double [:] rew_func = reward_function
    cdef double [:,:,::1] trans_func = transition_function
    cdef np.ndarray[DTYPE_t, ndim=1] v_a

    stop_criterion **= 2
    while not policy_stable:
        while True:
            delta = 0
            for s in range(n_s):
                v = V[s]

                # evaluate V[s] with belman eq!
                V_temp = 0
                for sp in range(n_s):
                    V_temp += trans_func[s, pi[s], sp] * (rew_func[sp] + gamma*V[sp])

                V[s] = V_temp
                delta = fmax(delta, (v - V[s])**2)

            if delta < stop_criterion:
                break

        policy_stable = True
        for s in range(n_s):
            b = pi[s]

            v_a = np.zeros(n_a, dtype=DTYPE)
            for a in range(n_a):
                for sp in range(n_s):
                    v_a[a] += trans_func[s, a, sp] * (rew_func[sp] + gamma*V[sp])

            pi[s] = np.argmax(v_a)
#
            if not b == pi[s]:
                policy_stable = False

    return np.array(pi)



cpdef np.ndarray[DTYPE_t] policy_evaluation(
        np.ndarray[INT_DTYPE_t, ndim=1] policy,
        np.ndarray[DTYPE_t, ndim=3] transition_function,
        np.ndarray[DTYPE_t, ndim=1] reward_function,
        float gamma,
        float stop_criterion):

    cdef int [:] pi = policy
    cdef double [:,:,::1] T = transition_function
    cdef double [:] R = reward_function

    cdef int n_s, sp, s
    n_s = transition_function.shape[0]
    cdef double [:] V = np.zeros(n_s, dtype=DTYPE)
    cdef double v, V_temp

    stop_criterion **= 2
    while True:
        delta = 0
        for s in range(n_s):
            v = V[s]

            V_temp = 0
            for sp in range(n_s):
                V_temp += T[s, pi[s], sp] * (R[sp] + gamma*V[sp])
            V[s] = V_temp

            delta = fmax(delta, (v - V[s])**2)

        if delta < stop_criterion:
            return np.array(V)

cdef int array_sum(int [:] array):
    cdef int sum = 0
    cdef int ii
    for ii in range(len(array)):
        sum += array[ii]
    return sum

cpdef double get_prior_log_probability(dict ctx_assignment, double alpha):
    """This takes in an assignment of contexts to groups and returns the
    prior probability over the assignment using a CRP
    :param alpha:
    :param ctx_assignment:
    """
    cdef int ii, k0, t
    cdef double log_prob = 0
    cdef list values = ctx_assignment.values()

    cdef int n_ctx = len(values)
    cdef int k
    if n_ctx > 0:
        k = max(values) + 1
    else:
        k = 1

    cdef int [:] n_k = np.zeros(k, dtype=INT_DTYPE)

    for k0 in values:
        if n_k[k0] == 0:
            log_prob += log(alpha / (array_sum(n_k) + alpha))
        else:
            log_prob += log(n_k[k0] / (array_sum(n_k) + alpha))
        n_k[k0] += 1

    return log_prob


cpdef np.ndarray[DTYPE_t, ndim=3] convert_transition_to_primitives(np.ndarray[DTYPE_t, ndim=3] transition_function, np.ndarray[DTYPE_t, ndim=2] pr_aa_given_a):
    cdef int n_states, n_aa, n_a, s, aa, sp, a
    n_states = transition_function.shape[0]
    n_aa  = transition_function.shape[1]
    n_a = pr_aa_given_a.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=3] transition_func_primitives = np.zeros((n_states,n_a, n_states))

    cdef double [:,:,:] t_abstract = np.zeros((n_states, n_aa + 1, n_states))
    for s in range(n_states):
        for aa in range(n_aa):
            for sp in range(n_states):
                t_abstract[s, aa, sp] = transition_function[s, aa, sp]

    for s in range(n_states):
        t_abstract[s, n_aa, s] = 1.0

    for s in range(n_states):
        for a in range(n_a):
            for aa in range(n_aa + 1):
                for sp in range(n_states):
                    transition_func_primitives[s, a, sp] += t_abstract[s, aa, sp] * pr_aa_given_a[aa, a]

    return transition_func_primitives


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[DTYPE_t] bellman_equation(
        np.ndarray[DTYPE_t, ndim=3] transition_function,
        np.ndarray[DTYPE_t, ndim=1] reward_function,
        int state,
        float gamma,
        float stop_criterion):
    """
    """
    cdef double [:,:,::1] T = transition_function
    cdef double [:] R = reward_function

    cdef int n_s, n_a, a, sp, s
    n_s = transition_function.shape[0]
    n_a = transition_function.shape[1]
    cdef double [:] v = np.zeros(n_s, dtype=DTYPE)
    cdef double r, q
    cdef double [:] v_temp

    stop_criterion **= 2
    while True:
        delta = 0
        v_temp = np.zeros(n_s, dtype=DTYPE)
        for s in range(n_s):
            q_max = -10000  # initialize q_max to some very negative number
            for a in range(n_a):
                q = 0  # need a single q-value for each (s,a) pair - don't need to save the full function, only the max

                # update the q-value using the bellman equation
                for sp in range(n_s):
                    q += T[s, a, sp] * R[sp] + gamma * T[s, a, sp] * v[sp]

                q_max = fmax(q_max, q)  # compare the q-value to the current maximum value

            v_temp[s] = q_max  # store the state value

            delta = fmax(delta, (v[s] - v_temp[s]) ** 2)  # cache the largest incremental update

        v = v_temp
        if delta < stop_criterion:
            break


    # use the bellman equation to get the q-values
    cdef double[:] q_values = np.zeros(n_a)
    for aa in range(n_a):
        for sp in range(n_s):
            q_values[aa] += T[s, aa, sp] * (R[sp] + gamma * v[sp])

    return np.array(q_values)
