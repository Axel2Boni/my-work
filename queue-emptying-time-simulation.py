# Description
# This project studies a discrete-time queueing model where one client is served
# at each time step and new clients arrive randomly.

# Model
# - 2 arrivals with probability p
# - 1 arrival with probability q
# - 0 arrivals with probability 1 - p - q

# Starting from an initial queue size x0, we estimate the expected time T
# until the queue becomes empty.
# In addition, given an observed or target value of T, the code explores how to
# infer or identify the arrival parameters p and q that are consistent with this
# expected emptying time.

import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

# Illustration of the process for given parameters
n = 500
x0 = 10
p = 0.2
q = 0.4


def simul_queue(n, x0):
    X = np.zeros(n+1)
    X[0] = x0
    for t in range(n):
        c = rd.random()
        if c < p:
            X[t+1] = X[t] + 2
        elif c < p + q:
            X[t+1] = X[t] + 1
        else:
            X[t+1] = X[t]
        if X[t+1] > 0:
            X[t+1] -= 1
    return X


nbr_of_simul = 5
for i in range(nbr_of_simul):
    X = simul_queue(n, x0)
    plt.plot(np.arange(n+1), X)
    plt.title(rf"Simulation of {nbr_of_simul} trajectories of $(X_k)$")

# Research of the expected value of T for given parameters


def simul_queue_T(x0):
    X = []
    X.append(x0)
    t = 0
    while X[t] > 0:
        X.append(0)
        c = rd.random()
        if c < p:
            X[t+1] = X[t] + 2
        elif c < p + q:
            X[t+1] = X[t] + 1
        else:
            X[t+1] = X[t]
        if X[t+1] > 0:
            X[t+1] -= 1
        t = t+1
    return t


nbr_of_iteration = 10**3
x0 = 10
T = np.zeros(nbr_of_iteration)
for i in range(nbr_of_iteration):
    T[i] = simul_queue_T(x0)
print("An estimation of E(T) is : ", np.mean(T))

###########################################################
# Parameter estimation given the value of T
# Example: for T = 50 and x0 = 10, the corresponding parameters are (p, q) = (0.2, 0.4),
# as obtained in the first part
T = 100
x0 = 10

# We rewrite the function `simul_queue_T` to make it more robust, i.e.,
# to detect whether T is almost surely finite and to directly return
# the expected value of T (or -1 if the process is not almost surely finite)


def expected_val_T(p, q, x0, T):
    nbr_of_simul = 1000
    L = []
    for n in range(nbr_of_simul):
        i = 0
        X = [x0]
        while X[i] != 0:
            X.append(0)
            c = rd.random()
            if c < p:
                X[i+1] = X[i] + 2
            elif c < p + q:
                X[i+1] = X[i] + 1
            else:
                X[i+1] = X[i]
            if X[i+1] > 0:
                X[i+1] -= 1

            i += 1
            # Is T almost surely finite ?
            if i > T*10**2:
                return -1
        L.append(i)
    return np.mean(L)


expected_val_T(0.2, 0.4, 10, 50)
# -> ≈50

# We compute a matrix of expected values for each (p, q) pair
# The first column (resp. row) contains the values of p (resp. q)

first_mat = np.zeros((10, 10))
for i in range(1, 10):
    first_mat[i, 0] = round(0.1 * i, 3)
for j in range(1, 10):
    first_mat[0, j] = round(0.1 * j, 3)

for i in range(1, 10):
    for j in range(1, 10):
        if first_mat[i, j-1] == -1:
            for k in range(j, 10):
                first_mat[i, k] = -1
        else:
            first_mat[i, j] = expected_val_T(
                first_mat[i, 0], first_mat[0, j], x0, T)

# We find the best candidate among all the parameter pairs:
eff_coord = (1, 1)
for i in range(1, 10):
    for j in range(1, 10):
        if abs(first_mat[i, j] - T) < abs(first_mat[eff_coord[0], eff_coord[1]] - T):
            eff_coord = (i, j)

eff_pair = (first_mat[eff_coord[0], 0], first_mat[0, eff_coord[1]])

# We perform a second iteration to further refine the parameter estimates:
second_mat = np.zeros((12, 12))
for i in range(1, 12):
    second_mat[i, 0] = round(eff_pair[0]-0.06 + 0.01 * i, 3)
for j in range(1, 12):
    second_mat[0, j] = round(eff_pair[1]-0.06 + 0.01 * j, 3)

for i in range(1, 12):
    for j in range(1, 12):
        if second_mat[i, j-1] == -1:
            for k in range(j, 10):
                second_mat[i, k] = -1
        else:
            second_mat[i, j] = expected_val_T(
                second_mat[i, 0], second_mat[0, j], x0, T)

# Again, we find the best candidate among all the parameter pairs:
eff_coord = (1, 1)
for i in range(1, 12):
    for j in range(1, 12):
        if abs(second_mat[i, j] - T) < abs(second_mat[eff_coord[0], eff_coord[1]] - T):
            eff_coord = (i, j)

eff_pair = (second_mat[eff_coord[0], 0], second_mat[0, eff_coord[1]])

# Empirical verification of the results:
expected_val_T(eff_pair[0], eff_pair[1], x0, T)
