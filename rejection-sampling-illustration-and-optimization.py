import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_func
from scipy.special import beta as beta_val

# Plot the theoretical Beta(a, b) probability density
x = np.linspace(0, 1, 300)
a = 2
b = 5


def f_a_b(a, b):
    return beta_func.pdf(x, a, b)


def f_a_b_x(a, b, x):
    return beta_func.pdf(x, a, b)


y = f_a_b(a, b)
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="blue")
plt.legend()
plt.show()

# Simulation of Beta(a, 1) distribution
n = 10000
U = rd.random(n)
V_a_1 = U**(1/a)
plt.hist(V_a_1, bins=30, density=True)

# Simulation of Beta(1, b) distribution
U = rd.random(n)
V_1_b = 1-(1-U)**(1/b)
plt.hist(V_1_b, bins=30, density=True)

# Existence of bounding constants for the Beta(a, b) density
B_a_b = beta_val(a, b)
k_a = 1/(a*B_a_b)
k_b = 1/(b*B_a_b)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="blue")
plt.plot(x, f_a_b(a, 1)*k_a, label=r'$f_{a,1}(x) * k_{a}$', color="red")
plt.plot(x, f_a_b(1, b)*k_b, label=r'$f_{1,b}(x) * k_{b}$', color="green")
plt.legend(fontsize=12)

# Setting up the rejection sampling method based on f(a, 1)
######################################################################
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="blue")
plt.plot(x, f_a_b(a, 1)*k_a, label=r'$f_{a,1}(x) * k_{a}$', color="red")
plt.legend(fontsize=12)

nb_val = 0
list_X = []
while nb_val < 3000:
    X = rd.random()  # Selecting an x-value within the domain under study
    G = X ** (1/a)  # Computing the value of the proposal distribution at X
    U = rd.random() * k_a * G  # Generating a random height between 0 and k*G
    if U <= f_a_b_x(a, b, X):  # Accept–reject decision
        list_X.append(X)
        nb_val += 1

plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="blue", linewidth=2)
plt.hist(list_X, bins=30, density=True,
         label='Approximation', color="red", alpha=0.8)
plt.title(r'Approximation via rejection sampling using $f_{a,1}(x)$')
plt.legend(fontsize=12)

# Setting up the rejection sampling method based on f(1, b)
######################################################################
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="blue")
plt.plot(x, f_a_b(a, 1)*k_a, label=r'$f_{1,b}(x) * k_{b}$', color="green")
plt.legend(fontsize=12)

nb_val = 0
list_X = []
while nb_val < 3000:
    X = rd.random()
    G = 1 - (1 - X) ** (1/b)
    U = rd.random() * k_b * G
    if U <= f_a_b_x(a, b, X):
        list_X.append(X)
        nb_val += 1

plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="blue", linewidth=2)
plt.hist(list_X, bins=30, density=True,
         label='Approximation', color="green", alpha=0.8)
plt.title(r'Approximation via rejection sampling using $f_{1,b}(x)$')
plt.legend(fontsize=12)

# Computation and visualization of errors
######################################################################
######################################################################
plt.figure(figsize=(8, 4))

counts, bin_edges = np.histogram(list_X, bins=30, density=True)[:2]
dz = bin_edges[1] - bin_edges[0]

error = 0
for i in range(len(bin_edges)-1):
    z = bin_edges[i] + dz / 2
    error += np.abs(counts[i] - f_a_b_x(a, b, z)) * dz
    plt.plot([z, z], [counts[i], f_a_b_x(a, b, z)], color="red", linestyle=":")


plt.plot(x, y, label=r'$f_{a,b}(x)$', color="blue", linewidth=2)
plt.hist(list_X, bins=30, density=True,
         label='Approximation', color="green", alpha=0.8)
plt.title(r'Approximation via rejection sampling using $f_{1,b}(x)$')
plt.legend(fontsize=12)
print("Error =", error)
