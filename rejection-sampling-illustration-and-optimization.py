import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_func
from scipy.special import beta as beta_val

a = 5
b = 2
bins = 30


def f_a_b(a, b):
    return beta_func.pdf(x, a, b)


def f_a_b_x(a, b, x):
    return beta_func.pdf(x, a, b)


# Plot the theoretical Beta(a, b) probability density
x = np.linspace(0, 1, 300)
y = f_a_b(a, b)
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="mediumblue", linewidth=2)
plt.title("Theoretical Beta(a, b) pdf", fontsize=14)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.8)
plt.show()

# Simulation of Beta(a, 1) distribution
n = 10000
U = rd.random(n)
V_a_1 = U**(1/a)
plt.figure(figsize=(8, 4))
plt.plot(x, f_a_b(a, 1), label=r'Theoretical density $f_{a,1}(x)$',
         color="mediumblue", linewidth=2)
plt.hist(V_a_1, bins=bins, density=True,
         label='Histogram of simulated values', color="firebrick", alpha=0.8)
plt.title(r'Simulation of the Beta(a, 1) distribution', fontsize=14)
plt.legend(fontsize=12)

# Simulation of Beta(1, b) distribution
U = rd.random(n)
V_1_b = 1-(1-U)**(1/b)
plt.figure(figsize=(8, 4))
plt.plot(x, f_a_b(1, b), label=r'Theoretical density $f_{1,b}(x)$',
         color="mediumblue", linewidth=2)
plt.hist(V_1_b, bins=bins, density=True,
         label='Histogram of simulated values', color="limegreen", alpha=0.8)
plt.title(r'Simulation of the Beta(1, b) distribution', fontsize=14)
plt.legend(fontsize=12)

# Existence of bounding constants for the Beta(a, b) density
B_a_b = beta_val(a, b)
k_a = 1/(a*B_a_b)
k_b = 1/(b*B_a_b)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="mediumblue", linewidth=2)
plt.plot(x, f_a_b(a, 1)*k_a, label=r'$f_{a,1}(x) * k_{a}$', color="firebrick")
plt.plot(x, f_a_b(1, b)*k_b, label=r'$f_{1,b}(x) * k_{b}$', color="limegreen")
plt.title('Beta(a, b) density and its bounds', fontsize=14)
plt.grid(axis='y', alpha=0.8)
plt.legend(fontsize=12)


plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="mediumblue", linewidth=2)
plt.plot(x, f_a_b(a, 1)*k_a, label=r'$f_{a,1}(x) * k_{a}$', color="firebrick")
plt.plot(x, f_a_b(1, b)*k_b, label=r'$f_{1,b}(x) * k_{b}$', color="limegreen")
plt.title('Zoom: Beta(a, b) density and its bounds', fontsize=14)
plt.ylim(0, max(y)*1.05)
plt.grid(axis='y', alpha=0.8)
plt.legend(fontsize=12)

# Setting up the rejection sampling method based on f(a, 1)
######################################################################
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="mediumblue", linewidth=2)
plt.plot(x, f_a_b(a, 1)*k_a, label=r'$f_{a,1}(x) * k_{a}$', color="firebrick")
plt.title(r'Rejection sampling setup with proposal $f_{a,1}(x)$', fontsize=14)
plt.grid(axis='y', alpha=0.8)
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
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="mediumblue", linewidth=2)
plt.hist(list_X, bins=bins, density=True,
         label="Simulated approximation", color="firebrick", alpha=0.8)
plt.title(
    r'Approximation via rejection sampling using $f_{a,1}(x)$', fontsize=14)
plt.legend(fontsize=12)

# Computation and visualization of errors
######################################################################
######################################################################
# For the rejection sampling method based on f(a, 1)
plt.figure(figsize=(8, 4))

counts, bin_edges = np.histogram(list_X, bins=bins, density=True)[:2]
dz = bin_edges[1] - bin_edges[0]

error = 0
for i in range(len(bin_edges)-1):
    z = bin_edges[i] + dz / 2
    error += np.abs(counts[i] - f_a_b_x(a, b, z)) * dz
    plt.plot([z, z], [counts[i], f_a_b_x(a, b, z)],
             color="black", linestyle=":")

plt.plot(x, y, label=r'$f_{a,b}(x)$', color="mediumblue", linewidth=2)
plt.hist(list_X, bins=bins, density=True,
         label="Simulated approximation", color="firebrick", alpha=0.8)
plt.plot([], [], color="black", linestyle=":",
         label=f"Total error = {error:.5f}")
plt.title(
    r'Approximation via rejection sampling using $f_{a,1}(x)$ with errors', fontsize=14)
plt.legend(fontsize=12)


# Setting up the rejection sampling method based on f(1, b)
######################################################################
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="mediumblue", linewidth=2)
plt.plot(x, f_a_b(1, b)*k_b, label=r'$f_{1,b}(x) * k_{b}$', color="limegreen")
plt.title(r'Rejection sampling setup with proposal $f_{1,b}(x)$', fontsize=14)
plt.grid(axis='y', alpha=0.8)
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
plt.plot(x, y, label=r'$f_{a,b}(x)$', color="mediumblue", linewidth=2)
plt.hist(list_X, bins=bins, density=True,
         label="Simulated approximation", color="limegreen", alpha=0.8)
plt.title(
    r'Approximation via rejection sampling using $f_{1,b}(x)$', fontsize=14)
plt.legend(fontsize=12)

# Computation and visualization of errors
######################################################################
######################################################################
# For the rejection sampling method based on f(1, b)
plt.figure(figsize=(8, 4))

counts, bin_edges = np.histogram(list_X, bins=bins, density=True)[:2]
dz = bin_edges[1] - bin_edges[0]

error = 0
for i in range(len(bin_edges)-1):
    z = bin_edges[i] + dz / 2
    error += np.abs(counts[i] - f_a_b_x(a, b, z)) * dz
    plt.plot([z, z], [counts[i], f_a_b_x(a, b, z)],
             color="black", linestyle=":")

plt.plot(x, y, label=r'$f_{a,b}(x)$', color="mediumblue", linewidth=2)
plt.hist(list_X, bins=bins, density=True,
         label="Simulated approximation", color="limegreen", alpha=0.8)
plt.plot([], [], color="black", linestyle=":",
         label=f"Total error = {error:.5f}")
plt.title(
    r'Approximation via rejection sampling using $f_{1,b}(x)$ with errors', fontsize=14)
plt.legend(fontsize=12)
