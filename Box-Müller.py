import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

# Define the PDF that we aim to approximate using the Box-Müller method
x = np.linspace(-5, 5, 100)
pdf = np.exp(-(x)**2/2) / np.sqrt(2 * np.pi)

# Approach 1: Box-Müller method using an exponential distribution and a uniform distribution over [0, 2π]
n = 10000
R = rd.exponential(2, n)
theta = rd.uniform(0, 2*np.pi, n)
X1 = np.sqrt(R)*np.cos(theta)
Y1 = np.sqrt(R)*np.sin(theta)

plt.plot(x, pdf, color='black', label='Th. PDF')
plt.hist(X1, bins=100, density=True, color='blue',
         label='Box-Müller approach1')
plt.legend()

# Approach 2: Box-Müller variant using only uniform random variables in [0, 1]
n = 10000
U = rd.random(n)
V = rd.random(n)
R = -2*np.log(U)
X2 = np.sqrt(R)*np.cos(theta)
Y2 = np.sqrt(R)*np.sin(theta)
theta = V*2*np.pi
plt.plot(x, pdf, color='black', label='Th. PDF')
plt.hist(X2, bins=100, density=True, color='red', label='Box-Müller approach2')
plt.legend()
