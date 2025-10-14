import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

n = 100000
R = rd.exponential(2,n)
theta = rd.uniform(0,2*np.pi,n)
X = np.sqrt(R)*np.cos(theta)
Y = np.sqrt(R)*np.sin(theta)

#plt.hist(X, bins=100, density=True, color='black')
#plt.hist(Y, bins=100, density=True, color='black')
plt.scatter(X,Y, color='red', marker='*')





