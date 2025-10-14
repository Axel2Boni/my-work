import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

for i in range(0,11):
    plt.plot([i,i],[0,10], color='black')

nb_needle = 10000
counter = 0
for i in range(0,nb_needle):
    X1 = rd.uniform(0,10)
    X2 = rd.uniform(0,10)
    theta = rd.uniform(0,np.pi)
    
    Y1 = X1 + np.sin(theta)
    Y2 = X2 + np.cos(theta)
    
    #compteur
    if np.floor(X1) == np.floor(Y1):
        plt.plot([X1,Y1],[X2,Y2], color = 'black')
    else:
        counter += 1  
        plt.plot([X1,Y1],[X2,Y2], color = 'red')

#RESULTS
#with counter/nb_needle approx. 2/np.pi
approx_pi = 2*(1/(counter/nb_needle))
print('An approximation of pi is given by: ', approx_pi)
print('Error: ', np.pi - approx_pi)