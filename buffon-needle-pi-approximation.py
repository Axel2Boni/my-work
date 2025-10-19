import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

for i in range(0, 11):
    plt.plot([i, i], [0, 10], color='black')

counter = 0
nb_needle = 10000
perc_needle_displayed = 0.05
for i in range(0, nb_needle):
    X1 = rd.uniform(0, 10)
    X2 = rd.uniform(0, 10)
    theta = rd.uniform(0, np.pi)
    p = rd.uniform()

    Y1 = X1 + np.sin(theta)
    Y2 = X2 + np.cos(theta)

    # counter
    if np.floor(X1) == np.floor(Y1):
        if p < perc_needle_displayed:
            plt.plot([X1, Y1], [X2, Y2], color='black')
    else:
        counter += 1
        if p < perc_needle_displayed:
            plt.plot([X1, Y1], [X2, Y2], color='red')
plt.figure()

# Results
# with counter/nb_needle approx. 2/np.pi
approx_pi = 2*(1/(counter/nb_needle))
print('An approximation of pi is given by: ', approx_pi)
print('Error: ', np.pi - approx_pi)
print('Percentage of needle displayed: ', perc_needle_displayed*100, '%')


# Calculation of errors

def approx_pi_with_n_needle(n):
    nb_needle = n
    counter = 0
    for i in range(0, nb_needle):
        X1 = rd.uniform(0, 10)
        theta = rd.uniform(0, np.pi)
        Y1 = X1 + np.sin(theta)
        # counter
        if np.floor(X1) != np.floor(Y1):
            counter += 1

    approx_pi = 2*(1/(counter/nb_needle))
    return approx_pi


n = np.array([10, 30, 100] + [10**j*i for j in range(2, 5)
             for i in range(2, 11)])
list_approx_pi = np.array([approx_pi_with_n_needle(k) for k in n])
list_error_pi = np.abs(list_approx_pi - np.pi)

plt.plot(n[:12], list_error_pi[:12], color='black',
         label='Approximation error')
plt.plot([0, n[11]], [0, 0], color='red', linewidth='2', linestyle='--')
plt.legend()
plt.figure()


# Error negligibility analysis
plt.loglog(n, list_error_pi, color='black', label='Approximation error')
negl_index1 = 1/3
plt.loglog(n, 1/n**negl_index1, color='turquoise', linewidth='2',
           linestyle=':', label=rf"$n^{{-{negl_index1:.2f}}}$")
negl_index2 = 1/2
plt.loglog(n, 1/n**negl_index2, color='darkcyan', linewidth='3',
           linestyle=':', label=rf"$n^{{-{negl_index2}}}$")
negl_index3 = 1
plt.loglog(n, 1/n**negl_index3, color='darkslategray', linewidth='2',
           linestyle=':', label=rf"$n^{{-{negl_index3}}}$")
plt.legend()
