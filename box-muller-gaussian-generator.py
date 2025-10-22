import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define the PDF that we aim to approximate using the Box-Müller method
x = np.linspace(-5, 5, 100)
pdf = np.exp(-(x)**2/2) / np.sqrt(2 * np.pi)


# Approach 1: Box-Müller method using an exponential distribution and a uniform distribution on [0, 2π]
n = 10000
R = rd.exponential(2, n)
theta = rd.uniform(0, 2*np.pi, n)
X1 = np.sqrt(R)*np.cos(theta)
Y1 = np.sqrt(R)*np.sin(theta)

plt.plot(x, pdf, color='black', label='Theoretical PDF')
plt.hist(X1, bins=100, density=True, color='red',
         label='Box–Müller (approach 1)')
plt.title("Normal distribution generated using Box-Müller method")
plt.legend(loc='center left', bbox_to_anchor=(0.6, 0.87))


# Approach 2: Box-Müller variant using only uniform random variables in [0, 1]
n = 10000
U = rd.random(n)
V = rd.random(n)
R = -2*np.log(U)
theta = V*2*np.pi
X2 = np.sqrt(R)*np.cos(theta)
Y2 = np.sqrt(R)*np.sin(theta)
plt.figure()
plt.plot(x, pdf, color='black', label='Theoretical PDF')
plt.hist(X2, bins=100, density=True, color='red',
         label='Box–Müller (approach 2)')
plt.title("Normal distribution generated using Box-Müller method")
plt.legend(loc='center left', bbox_to_anchor=(0.6, 0.87))

# Empirical study of variable independence using a scatter plot
# -> Computation of the R² (coefficient of determination)
model = LinearRegression()
X2_mod = np.array(X2).reshape(-1, 1)
Y2_mod = np.array(Y2).reshape(-1, 1)
model.fit(X2_mod, Y2_mod)
Y2_pred = model.predict(X2_mod)
r2 = r2_score(Y2, Y2_pred)

slope = model.coef_.item()
intercept = model.intercept_.item()
plt.figure()
plt.scatter(X2, Y2, s=10, color='blue', label='Data points')
plt.plot(X2, Y2_pred, color='red', linewidth=2, label=f"Regression line: y = {
         slope:.3f}x + {intercept:.3f}")

plt.xlabel('Values of X')
plt.ylabel('Values of Y')
plt.axis('equal')
plt.title(
    f"Scatter plot for empirical study of variable independence\nR² = {r2:.7f}")
plt.legend()


# Illustrating Point Distribution with Concentric Circles
# -> Calculation and sorting of distances from the origin for each point
dist = np.sqrt(X2**2+Y2**2)
dist = np.sort(dist)

nb_quantile = 5
radius = np.zeros(nb_quantile-1)
increment = int(len(dist)/nb_quantile)

for j in range(len(radius)):
    radius[j] = dist[(j+1)*increment]

nb_point = 0
perc = 0
for i in range(len(radius)):
    plt.figure()

    fig, ax = plt.subplots()
    ax.scatter(X2, Y2, s=5, color='black', label='Data points')
    ax.set_aspect('equal', adjustable='box')
    colors = cm.autumn(np.linspace(0, 1, len(radius)))

    for r, c in zip(radius, colors):
        circle = plt.Circle((0, 0), r, fill=False, color=c,
                            linewidth=5, linestyle=':')
        ax.add_artist(circle)

    # Adding the data ---
    nb_point += int(len(dist)/nb_quantile)
    perc += 1/nb_quantile * 100

    plt.plot([], [], color=colors[i], linestyle=':', linewidth=5,
             label=f"Points included: {nb_point}\
             \nPercentage included: {perc:.1f}%")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.title(f"Radial distribution visualization — Circle {
              i+1} (Radius = {radius[i]:.5f})")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.87))
    # ---

    plt.xlim(-radius[i]*1.1, radius[i]*1.1)
    plt.ylim(-radius[i]*1.1, radius[i]*1.1)
    plt.pause(1)

# Creation of the final visualization
plt.figure()

fig, ax = plt.subplots()
ax.scatter(X2, Y2, s=5, color='black', label='Data points')
ax.set_aspect('equal', adjustable='box')
colors = cm.autumn(np.linspace(0, 1, len(radius)))

for r, c in zip(radius, colors):
    circle = plt.Circle((0, 0), r, fill=False, color=c,
                        linewidth=3, linestyle=':')
    ax.add_artist(circle)

# Title and legend ---
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
plt.title(f"Radial distribution — Final visualization (Max distance = {
          dist[-1]:.5f})")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.87))
# ---
