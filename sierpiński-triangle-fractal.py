import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import time

# Create the first triangle
plt.plot([0, 1], [0, 0], color='black', linewidth=2)
plt.plot([0, 0.5], [0, 1], color='black', linewidth=2)
plt.plot([0.5, 1], [1, 0], color='black', linewidth=2)

# Create a point in the triangle
x, y = rd.random(2)
while y > -2*abs(x-0.5)+1:
    x, y = rd.random(2)
plt.scatter(x, y, color='red', s=30)

# Select a random vertex of the triangle
u = rd.random()
if u < 1/3:
    vertex = [0, 0]
elif u < 2/3:
    vertex = [0.5, 1]
else:
    vertex = [1, 0]

plt.scatter(0, 0, color='black', s=50)
plt.scatter(0.5, 1, color='black', s=50)
plt.scatter(1, 0, color='black', s=50)
plt.scatter(vertex[0], vertex[1], color='blue', s=50)

# Connect the point to this vertex
plt.plot([vertex[0], x], [vertex[1], y], color='grey',
         linestyle='dashed', linewidth=2)

# Find the middle of this segment
plt.scatter((vertex[0]+x)/2, (vertex[1]+y)/2, color='red', s=50)

# Step by step visualization
##############################################################################
##############################################################################

x, y = rd.random(2)
while y > -2*abs(x-0.5)+1:
    x, y = rd.random(2)

X = [x]
Y = [y]
n = 5
waiting_time = 0.2

for i in range(n):
    # Triangle + Points
    plt.title(f"Visualization - Step 1\niteration n°{i+1}/{n}", fontsize=14)
    plt.plot([0, 1], [0, 0], color='black', linewidth=2)
    plt.plot([0, 0.5], [0, 1], color='black', linewidth=2)
    plt.plot([0.5, 1], [1, 0], color='black', linewidth=2)
    plt.scatter(X[:-1], Y[:-1], color='orange', s=30)
    plt.scatter(X[-1], Y[-1], color='red', s=100)  # last point
    plt.scatter(0, 0, color='black', s=50)
    plt.scatter(0.5, 1, color='black', s=50)
    plt.scatter(1, 0, color='black', s=50)
    plt.show()
    time.sleep(waiting_time)

    # Triangle + Points + Vertex
    plt.title(f"Visualization - Step 2\niteration n°{i+1}/{n}", fontsize=14)
    plt.plot([0, 1], [0, 0], color='black', linewidth=2)
    plt.plot([0, 0.5], [0, 1], color='black', linewidth=2)
    plt.plot([0.5, 1], [1, 0], color='black', linewidth=2)
    plt.scatter(X[:-1], Y[:-1], color='orange', s=30)
    plt.scatter(X[-1], Y[-1], color='red', s=100)  # last point
    u = rd.random()
    if u < 1/3:
        vertex = [0, 0]
    elif u < 2/3:
        vertex = [0.5, 1]
    else:
        vertex = [1, 0]
    plt.scatter(0, 0, color='black', s=50)
    plt.scatter(0.5, 1, color='black', s=50)
    plt.scatter(1, 0, color='black', s=50)
    plt.scatter(vertex[0], vertex[1], color='blue', s=100)
    plt.show()
    time.sleep(waiting_time)

    # Triangle + Points + Vertex + Line
    plt.title(f"Visualization - Step 3\niteration n°{i+1}/{n}", fontsize=14)
    plt.plot([0, 1], [0, 0], color='black', linewidth=2)
    plt.plot([0, 0.5], [0, 1], color='black', linewidth=2)
    plt.plot([0.5, 1], [1, 0], color='black', linewidth=2)
    plt.scatter(X[:-1], Y[:-1], color='orange', s=30)
    plt.scatter(X[-1], Y[-1], color='red', s=100)  # last point
    plt.scatter(0, 0, color='black', s=50)
    plt.scatter(0.5, 1, color='black', s=50)
    plt.scatter(1, 0, color='black', s=50)
    plt.scatter(vertex[0], vertex[1], color='blue', s=100)
    plt.plot([vertex[0], X[-1]], [vertex[1], Y[-1]], color='grey',
             linestyle='dashed', linewidth=2)
    plt.show()
    time.sleep(waiting_time)

    # Triangle + Points + Vertex + Line + Middle
    plt.title(f"Visualization - Step 4\niteration n°{i+1}/{n}", fontsize=14)
    X.append((vertex[0]+X[-1])/2)
    Y.append((vertex[1]+Y[-1])/2)
    plt.plot([0, 1], [0, 0], color='black', linewidth=2)
    plt.plot([0, 0.5], [0, 1], color='black', linewidth=2)
    plt.plot([0.5, 1], [1, 0], color='black', linewidth=2)
    plt.scatter(X[:-1], Y[:-1], color='orange', s=30)
    plt.scatter(X[-1], Y[-1], color='red', s=100)  # last point
    plt.scatter(0, 0, color='black', s=50)
    plt.scatter(0.5, 1, color='black', s=50)
    plt.scatter(1, 0, color='black', s=50)
    plt.scatter(vertex[0], vertex[1], color='blue', s=100)
    plt.plot([vertex[0], X[-2]], [vertex[1], Y[-2]], color='grey',
             linestyle='dashed', linewidth=2)
    plt.show()
    time.sleep(waiting_time)

# Fractal computation
##############################################################################
##############################################################################

for n in [10 ** i for i in range(2, 7)]:
    start_time = time.time()
    x, y = rd.random(2)
    while y > -2*abs(x-0.5)+1:
        x, y = rd.random(2)

    X = [x]
    Y = [y]
    for i in range(n):
        u = rd.random()
        if u < 1/3:
            vertex = [0, 0]
        elif u < 2/3:
            vertex = [0.5, 1]
        else:
            vertex = [1, 0]
        X.append((vertex[0]+X[-1])/2)
        Y.append((vertex[1]+Y[-1])/2)
    end_time = time.time()

    plt.plot([0, 1], [0, 0], color='black', linewidth=2)
    plt.plot([0, 0.5], [0, 1], color='black', linewidth=2)
    plt.plot([0.5, 1], [1, 0], color='black', linewidth=2)
    plt.scatter(X[3:], Y[3:], color='orange', s=2)

    plt.title(f"Time = {end_time-start_time:.3f}s | Points = {n:,}")
    plt.show()
