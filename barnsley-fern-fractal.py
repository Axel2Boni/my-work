import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import time

# Barnsley fern computation


def barnsley_fern(n):
    trans = [
        # a, b, c, d, e, f, probability
        (0.00, 0.00, 0.00, 0.16, 0.00, 0.00, 0.01),
        (0.85, 0.04, -0.04, 0.85, 0.00, 1.60, 0.85),
        (0.20, -0.26, 0.23, 0.22, 0.00, 1.60, 0.07),
        (-0.15, 0.28, 0.26, 0.24, 0.00, 0.44, 0.07),
    ]

    x, y = 0, 0
    X, Y = np.zeros(n), np.zeros(n)

    for i in range(n):
        r = rd.random()
        trans_selection = 0
        for (a, b, c, d, e, f, p) in trans:
            trans_selection += p
            if r <= trans_selection:
                x, y = a * x + b * y + e, c * x + d * y + f
                break
        X[i], Y[i] = x, y

    return X, Y


# Visualize the Barnsley fern
s = 20
for n in [10**i for i in range(2, 7)]:
    start_time = time.time()
    X, Y = barnsley_fern(n)
    plt.figure(figsize=(6, 10))
    plt.scatter(X, Y, s=s, color='green')
    s = s/4
    plt.axis("off")
    end_time = time.time()
    plt.title(f"Barnsley Fern (Fractal)\ns = {
              4*s:.1f} | Points = {n:,} | Time = {end_time-start_time:.3f}s", fontsize=14)
    plt.show()
