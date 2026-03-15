import numpy as np
import matplotlib.pyplot as plt

##############################################
##############################################
# Parameters:
S = 50  # Spot price
r = 0.1  # Risk-free rate
s = 0.4  # Volatility
T = 5/12  # Time to maturity

# From the CRR model:
n = 5  # for n steps
dt = T/n
u = np.exp(s*np.sqrt(dt))
d = np.exp(-s*np.sqrt(dt))
##############################################
##############################################

SP = np.zeros((n+1, n+1))  # Stock prices
for t in range(n+1):
    for l in range(t+1):
        SP[l, t] = S * u**(t-l) * d**l

# Construction of a replication portfolio:


def replication_portfolio_price(price_u, price_d):
    erdt = np.exp(r*dt)
    return np.exp(-r*dt)*(((erdt-d)*price_u + (-erdt+u)*price_d)/(u-d))


# Assuming a put option:
K = 50  # ATM option

P = np.zeros((n+1, n+1))  # Profit
for t in range(n, -1, -1):
    for l in range(t+1):
        if t == n:
            P[l, t] = max(K - SP[l, t], 0)
        else:
            P[l, t] = replication_portfolio_price(P[l, t+1], P[l+1, t+1])

##############################################
##############################################
# We define a function (following the same structure as above)
# to compute the ATM put price for different parameter values:


def pricing_put_option_CRR(n, S=50, K=50, r=0.1, s=0.4, T=5/12):
    dt = T/n
    u = np.exp(s*np.sqrt(dt))
    d = np.exp(-s*np.sqrt(dt))
    # Verify the no-arbitrage condition
    if not (d < np.exp(r*dt) < u):
        raise ValueError(
            "No-arbitrage condition violated: ensure d < exp(r*dt) < u.")

    SP = np.zeros((n+1, n+1))  # Stock prices
    for t in range(n+1):
        for l in range(t+1):
            SP[l, t] = S * u**(t-l) * d**l

    def replication_portfolio_price(price_u, price_d):
        erdt = np.exp(r*dt)
        return np.exp(-r*dt)*(((erdt-d)*price_u + (-erdt+u)*price_d)/(u-d))

    P = np.zeros((n+1, n+1))  # Profit
    for t in range(n, -1, -1):
        for l in range(t+1):
            if t == n:
                P[l, t] = max(K - SP[l, t], 0)
            else:
                P[l, t] = replication_portfolio_price(P[l, t+1], P[l+1, t+1])
    return P[0, 0]

##############################################
##############################################


n_values = [i for i in range(2, 150)]
prices = [pricing_put_option_CRR(n, r=0.5) for n in n_values]
option_price_approx = pricing_put_option_CRR(1000)

plt.figure(figsize=(12, 4))
plt.plot(n_values, prices, marker='o', ms='4')
plt.axhline(y=option_price_approx, ls='--', color='black',
            label=f"y={option_price_approx:.4f}")
plt.title("Convergence of CRR option prices")
plt.legend()
plt.grid(True)
plt.show()

# Due to the observed odd–even oscillatory behavior,
# a more informative visualization can be obtained by adjusting the plot.

n_values = [i for i in range(2, 200, 2)]
prices = [pricing_put_option_CRR(n) for n in n_values]

plt.figure(figsize=(8, 4))
plt.plot(n_values, prices)
plt.axhline(y=option_price_approx, ls='--', color='black',
            label=f"y={option_price_approx:.4f}")
plt.title("Convergence of CRR option prices (Even Number of Steps Only)")
plt.legend()
plt.grid(True)
plt.show()
