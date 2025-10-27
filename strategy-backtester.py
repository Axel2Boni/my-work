import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# for Apple Inc. (AAPL)

start_date = "2010-12-31"
end_date = "2024-12-31"
data = yf.download("AAPL", start=start_date, end=end_date)
df = data[["Close"]].copy()

# 1st plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, df, label="Closing price", color="black")
plt.title("AAPL price over time", fontsize=18)
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

##### Moving average crossover strategy #####
#####################################################################
#####################################################################
short_period = 5
long_period = 50
#####################################################################
#####################################################################

# Long-only strategy
df = data[["Close"]].copy()

df["A_short"] = df["Close"].rolling(window=short_period).mean()
df["A_long"] = df["Close"].rolling(window=long_period).mean()
df["Signal"] = np.where(df["A_short"] > df["A_long"],
                        1, 0)  # Long-only strategy
df["Position_Change"] = df["Signal"].diff()
df["Returns"] = df["Close"].pct_change()
df["Strategy_Returns"] = (df["Returns"] * df["Signal"].shift(1))

# Earnings computation
df["Cumulative_Strategy"] = (1 + df["Strategy_Returns"]).cumprod()
df["Cumulative_B&H"] = (1 + df["Returns"]).cumprod()

# Performance visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Cumulative_B&H"],
         label="Buy & Hold", linewidth=2, color="black", linestyle="--")
plt.plot(df.index, df["Cumulative_Strategy"],
         label="Long-only MA Crossover Strategy", linewidth=2, color="blue")
sw_at_end = df["Cumulative_Strategy"].iat[-1] - df["Cumulative_B&H"].iat[-1]
plt.plot([], [], marker='o', color='red', linestyle='None', markersize=8, label= f"Strategy's worthiness = {sw_at_end:.3f}")
plt.title(f"#1: MA Crossover Strategy (with {short_period} and {long_period} days) vs. B&H", fontsize=18)
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend(fontsize=14)
plt.grid(axis='y', alpha=0.8)
plt.show()


# Long-short strategy
df = data[["Close"]].copy()

df["A_short"] = df["Close"].rolling(window=short_period).mean()
df["A_long"] = df["Close"].rolling(window=long_period).mean()
df["Signal"] = np.where(df["A_short"] > df["A_long"],
                        1, -1)  # Long-short strategy
df["Position_Change"] = df["Signal"].diff()
df["Returns"] = df["Close"].pct_change()
df["Strategy_Returns"] = (df["Returns"] * df["Signal"].shift(1))

# Earnings computation
df["Cumulative_Strategy"] = (1 + df["Strategy_Returns"]).cumprod()
df["Cumulative_B&H"] = (1 + df["Returns"]).cumprod()

# Performance visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Cumulative_B&H"],
         label="Buy & Hold", linewidth=2, color="black", linestyle="--")
plt.plot(df.index, df["Cumulative_Strategy"],
         label="Long-short MA Crossover Strategy", linewidth=2, color="blue")
sw_at_end = df["Cumulative_Strategy"].iat[-1] - df["Cumulative_B&H"].iat[-1]
plt.plot([], [], marker='o', color='red', linestyle='None', markersize=8, label= f"Strategy's worthiness = {sw_at_end:.3f}")
plt.title(f"#1: MA Crossover Strategy (with {short_period} and {long_period} days) vs. B&H", fontsize=18)
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend(fontsize=14)
plt.grid(axis='y', alpha=0.8)
plt.show()
