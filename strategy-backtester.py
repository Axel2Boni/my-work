import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

dict_tickers = {
    "^FCHI": "CAC 40",
    "AAPL": "Apple Inc.",
    "GC=F": "Gold",
    "BTC-USD": "Bitcoin"
}

ticker = "BTC-USD"
asset = dict_tickers.get(ticker, "Unknown company")

start_date = "2010-12-31"
end_date = "2024-12-31"
data = yf.download(ticker, start=start_date, end=end_date)
df = data[["Close"]].copy()

# 1st plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, df, label="Closing price", color="black")
plt.title(f"{asset} price over time", fontsize=18)
plt.ylabel("Closing Price (USD)")
plt.legend(fontsize=14)
plt.grid(True)
plt.show()

##### Moving average crossover strategy #####
#####################################################################
#####################################################################
short_period = 1
long_period = 40
#####################################################################
#####################################################################

# Long-only strategy
#####################################################################
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
plt.plot([], [], marker='o', color='red', linestyle='None',
         markersize=8, label=f"Strategy's worthiness = {sw_at_end:.3f}")
plt.title(f"#1: MA Crossover Strategy for {asset} (with {
          short_period} and {long_period} days)", fontsize=18)
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend(fontsize=14)
plt.grid(axis='y', alpha=0.8)
plt.show()

# Long-short strategy
#####################################################################
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
plt.plot([], [], marker='o', color='red', linestyle='None',
         markersize=8, label=f"Strategy's worthiness = {sw_at_end:.3f}")
plt.title(f"#1: MA Crossover Strategy for {asset} (with {
          short_period} and {long_period} days)", fontsize=18)
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend(fontsize=14)
plt.grid(axis='y', alpha=0.8)
plt.show()

# Optimization of long-short strategy
sw_cross_tabulation = np.zeros((10, 10))
short_period_list = np.arange(1, 11, 1, dtype=int)  # row
long_period_list = np.arange(10, 56, 5, dtype=int)  # column

for row_index, row_value in enumerate(short_period_list):
    for column_index, column_value in enumerate(long_period_list):
        df = data[["Close"]].copy()
        df["A_short"] = df["Close"].rolling(window=row_value).mean()
        df["A_long"] = df["Close"].rolling(window=column_value).mean()
        df["Signal"] = np.where(df["A_short"] > df["A_long"], 1, -1)
        df["Position_Change"] = df["Signal"].diff()
        df["Returns"] = df["Close"].pct_change()
        df["Strategy_Returns"] = (df["Returns"] * df["Signal"].shift(1))
        df["Cumulative_Strategy"] = (1 + df["Strategy_Returns"]).cumprod()
        sw_cross_tabulation[row_index,
                            column_index] = df["Cumulative_Strategy"].iat[-1]

max_index = np.unravel_index(
    np.argmax(sw_cross_tabulation), sw_cross_tabulation.shape)

print(f'For long-short strategy with {asset}; Optimization: short =',
      short_period_list[max_index[0]], 'and long =', long_period_list[max_index[1]])
#####
