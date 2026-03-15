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

ticker = "^FCHI"
asset = dict_tickers.get(ticker, "Unknown company")

start_date = "2020-12-31"
end_date = "2022-12-31"
data = yf.download(ticker, start=start_date, end=end_date)
df = data[["Close"]].copy()
df["Returns"] = df["Close"].pct_change()
df["Cumulative_B&H"] = (1 + df["Returns"]).cumprod()

# 1st plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Closing price",
         color="black", linewidth=2)
plt.title(f"Price fluctuations of {asset} over time", fontsize=18)
plt.ylabel("Closing Price (USD)")
plt.legend(fontsize=14)
plt.grid(True)
plt.show()


#####################################################################
#####################################################################
##### Moving average crossover strategy #####
short_period = 2
long_period = 35
#####################################################################
#####################################################################

# Long-only strategy
#####################################################################
df["A_short"] = df["Close"].rolling(window=short_period).mean()
df["A_long"] = df["Close"].rolling(window=long_period).mean()
df["Signal"] = np.where(df["A_short"] > df["A_long"],
                        1, 0)  # Long-only strategy

# Earnings computation
df["Strategy_Returns"] = df["Returns"] * df["Signal"].shift(1)
df["Cumulative_Strategy"] = (1 + df["Strategy_Returns"]).cumprod()

# Performance visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Cumulative_B&H"],
         label="Buy & Hold strategy", linewidth=2, color="black", linestyle="--")
plt.plot(df.index, df["Cumulative_Strategy"],
         label="Long-only MAC strategy", linewidth=2, color="blue")
sw_at_end = df["Cumulative_Strategy"].iat[-1] - df["Cumulative_B&H"].iat[-1]
plt.plot([df.index[-1], df.index[-1]], [df["Cumulative_Strategy"].iat[-1], df["Cumulative_B&H"].iat[-1]], marker='o',
         markersize=8, color='red', linestyle=':', linewidth=1.5, label=f"Strategy's worthiness = {sw_at_end:.3f}")
plt.title(f"1)a) Moving Average Crossover Strategy for {asset} (with {
          short_period} and {long_period} days)", fontsize=18)
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend(fontsize=14)
plt.grid(axis='y', alpha=0.8)
plt.show()

# Long-short strategy
#####################################################################
df["A_short"] = df["Close"].rolling(window=short_period).mean()
df["A_long"] = df["Close"].rolling(window=long_period).mean()
df["Signal"] = np.where(df["A_short"] > df["A_long"],
                        1, -1)  # Long-short strategy

# Earnings computation
df["Strategy_Returns"] = (df["Returns"] * df["Signal"].shift(1))
df["Cumulative_Strategy"] = (1 + df["Strategy_Returns"]).cumprod()

# Performance visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Cumulative_B&H"],
         label="Buy & Hold strategy", linewidth=2, color="black", linestyle="--")
plt.plot(df.index, df["Cumulative_Strategy"],
         label="Long-short MAC strategy", linewidth=2, color="blue")
sw_at_end = df["Cumulative_Strategy"].iat[-1] - df["Cumulative_B&H"].iat[-1]
plt.plot([df.index[-1], df.index[-1]], [df["Cumulative_Strategy"].iat[-1], df["Cumulative_B&H"].iat[-1]], marker='o',
         markersize=8, color='red', linestyle=':', linewidth=1.5, label=f"Strategy's worthiness = {sw_at_end:.3f}")
plt.title(f"1)b) Moving Average Crossover Strategy for {asset} (with {
          short_period} and {long_period} days)", fontsize=18)
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend(fontsize=14)
plt.grid(axis='y', alpha=0.8)
plt.show()

# Optimization of long-short strategy
#####################################################################
sw_cross_tabulation = np.zeros((10, 10))
short_period_list = np.arange(1, 11, 1, dtype=int)  # row
long_period_list = np.arange(10, 56, 5, dtype=int)  # column

for row_index, row_value in enumerate(short_period_list):
    for column_index, column_value in enumerate(long_period_list):
        df["A_short"] = df["Close"].rolling(window=row_value).mean()
        df["A_long"] = df["Close"].rolling(window=column_value).mean()
        df["Signal"] = np.where(df["A_short"] > df["A_long"], 1, -1)
        df["Position_Change"] = df["Signal"].diff()
        df["Strategy_Returns"] = df["Returns"] * df["Signal"].shift(1)
        df["Cumulative_Strategy"] = (1 + df["Strategy_Returns"]).cumprod()
        sw_cross_tabulation[row_index,
                            column_index] = df["Cumulative_Strategy"].iat[-1]

max_index = np.unravel_index(
    np.argmax(sw_cross_tabulation), sw_cross_tabulation.shape)

print(f'For long-short strategy with {asset}; Optimization: short =',
      short_period_list[max_index[0]], 'and long =', long_period_list[max_index[1]])

#####################################################################
#####################################################################
##### Relative Strength Index (RSI) Strategy #####
period_length = 14
buy_signal_value = 30
sell_signal_value = 70
#####################################################################
#####################################################################


def rsi_computation(df, window=period_length):
    delta_price = df['Close'].diff()
    gain = (delta_price.where(delta_price > 0, 0))
    loss = (-delta_price.where(delta_price < 0, 0))
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = (rs / (1 + rs))*100
    return rsi


df['RSI'] = rsi_computation(df).squeeze()

# RSI visualization
#####################################################################
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["RSI"],
         label="RSI", linewidth=2, color="purple")
plt.axhline(y=buy_signal_value, label="Buy signal (asset oversold)",
            linewidth=5, linestyle=':', color='green')
plt.axhline(y=sell_signal_value, label="Sell signal (asset overbought)",
            linewidth=5, linestyle=':', color='red')
plt.title("RSI and Trading signals", fontsize=18)
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend(fontsize=14)
plt.grid(axis='y', alpha=0.8)
plt.show()
#####################################################################

df['Signal_RSI'] = 0
df.loc[df['RSI'] < buy_signal_value, 'Signal_RSI'] = 1  # Buy signal
df.loc[df['RSI'] > sell_signal_value, 'Signal_RSI'] = -1  # Sell signal
df['Signal_RSI'] = df['Signal_RSI'].replace(0, np.nan).ffill()

# Earnings computation
df["RSI_Strategy_Returns"] = df["Returns"] * df["Signal_RSI"].shift(1)
df["Cumulative_RSI_Strategy"] = (1 + df["RSI_Strategy_Returns"]).cumprod()

# Performance visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Cumulative_B&H"],
         label="Buy & Hold strategy", linewidth=2, color="black", linestyle="--")
plt.plot(df.index, df["Cumulative_RSI_Strategy"],
         label="RSI strategy", linewidth=2, color="blue")
sw_at_end = df["Cumulative_RSI_Strategy"].iat[-1] - \
    df["Cumulative_B&H"].iat[-1]
plt.plot([df.index[-1], df.index[-1]], [df["Cumulative_RSI_Strategy"].iat[-1], df["Cumulative_B&H"].iat[-1]], marker='o',
         markersize=8, color='red', linestyle=':', linewidth=1.5, label=f"Strategy's worthiness = {sw_at_end:.3f}")
plt.title(f"2) Relative Strength Index Strategy for {
          asset} (period length = {period_length} days)", fontsize=18)
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend(fontsize=14)
plt.grid(axis='y', alpha=0.8)
plt.show()
