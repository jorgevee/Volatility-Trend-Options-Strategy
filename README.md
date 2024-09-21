# Volatility-Trend Options Strategy

This repository contains a Python script that implements an intraday options trading strategy based on volatility and trend regimes. The strategy leverages the VIX index as a proxy for market volatility and the SPY ETF to identify the prevailing market trend.

## Strategy Description

The strategy aims to capitalize on short-term market movements by dynamically selecting out-of-the-money (OTM) option spreads based on the identified volatility and trend regimes.

**Regime Determination:**

- **Volatility Regime:** Determined using a moving average crossover on the VIX index.
- **Trend Regime:** Determined using a moving average crossover on the SPY ETF.

**Option Spread Selection:**

- At 9:35 AM ET, the strategy fetches SPXW (weekly) option contracts expiring on the current day.
- Based on the trend regime:
  - **Uptrend:** Selects an OTM call spread.
  - **Downtrend:** Selects an OTM put spread.

**Spread Execution and Monitoring:**

- The strategy fetches option quotes and calculates the mid-price for both legs of the selected spread.
- It then executes the spread and continuously monitors its performance by fetching the latest option quotes.
- Live PnL, trade details, and the percentage distance of the underlying price from the short strike are printed to the console.

## Tools Used

- **Python:** Programming language for implementing the strategy.
- **Pandas:** Data manipulation and analysis library.
- **NumPy:** Numerical computing library.
- **Matplotlib:** Plotting library for visualization.
- **Requests:** HTTP library for fetching data from the Polygon.io API.
- **Pandas Market Calendars:** Library for accessing trading calendars.
- **Polygon.io API:** Data provider for historical and intraday market data, as well as options data.

## Potential Improvements

**Technical Improvements:**

- **Enhance Regime Determination:** Implement more sophisticated methods for regime determination, such as Hidden Markov Models or machine learning algorithms.
- **Refine Expected Move Calculation:** Incorporate factors like implied volatility, historical volatility, and market sentiment into the expected move calculation.
- **Dynamic Spread Width:** Adjust the spread width based on market conditions.
- **Simulate Realistic Execution:** Use a more realistic order execution model that considers slippage and order book dynamics.
- **Implement Robust Error Handling:** Improve error handling to gracefully handle API request failures and other potential errors.

**Financial Improvements:**

- **Incorporate Slippage and Commissions:** Explicitly factor in slippage and commissions into the PnL calculation.
- **Conduct Thorough Backtesting:** Backtest the strategy over a significant historical period to assess its performance and identify potential weaknesses.
- **Explore Multiple Timeframes:** Incorporate data from different timeframes to gain a broader perspective on market trends and volatility.
- **Implement Risk Management:** Add stop-loss and take-profit orders to limit potential losses and secure profits.
- **Paper Trading:** Thoroughly test the strategy in a paper trading environment before deploying it with real capital.

## Disclaimer

This strategy is for educational and informational purposes only and should not be considered financial advice. Options trading involves significant risk and is not suitable for all investors. Consult with a qualified financial advisor before making any investment decisions.
