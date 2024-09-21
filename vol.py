# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import logging
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import  timezone
import pytz
# Configuration Management (using environment variables)
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

# Logging setup
logging.basicConfig(filename="options_trading.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
CALENDAR = get_calendar("NYSE")
TICKER = "I:SPX"
INDEX_TICKER = "I:VIX1D"
OPTIONS_TICKER = "SPX"
BASE_URL = "https://api.polygon.io"
MAX_WORKERS = 2  # Define max workers as a constant


# --- Data Retrieval Functions ---

@lru_cache(maxsize=128)
def get_historical_data(ticker, start_date, end_date):
    """Retrieves historical data from Polygon.io."""
    try:
        response = requests.get(
            f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
        )
        response.raise_for_status()
        data = pd.json_normalize(response.json()["results"]).set_index("t")
        data.index = pd.to_datetime(data.index, unit="ms", utc=True).tz_convert("America/New_York")
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP error fetching historical data for {ticker}: {e}")
    except (KeyError, ValueError) as e:
        logging.error(f"Error parsing historical data for {ticker}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error fetching historical data for {ticker}: {e}")
    return None


@lru_cache(maxsize=128)
def get_intraday_data(ticker, date):
    """Retrieves intraday (minute) data from Polygon.io."""
    try:
        response = requests.get(
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
        )
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = pd.json_normalize(response.json()["results"]).set_index("t")
        data.index = pd.to_datetime(data.index, unit="ms", utc=True).tz_convert(
            "America/New_York"
        )
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP error fetching intraday data for {ticker}: {e}")
        return None  # Or raise an exception if you want to stop execution
    except (KeyError, ValueError) as e:
        logging.error(f"Error parsing intraday data for {ticker}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error fetching intraday data for {ticker}: {e}")
        return None


def get_options_contracts(contract_type, expiration_date):
    """Retrieves option contracts data from Polygon.io."""
    try:
        contracts = pd.json_normalize(
            requests.get(
                f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={OPTIONS_TICKER}&contract_type={contract_type}&as_of={today}&expiration_date={expiration_date}&limit=1000&apiKey={POLYGON_API_KEY}"
            ).json()["results"]
        )
        return contracts
    except Exception as e:
        logging.error(f"Error fetching options contracts: {e}")
        return None


def get_option_quotes(ticker, start_timestamp, end_timestamp):
    """Retrieves option quotes data from Polygon.io."""
    try:
        quotes = pd.json_normalize(
            requests.get(
                f"https://api.polygon.io/v3/quotes/{ticker}?timestamp.gte={start_timestamp}×tamp.lt={end_timestamp}&order=asc&limit=5000&sort=timestamp&apiKey={POLYGON_API_KEY}"
            ).json()["results"]
        ).set_index("sip_timestamp")
        quotes.index = pd.to_datetime(quotes.index, unit="ns", utc=True).tz_convert(
            "America/New_York"
        )
        return quotes
    except Exception as e:
        logging.error(f"Error fetching option quotes for {ticker}: {e}")
        return None


# --- Data Processing Functions ---

def calculate_regime(data, window_short=20, window_long=60):
    """Calculates regime (volatility or trend) based on moving averages."""
    if data is None:
        logging.error("Data is None. Cannot calculate regime.")
        return None

    data["short_avg"] = data["c"].rolling(window=window_short).mean()
    data["long_avg"] = data["c"].rolling(window=window_long).mean()
    data["regime"] = data.apply(lambda row: 1 if row["c"] > row["short_avg"] else 0, axis=1)
    return data


# --- Trading Strategy Functions ---

def calculate_expected_move(index_price):
    """Calculates expected move based on VIX."""
    return (round((index_price / np.sqrt(252)), 2) / 100) * 0.50


def get_option_spread_details(
    price, expected_move, expiration_date, quote_timestamp, quote_minute_after_timestamp, side
):
    """Gets details for the option spread (call or put)."""

    if side == "Call":
        valid_options = get_options_contracts("call", expiration_date)
        valid_options = valid_options[valid_options["ticker"].str.contains("SPXW")].copy()
        valid_options["days_to_exp"] = (pd.to_datetime(valid_options["expiration_date"]) - pd.to_datetime(date)).dt.days
        valid_options["distance_from_price"] = abs(valid_options["strike_price"] - price)
        otm_options = valid_options[valid_options["strike_price"] >= price + (price * expected_move)]
    elif side == "Put":
        valid_options = get_options_contracts("put", expiration_date)
        valid_options = valid_options[valid_options["ticker"].str.contains("SPXW")].copy()
        valid_options["days_to_exp"] = (pd.to_datetime(valid_options["expiration_date"]) - pd.to_datetime(date)).dt.days
        valid_options["distance_from_price"] = abs(price - valid_options["strike_price"])
        otm_options = valid_options[valid_options["strike_price"] <= price - (price * expected_move)]

    short_option = otm_options.iloc[[0]]
    long_option = otm_options.iloc[[1]]

    short_strike = short_option["strike_price"].iloc[0]
    long_strike = long_option["strike_price"].iloc[0]

    # Use a ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=2) as executor:
        short_option_quotes_future = executor.submit(
            get_option_quotes, short_option['ticker'].iloc[0], quote_timestamp, quote_minute_after_timestamp
        )
        long_option_quotes_future = executor.submit(
            get_option_quotes, long_option['ticker'].iloc[0], quote_timestamp, quote_minute_after_timestamp
        )

        short_option_quotes = short_option_quotes_future.result()
        long_option_quotes = long_option_quotes_future.result()

    if short_option_quotes is None or long_option_quotes is None:
        return None  # Handle API errors gracefully

    short_option_quote = short_option_quotes.median(numeric_only=True).to_frame().copy().T
    short_option_quote["mid_price"] = (
        short_option_quote["bid_price"] + short_option_quote["ask_price"]
    ) / 2

    long_option_quote = long_option_quotes.median(numeric_only=True).to_frame().copy().T
    long_option_quote["mid_price"] = (
        long_option_quote["bid_price"] + long_option_quote["ask_price"]
    ) / 2

    spread_value = short_option_quote["mid_price"].iloc[0] - long_option_quote["mid_price"].iloc[0]

    return {
        "short_strike": short_strike,
        "long_strike": long_strike,
        "spread_value": spread_value,
        "short_option_ticker": short_option['ticker'].iloc[0],
        "long_option_ticker": long_option['ticker'].iloc[0],
    }



def execute_trading_strategy(today, expiration_date, trend_regime):
    """Executes the volatility-based options trading strategy."""
    try:
        if not CALENDAR.schedule(start_date=today, end_date=today).empty:
            underlying_data = get_intraday_data(TICKER, today)
            index_data = get_intraday_data(INDEX_TICKER, today)

            if underlying_data is None or index_data is None:
              logging.warning(f"Failed to retrieve data for {today}. Skipping this iteration.")
              return  # Skip the rest of the trading logic for this iteration

            index_price = index_data[index_data.index.time >= pd.Timestamp("09:35").time()][
                "c"
            ].iloc[0]
            price = underlying_data[underlying_data.index.time >= pd.Timestamp("09:35").time()][
                "c"
            ].iloc[0]

            expected_move = calculate_expected_move(index_price)

            minute_timestamp = (
                pd.to_datetime(today).tz_localize("America/New_York")
                + timedelta(
                    hours=pd.Timestamp("09:35").time().hour,
                    minutes=pd.Timestamp("09:35").time().minute,
                )
            )
            quote_timestamp = minute_timestamp.value
            minute_after_timestamp = (
                pd.to_datetime(today).tz_localize("America/New_York")
                + timedelta(
                    hours=pd.Timestamp("09:36").time().hour,
                    minutes=pd.Timestamp("09:36").time().minute,
                )
            )
            quote_minute_after_timestamp = minute_after_timestamp.value

            # Determine trade direction (Call or Put) based on trend_regime
            side = "Call" if trend_regime == 0 else "Put"

            spread_details = get_option_spread_details(
                price,
                expected_move,
                expiration_date,
                quote_timestamp,
                quote_minute_after_timestamp,
                side,
            )

            if spread_details is None:
                return  # Handle errors in getting spread details

            short_strike = spread_details["short_strike"]
            long_strike = spread_details["long_strike"]
            cost = spread_details["spread_value"]
            short_option_ticker = spread_details["short_option_ticker"]
            long_option_ticker = spread_details["long_option_ticker"]

            underlying_data["distance_from_short_strike"] = round(
                (
                    (short_strike - underlying_data["c"])
                    if side == "Put"
                    else (underlying_data["c"] - short_strike)
                )
                / underlying_data["c"].iloc[0]
                * 100,
                2,
            )

            # Use a ThreadPoolExecutor for parallel API calls
            with ThreadPoolExecutor(max_workers=2) as executor:
                updated_short_quotes_future = executor.submit(
                    get_option_quotes, short_option_ticker, None, None, limit=100, order="desc" # Assuming API supports these params
                )
                updated_long_quotes_future = executor.submit(
                    get_option_quotes, long_option_ticker, None, None, limit=100, order="desc" # Assuming API supports these params
                )

                updated_short_quotes = updated_short_quotes_future.result()
                updated_long_quotes = updated_long_quotes_future.result()


            if updated_short_quotes is None or updated_long_quotes is None:
                return  # Handle API errors gracefully

            updated_short_quotes["mid_price"] = (
                updated_short_quotes["bid_price"] + updated_short_quotes["ask_price"]
            ) / 2

            updated_long_quotes["mid_price"] = (
                updated_long_quotes["bid_price"] + updated_long_quotes["ask_price"]
            ) / 2

            updated_spread_value = (
                updated_short_quotes["mid_price"].iloc[0]
                - updated_long_quotes["mid_price"].iloc[0]
            )

            gross_pnl = cost - updated_spread_value
            gross_pnl_percent = round((gross_pnl / cost) * 100, 2)

            print(
                f"\nLive PnL: ${round(gross_pnl*100,2)} | {gross_pnl_percent}% | {updated_short_quotes.index[0].strftime('%H:%M')}"
            )
            print(
                f"Side: {side} | Short Strike: {short_strike} | Long Strike: {long_strike} | % Away from strike: {underlying_data['distance_from_short_strike'].iloc[-1]}%"
            )
        else:
            logging.info(f"Market is closed on {today}. Skipping trading strategy execution.")
    except Exception as e:
        logging.error(f"Error executing trading strategy: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    trading_dates = CALENDAR.schedule(
        start_date="2023-01-01", end_date=(datetime.today() + timedelta(days=1))
    ).index.strftime("%Y-%m-%d").values
    date = trading_dates[-1]

    vix_data = get_historical_data("I:VIX1D", "2023-05-01", date)
    if vix_data is not None:
        vix_data = calculate_regime(vix_data)
        vol_regime = vix_data["regime"].iloc[-1]
    else:
        logging.error("Failed to retrieve VIX data. Exiting.")
        exit(1)

    spy_data = get_historical_data("SPY", "2020-01-01", date)
    if spy_data is not None:
        spy_data = calculate_regime(spy_data)
        trend_regime = spy_data["regime"].iloc[-1]
    else:
        logging.error("Failed to retrieve SPY data. Exiting.")
        exit(1)

    real_trading_dates = CALENDAR.schedule(
        start_date=(datetime.today() - timedelta(days=10)), end_date=datetime.today()
    ).index.strftime("%Y-%m-%d").values
    today = real_trading_dates[-1]

    while True:
        nyse = get_calendar('NYSE')
        # including pre and post-market
        early = nyse.schedule(start_date='2012-07-01', end_date='2024-09-27')
        ny_timezone = nyse.tz.zone
        now = datetime.now(pytz.timezone(ny_timezone))

        # Check if the current time is within trading hours
        # Get the current time in utc-4
        current_time = datetime.now(timezone(timedelta(hours=-4))).strftime('%H:%M')

        # Format the time as 'YYYY-MM-DD HH:MM'
        if not nyse.open_at_time(early, pd.Timestamp(current_time, tz='America/New_York')):
            logging.info("Outside trading hours. Exiting loop.")
            print("Outside trading hours")
            break

        execute_trading_strategy(today, date, trend_regime)
        time.sleep(60)  # Sleep for 60 seconds before the next iteration