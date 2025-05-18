import os
import yaml
import yfinance as yf
import pandas as pd


class YahooFinanceDataFetcher:
    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.symbol_map = self.config.get("symbols_yfinance", {})
        self.coeff_map = self.config.get("symbol_coefficients", {})
        self.daily_period = self.config.get("daily_period", "10y")
        self.daily_interval = self.config.get("daily_interval", "1d")
        self.std_multiplier = float(self.config.get("std_multiplier", 1.97))

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        try:
            return yf.download(ticker, period=self.daily_period, interval=self.daily_interval, progress=False)
        except Exception as e:
            print(f"❌ Error fetching '{ticker}': {e}")
            return pd.DataFrame()

    def clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if data.empty:
            return data

        data = data.reset_index()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data.rename(columns={"Date": "Datetime", "datetime": "Datetime"}, inplace=True)
        data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
                data[col] = data[col].apply(lambda x: 0 if pd.notnull(x) and x < 0 else x)

        data["Symbol"] = symbol
        return data[["Symbol", "Datetime"] + [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in data.columns]]

    def process_all_symbols(self) -> dict:
        symbol_data = {}
        for symbol, ticker in self.symbol_map.items():
            print(f"📈 Fetching {symbol} ({ticker})...")
            raw_data = self.fetch_data(ticker)
            if not raw_data.empty:
                symbol_data[symbol] = self.clean_data(raw_data, symbol)
        return symbol_data

    def get_data(self) -> pd.DataFrame:
        """
        Fetch, clean, and combine all symbol data into a single DataFrame.
        """
        symbol_data = self.process_all_symbols()
        df = pd.concat(symbol_data.values(), ignore_index=True)
        return df
