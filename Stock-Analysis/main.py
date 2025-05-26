import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from data_fetcher import YahooFinanceDataFetcher
from yfinance_news_fetcher import YahooFinanceNewsFetcher
from news_summarizer_sentiment_analyzer import NewsProcessor
from google_sheet_api import GoogleSheetsUploader

# ========== Configuration ========== #
BASE_DIR = os.getcwd()
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
CREDENTIAL_GOOGLE_SHEETS_PATH = os.path.join(BASE_DIR, "credential_google_sheets.json")
CREDENTIAL_GEMINI_PATH = os.path.join(BASE_DIR, "credential_gemini.json")

# ========== Data Fetching ========== #
fetcher = YahooFinanceDataFetcher(CONFIG_PATH)
full_df = fetcher.get_data()
full_df["Datetime"] = pd.to_datetime(full_df["Datetime"], errors="coerce").dt.date
symbol_list = full_df["Symbol"].unique()

news_fetcher = YahooFinanceNewsFetcher(CONFIG_PATH)
df_news = news_fetcher.fetch_all_news()

# ========== News Summarization & Sentiment Analysis ========== #
try:
    processor = NewsProcessor(config_path=CONFIG_PATH, gemini_credentials=CREDENTIAL_GEMINI_PATH)
    if not df_news.empty:
        df_sentiment = processor.summarize_news(df_news)
        df_sentiment = processor.analyze_sentiment(df_sentiment)
        df_sentiment["Last Updated"] = pd.to_datetime(df_sentiment["Last Updated"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        print("✅ News analysis completed.")
    else:
        print("❌ No news data available.")
except Exception as e:
    print(f"❌ Error during news processing: {e}")
    df_sentiment = pd.DataFrame()

# ========== Forecasting ========== #
forecast_dfs = []

for symbol in symbol_list:
    symbol_df = full_df[full_df["Symbol"] == symbol][["Datetime", "Close"]].dropna().copy()
    symbol_df["Datetime"] = pd.to_datetime(symbol_df["Datetime"])
    symbol_df = symbol_df[symbol_df["Datetime"] >= symbol_df["Datetime"].max() - pd.DateOffset(years=5)]

    symbol_df["day_number"] = (symbol_df["Datetime"] - symbol_df["Datetime"].min()).dt.days
    X = symbol_df[["day_number"]]
    y = symbol_df["Close"]

    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    y_pred = model.predict(X_poly)
    std_bias = 2 * np.std(y - y_pred)

    future_days = np.arange(X["day_number"].max() + 1, X["day_number"].max() + 366).reshape(-1, 1)
    future_days_poly = poly.transform(future_days)
    future_forecast = model.predict(future_days_poly)

    upper_bound = future_forecast + std_bias
    lower_bound = future_forecast - std_bias
    forecast_dates = pd.date_range(symbol_df["Datetime"].max() + pd.Timedelta(days=1), periods=365)

    forecast_df = pd.DataFrame({
        "Symbol": symbol,
        "Datetime": forecast_dates,
        "Forecast_Price": future_forecast,
        "Upper_Bound": upper_bound,
        "Lower_Bound": lower_bound
    })

    forecast_dfs.append(forecast_df)

all_forecast_df = pd.concat(forecast_dfs, ignore_index=True)

# ========== Upload to Google Sheets ========== #
try:
    print("📤 Uploading data to Google Sheets...")
    uploader = GoogleSheetsUploader(CREDENTIAL_GOOGLE_SHEETS_PATH, "Stock Analysis")
    uploader.upload_dataframe(full_df, "Price", replace=True)
    uploader.upload_dataframe(all_forecast_df, "Forecast", replace=True)
    uploader.upload_dataframe(df_sentiment, "Sentiment Analysis", replace=True)
    print("✅ Upload successful!")
except Exception as e:
    print(f"❌ Upload failed: {e}")
