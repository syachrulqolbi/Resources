import google.generativeai as genai
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import pandas as pd
import json
import yaml

class NewsProcessor:
    """
    Class for processing financial news, summarizing articles using Gemini AI,
    and performing sentiment analysis with FinBERT.
    """
    def __init__(self, config_path: str, gemini_credentials: str):
        """
        Initialize the NewsProcessor class.
        Loads configuration, API keys, and necessary models.
        """
        self.config_path = config_path
        self.api_key = self._load_gemini_api_key(gemini_credentials)
        self.symbols = self._load_symbols()
        self.tokenizer, self.model = self._load_finbert()
        genai.configure(api_key=self.api_key)

    def _load_symbols(self) -> list:
        """Load trading symbols from the YAML configuration file."""
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
            return list(config.get("symbols_tradingview", {}).keys())
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"❌ Error loading symbols: {e}")
            return []

    def _load_gemini_api_key(self, gemini_credentials: str) -> str:
        """Load the Gemini API key from the credentials JSON file."""
        try:
            with open(gemini_credentials, "r") as file:
                credentials = json.load(file)
            return credentials.get("api_key", "")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Error loading Gemini API key: {e}")
            return ""

    def _load_finbert(self):
        """Load the FinBERT model and tokenizer for sentiment analysis."""
        model_name = "ProsusAI/finbert"
        try:
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForSequenceClassification.from_pretrained(model_name)
            return tokenizer, model
        except Exception as e:
            print(f"❌ Error loading FinBERT model: {e}")
            return None, None

    def summarize_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize financial news articles for each stock symbol using Gemini AI.
        Returns a DataFrame with summarized news and last updated timestamps.
        """
        if df.empty:
            print("No data to summarize.")
            return pd.DataFrame(columns=["Symbol", "Summary", "Last News", "Last Updated"])

        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime"])
        latest_dates = df.groupby("Symbol")["Datetime"].max().reset_index().rename(columns={"Datetime": "Last News"})
        results = []

        model = genai.GenerativeModel("gemini-2.0-flash")
        
        for symbol in df["Symbol"].unique():
            latest_published = latest_dates.loc[latest_dates["Symbol"] == symbol, "Last News"].values[0]
            news_summaries = "\n".join(df[df["Symbol"] == symbol]["Summary"].dropna().tolist()).strip()

            if not news_summaries:
                results.append({"Symbol": symbol, "Summary": "", "Last News": latest_published})
                continue
            
            try:
                response = model.generate_content(f"Generate a concise 1-paragraph summary of the following news articles:\n{news_summaries}")
                summary_text = response.text.strip() if response.text else "No summary available"
            except Exception as e:
                print(f"❌ Error summarizing news for {symbol}: {e}")
                summary_text = ""

            results.append({"Symbol": symbol, "Summary": summary_text, "Last News": latest_published, "Last Updated": pd.Timestamp.now().strftime("%-m/%-d/%Y %-I:%M:%S")})

        return pd.DataFrame(results)

    def analyze_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform sentiment analysis on the summarized news using FinBERT.
        Adds sentiment label and confidence score to the DataFrame.
        """
        if self.model is None or self.tokenizer is None:
            print("❌ FinBERT model not available. Sentiment analysis skipped.")
            return df
        
        df = df.copy()
        df[["Sentiment", "Confidence"]] = ""
        sentiment_pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        
        for index, row in df.iterrows():
            if pd.notna(row["Summary"]):
                try:
                    result = sentiment_pipeline(row["Summary"], truncation=True, max_length=512)[0]
                    df.at[index, "Sentiment"] = result["label"]
                    df.at[index, "Confidence"] = result["score"]
                except Exception as e:
                    print(f"❌ Error analyzing sentiment for {row['Symbol']}: {e}")
                    df.at[index, "Sentiment"] = "Error"
                    df.at[index, "Confidence"] = ""
        
        return df
