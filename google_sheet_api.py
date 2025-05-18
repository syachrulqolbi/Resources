import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe, get_as_dataframe
from gspread.exceptions import SpreadsheetNotFound


class GoogleSheetsUploader:
    def __init__(self, credentials_file, spreadsheet_name):
        self.credentials_file = credentials_file
        self.spreadsheet_name = spreadsheet_name
        self.scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        self.client = self.authenticate()

    def authenticate(self):
        creds = Credentials.from_service_account_file(self.credentials_file).with_scopes(self.scopes)
        return gspread.authorize(creds)

    def get_sheet(self, name_sheet):
        try:
            spreadsheet = self.client.open(self.spreadsheet_name)
            if name_sheet not in [ws.title for ws in spreadsheet.worksheets()]:
                return spreadsheet.add_worksheet(title=name_sheet, rows="100", cols="20")  # Create sheet if not exists
            else:
                return spreadsheet.worksheet(name_sheet)
        except SpreadsheetNotFound:
            raise FileNotFoundError(f"Spreadsheet '{self.spreadsheet_name}' not found. Please check the name or ID.")
        except gspread.exceptions.WorksheetNotFound:
            raise FileNotFoundError(f"Worksheet '{name_sheet}' not found in the spreadsheet.")

    def clear_sheet(self, sheet):
        """Clears all existing data from the sheet."""
        sheet.clear()
        print(f"✅ Cleared all data from sheet: {sheet.title}")

    def get_sheet_as_dataframe(self, name_sheet: str):
        """Retrieve Google Sheets data as a Pandas DataFrame."""
        try:
            sheet = self.get_sheet(name_sheet)
            df = get_as_dataframe(sheet, evaluate_formulas=True)
            print(f"✅ Successfully retrieved data from '{name_sheet}' as a DataFrame.")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve data from Google Sheets: {e}")

    def upload_dataframe(self, df: pd.DataFrame, name_sheet: str, replace: bool = True):
        """Uploads a DataFrame directly to Google Sheets.
        
        If replace=False, it updates existing rows based on 'Symbol' and retains old rows.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Provided data is not a pandas DataFrame.")

        sheet = self.get_sheet(name_sheet)

        if not replace:
            try:
                df_old = self.get_sheet_as_dataframe(name_sheet)

                if 'Symbol' not in df.columns or 'Symbol' not in df_old.columns:
                    raise ValueError("Both dataframes must contain 'Symbol' column for update mode.")

                df_old.set_index("Symbol", inplace=True)
                df.set_index("Symbol", inplace=True)

                # Update old with new, and append any new symbols
                df_updated = df_old.copy()
                df_updated.update(df)
                df_combined = pd.concat([df_updated, df[~df.index.isin(df_old.index)]])
                df = df_combined.reset_index()

            except Exception as e:
                raise RuntimeError(f"Failed to update existing data: {e}")

        self.clear_sheet(sheet)
        set_with_dataframe(sheet, df)

        print(f"✅ DataFrame successfully uploaded to Google Sheets: {name_sheet}!")

