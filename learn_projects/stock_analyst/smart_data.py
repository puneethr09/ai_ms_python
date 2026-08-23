import yfinance as yf
import pandas as pd
import numpy as np
import time

class SmartDataEngine:
    """
    Robust data extraction and quantitative computation layer for Indian Stocks.
    Provides verified multi-year financial statements, automated currency alignment,
    Piotroski / DuPont support, and safe extraction for non-existent items.
    """
    
    def __init__(self, ticker):
        self.ticker_symbol = ticker
        self.ticker = yf.Ticker(ticker)
        
        # Retry logic for fetching data (handles rate limits globally)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.info = self.ticker.info or {}
                self.financials = self.ticker.financials
                self.balance_sheet = self.ticker.balance_sheet
                self.cashflow = self.ticker.cashflow
                break
            except Exception as e:
                error_str = str(e)
                if "Rate" in error_str or "Too Many" in error_str:
                    wait_time = 15 + (attempt * 5)
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        self.info = {}
                        self.financials = pd.DataFrame()
                        self.balance_sheet = pd.DataFrame()
                        self.cashflow = pd.DataFrame()
                else:
                    self.info = {}
                    self.financials = pd.DataFrame()
                    self.balance_sheet = pd.DataFrame()
                    self.cashflow = pd.DataFrame()
                    break
        
        self.has_data = not (self.financials.empty or self.balance_sheet.empty or self.cashflow.empty)
        
        # Currency Normalization
        self.fx_rate = 1.0
        price_curr = self.info.get("currency", "INR")
        fin_curr = self.info.get("financialCurrency", "INR")
        
        if price_curr == "INR" and fin_curr == "USD":
            try:
                if not self.financials.empty and "Total Revenue" in self.financials.index:
                    rev = self.financials.loc["Total Revenue"].iloc[0]
                    if rev > 500_000_000_000:  # > 500 Billion is already INR
                        self.fx_rate = 1.0
                    else:
                        self.fx_rate = 84.0
                else:
                    self.fx_rate = 84.0
            except Exception:
                self.fx_rate = 84.0

    def get_financials_safe(self, df, key, year_idx=0, default=0.0):
        """Safely retrieve a value from a DataFrame row (key) and column (year_idx)."""
        if not self.has_data or df is None or df.empty:
            return default
        try:
            if key in df.index and len(df.columns) > year_idx:
                val = df.loc[key].iloc[year_idx]
                if pd.isna(val) or val is None:
                    return default
                return float(val) * self.fx_rate
        except Exception:
            return default
        return default

    def has_row(self, df, key):
        """Checks if a row key exists in DataFrame."""
        return self.has_data and df is not None and not df.empty and key in df.index

    def get_available_years(self, df):
        """Returns the number of available annual columns."""
        if not self.has_data or df is None or df.empty:
            return 0
        return len(df.columns)

    def calculate_invested_capital(self, year_idx=0):
        """Invested Capital = Total Equity + Total Debt - Cash & Equivalents."""
        total_equity = self.get_financials_safe(self.balance_sheet, "Stockholders Equity", year_idx)
        long_term_debt = self.get_financials_safe(self.balance_sheet, "Long Term Debt", year_idx)
        current_debt = self.get_financials_safe(self.balance_sheet, "Current Debt", year_idx)
        total_debt_bs = self.get_financials_safe(self.balance_sheet, "Total Debt", year_idx)
        total_debt = total_debt_bs if total_debt_bs > 0 else (long_term_debt + current_debt)
        cash = self.get_financials_safe(self.balance_sheet, "Cash And Cash Equivalents", year_idx)
        return total_equity + total_debt - cash

    def calculate_nopat(self, year_idx=0):
        """NOPAT = Operating Income * (1 - Tax Rate)."""
        operating_income = self.get_financials_safe(self.financials, "Operating Income", year_idx)
        tax_provision = self.get_financials_safe(self.financials, "Tax Provision", year_idx)
        pretax_income = self.get_financials_safe(self.financials, "Pretax Income", year_idx)
        
        if pretax_income <= 0:
            tax_rate = 0.25
        else:
            tax_rate = tax_provision / pretax_income
            tax_rate = max(0.0, min(tax_rate, 0.40))
            
        return operating_income * (1 - tax_rate)

    def calculate_roic(self, year_idx=0):
        """ROIC = NOPAT / Invested Capital."""
        ic = self.calculate_invested_capital(year_idx)
        nopat = self.calculate_nopat(year_idx)
        if ic <= 0:
            return 0.0
        return (nopat / ic) * 100

    def calculate_fcf(self, year_idx=0):
        """Free Cash Flow = Operating Cash Flow - Capital Expenditure."""
        ocf = self.get_financials_safe(self.cashflow, "Operating Cash Flow", year_idx)
        capex = self.get_financials_safe(self.cashflow, "Capital Expenditure", year_idx)
        if capex == 0.0:
            capex = self.get_financials_safe(self.cashflow, "Capital Expenditures", year_idx)
        
        # Ensure CapEx is treated as cash outflow (negative)
        if capex > 0:
            capex = -capex
            
        return ocf + capex

    def calculate_multi_year_cagr(self, df, key, max_years=3):
        """Calculates annualized CAGR across all available historical years up to max_years."""
        avail = self.get_available_years(df)
        if avail < 2:
            return None
        target_idx = min(avail - 1, max_years)
        val_curr = self.get_financials_safe(df, key, 0)
        val_prev = self.get_financials_safe(df, key, target_idx)
        
        if val_curr > 0 and val_prev > 0 and target_idx > 0:
            cagr = (val_curr / val_prev) ** (1.0 / target_idx) - 1.0
            return cagr
        return None

    def get_52w_position(self):
        """Computes 52-week position and distance from high/low."""
        current = self.info.get("currentPrice", 0) or 0
        low = self.info.get("fiftyTwoWeekLow", 0) or 0
        high = self.info.get("fiftyTwoWeekHigh", 0) or 0
        
        if high > low and low > 0 and current > 0:
            range_pct = ((current - low) / (high - low)) * 100.0
            discount_from_high = ((high - current) / high) * 100.0
            premium_from_low = ((current - low) / low) * 100.0
            return {
                "current_price": current,
                "fifty_two_week_low": low,
                "fifty_two_week_high": high,
                "range_position_pct": round(range_pct, 1),
                "discount_from_high_pct": round(discount_from_high, 1),
                "premium_from_low_pct": round(premium_from_low, 1),
                "status": "Near 52W Low (Discount)" if range_pct < 25 else ("Near 52W High (Momentum)" if range_pct > 75 else "Mid-Range")
            }
        return {
            "current_price": current,
            "fifty_two_week_low": low,
            "fifty_two_week_high": high,
            "range_position_pct": 50.0,
            "discount_from_high_pct": 0.0,
            "premium_from_low_pct": 0.0,
            "status": "N/A"
        }

    def get_manual_metrics(self):
        """Returns verified Dorsey metrics calculated manually."""
        metrics = {
            "ROIC_Current": self.calculate_roic(0),
            "ROIC_1Y_Ago": self.calculate_roic(1),
            "ROIC_2Y_Ago": self.calculate_roic(2),
            "Invested_Capital": self.calculate_invested_capital(0),
            "FCF": self.calculate_fcf(0),
            "Debt_to_Equity_Manual": 0.0
        }
        equity = self.get_financials_safe(self.balance_sheet, "Stockholders Equity", 0)
        debt = self.get_financials_safe(self.balance_sheet, "Total Debt", 0)
        if equity > 0:
            metrics["Debt_to_Equity_Manual"] = debt / equity
        return metrics
