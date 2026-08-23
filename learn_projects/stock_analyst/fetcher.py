import yfinance as yf
import numpy as np
import logging

logger = logging.getLogger("stock_analyst.fetcher")

def fetch_indian_stock_data(symbol: str) -> dict:
    """
    Fetches real fundamental metrics for an Indian stock (NSE/BSE).
    Extracts metrics directly from Yahoo Finance info and financial statements.
    Missing metrics are set to None (no fake/placeholder values).
    """
    ticker_sym = symbol.upper().strip()
    if not ticker_sym.endswith(".NS") and not ticker_sym.endswith(".BO"):
        ticker_sym = f"{ticker_sym}.NS"

    logger.info(f"Fetching financial fundamentals for {ticker_sym}...")

    try:
        stock = yf.Ticker(ticker_sym)
        info = stock.info or {}

        # 1. Price
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is not None:
            price = round(float(price), 2)

        # 2. P/E Ratio (Trailing preferred, Forward as fallback)
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe is not None and not np.isnan(pe) and pe > 0:
            pe = round(float(pe), 2)
        else:
            pe = None

        # 3. Return on Equity (ROE)
        roe = info.get("returnOnEquity")
        if roe is not None and not np.isnan(roe):
            roe = round(float(roe) * 100, 2)
        else:
            # Calculate mathematically from Net Income and Stockholders Equity
            try:
                if stock.financials is not None and not stock.financials.empty and stock.balance_sheet is not None and not stock.balance_sheet.empty:
                    net_inc = stock.financials.loc["Net Income"].dropna().iloc[0]
                    eq = stock.balance_sheet.loc["Stockholders Equity"].dropna().iloc[0]
                    if eq and eq != 0:
                        roe = round(float(net_inc / eq) * 100, 2)
            except Exception:
                roe = None

        # 4. Debt to Equity
        de = info.get("debtToEquity")
        if de is not None and not np.isnan(de):
            de = round(float(de), 2)
        else:
            de = None

        # 5. Operating Margin
        op_margin = info.get("operatingMargins")
        if op_margin is None or np.isnan(op_margin):
            op_margin = info.get("profitMargins")
        if op_margin is not None and not np.isnan(op_margin):
            op_margin = round(float(op_margin) * 100, 2)
        else:
            op_margin = None

        # 6. Free Cash Flow (converted to INR Crores)
        fcf = info.get("freeCashflow")
        if fcf is not None and not np.isnan(fcf):
            fcf_cr = round(float(fcf) / 1e7, 2)
        else:
            try:
                if stock.cashflow is not None and not stock.cashflow.empty:
                    if "Free Cash Flow" in stock.cashflow.index:
                        val = stock.cashflow.loc["Free Cash Flow"].dropna().iloc[0]
                        fcf_cr = round(float(val) / 1e7, 2)
                    elif "Operating Cash Flow" in stock.cashflow.index and "Capital Expenditure" in stock.cashflow.index:
                        ocf = stock.cashflow.loc["Operating Cash Flow"].dropna().iloc[0]
                        capex = stock.cashflow.loc["Capital Expenditure"].dropna().iloc[0]
                        fcf_cr = round(float(ocf + capex) / 1e7, 2)
                    else:
                        fcf_cr = None
                else:
                    fcf_cr = None
            except Exception:
                fcf_cr = None

        # 7. Market Cap in INR Crores
        mcap = info.get("marketCap")
        mcap_cr = round(float(mcap) / 1e7, 2) if mcap else None

        data = {
            "ticker": ticker_sym,
            "company_name": info.get("shortName") or info.get("longName") or ticker_sym,
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or "Unknown",
            "current_price": price,
            "currency": info.get("currency", "INR"),
            "market_cap_cr": mcap_cr,
            "pe_ratio": pe,
            "debt_to_equity": de,
            "roe": roe,
            "operating_margin": op_margin,
            "free_cashflow_cr": fcf_cr,
            "business_summary": (info.get("longBusinessSummary", "") or "")[:500]
        }
        return data

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker_sym}: {e}")
        return None
