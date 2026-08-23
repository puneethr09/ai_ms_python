import yfinance as yf
import logging

logger = logging.getLogger("stock_analyst.fetcher")

def fetch_indian_stock_data(symbol: str) -> dict:
    """
    Fetches fundamental financial metrics for an Indian stock (NSE/BSE).
    Automatically appends .NS (National Stock Exchange) if not provided.
    """
    ticker_sym = symbol.upper()
    if not ticker_sym.endswith(".NS") and not ticker_sym.endswith(".BO"):
        ticker_sym = f"{ticker_sym}.NS"

    logger.info(f"Fetching financial fundamentals for {ticker_sym} via Yahoo Finance...")

    try:
        stock = yf.Ticker(ticker_sym)
        info = stock.info

        data = {
            "ticker": ticker_sym,
            "company_name": info.get("shortName") or info.get("longName") or ticker_sym,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or 0.0,
            "currency": info.get("currency", "INR"),
            "market_cap_cr": round((info.get("marketCap", 0) / 1e7), 2),  # Converted to INR Crores
            "pe_ratio": round(info.get("trailingPE", 0.0) or 0.0, 2),
            "forward_pe": round(info.get("forwardPE", 0.0) or 0.0, 2),
            "debt_to_equity": round(info.get("debtToEquity", 0.0) or 0.0, 2),
            "roe": round((info.get("returnOnEquity", 0.0) or 0.0) * 100, 2),
            "profit_margin": round((info.get("profitMargins", 0.0) or 0.0) * 100, 2),
            "free_cashflow_cr": round((info.get("freeCashflow", 0) or 0) / 1e7, 2),
            "summary": info.get("longBusinessSummary", "")[:600]
        }
        return data

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker_sym}: {e}")
        return None
