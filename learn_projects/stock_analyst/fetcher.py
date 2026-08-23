import yfinance as yf
import logging

logger = logging.getLogger("stock_analyst.fetcher")

def fetch_indian_stock_data(symbol: str) -> dict:
    """
    Fetches comprehensive fundamental metrics for an Indian stock (NSE/BSE).
    Automatically appends .NS (National Stock Exchange) if not provided.
    """
    ticker_sym = symbol.upper().strip()
    if not ticker_sym.endswith(".NS") and not ticker_sym.endswith(".BO"):
        ticker_sym = f"{ticker_sym}.NS"

    logger.info(f"Fetching deep financial fundamentals for {ticker_sym}...")

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
            "market_cap_cr": round((info.get("marketCap", 0) / 1e7), 2),
            "pe_ratio": round(info.get("trailingPE", 0.0) or 0.0, 2),
            "forward_pe": round(info.get("forwardPE", 0.0) or 0.0, 2),
            "price_to_book": round(info.get("priceToBook", 0.0) or 0.0, 2),
            "debt_to_equity": round(info.get("debtToEquity", 0.0) or 0.0, 2),
            "roe": round((info.get("returnOnEquity", 0.0) or 0.0) * 100, 2),
            "roa": round((info.get("returnOnAssets", 0.0) or 0.0) * 100, 2),
            "profit_margin": round((info.get("profitMargins", 0.0) or 0.0) * 100, 2),
            "operating_margin": round((info.get("operatingMargins", 0.0) or 0.0) * 100, 2),
            "revenue_growth": round((info.get("revenueGrowth", 0.0) or 0.0) * 100, 2),
            "earnings_growth": round((info.get("earningsGrowth", 0.0) or 0.0) * 100, 2),
            "dividend_yield": round((info.get("dividendYield", 0.0) or 0.0) * 100, 2),
            "free_cashflow_cr": round((info.get("freeCashflow", 0) or 0) / 1e7, 2),
            "beta": round(info.get("beta", 1.0) or 1.0, 2),
            "52w_high": info.get("fiftyTwoWeekHigh", 0.0),
            "52w_low": info.get("fiftyTwoWeekLow", 0.0),
            "business_summary": (info.get("longBusinessSummary", "") or "")[:500]
        }
        return data

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker_sym}: {e}")
        return None
