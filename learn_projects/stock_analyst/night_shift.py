import json
import logging
import httpx
import os
from db import init_db, save_stock_report
from fetcher import fetch_indian_stock_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("night_shift")

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions")

# Default Indian Stock Watchlist
WATCHLIST = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "TATAMOTORS.NS",
    "INFY.NS",
    "ITC.NS"
]

SYSTEM_PROMPT = """You are a senior Indian stock market equity research analyst.
Analyze the provided fundamental financial metrics for the company.

You MUST respond strictly in the following JSON format:
{
  "ai_score": 8,
  "ai_verdict": "One sentence summary of investment valuation (Bullish / Neutral / Bearish).",
  "moat_analysis": "One concise sentence analyzing competitive advantages and pricing power.",
  "top_risks": "One concise sentence listing top 2 debt or market risks."
}
"""

def analyze_with_ai(financials: dict) -> dict:
    """Prompts the local LLM on Pi 5 to perform fundamental analysis."""
    user_prompt = (
        f"Company: {financials['company_name']} ({financials['ticker']})\n"
        f"Sector: {financials['sector']}\n"
        f"Current Price: ₹{financials['current_price']}\n"
        f"Market Cap: ₹{financials['market_cap_cr']} Crores\n"
        f"P/E Ratio: {financials['pe_ratio']} (Forward P/E: {financials['forward_pe']})\n"
        f"Debt-to-Equity: {financials['debt_to_equity']}\n"
        f"ROE: {financials['roe']}%\n"
        f"Profit Margin: {financials['profit_margin']}%\n"
        f"Free Cash Flow: ₹{financials['free_cashflow_cr']} Crores\n"
    )

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 200
    }

    try:
        with httpx.Client(timeout=35.0) as client:
            res = client.post(LLAMA_SERVER_URL, json=payload)
            res.raise_for_status()
            content = res.json()["choices"][0]["message"]["content"].strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)

    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        return {
            "ai_score": 5,
            "ai_verdict": "Fundamental data fetched, but local AI evaluation timed out.",
            "moat_analysis": "Standard industry player.",
            "top_risks": "Macro market risk."
        }

def run_night_shift(tickers: list = None):
    """Executes the overnight batch processing pipeline."""
    init_db()
    symbols = tickers or WATCHLIST

    print("\n" + "=" * 60)
    print("🌙 THE NIGHT SHIFT: Running Indian Stock Fundamental Batch Analysis...")
    print(f"📊 Analyzing {len(symbols)} companies on Pi 5...")
    print("=" * 60)

    for sym in symbols:
        data = fetch_indian_stock_data(sym)
        if not data:
            continue

        print(f"\n🧠 Analyzing {data['company_name']} ({data['ticker']}) with Local LLM...")
        ai_res = analyze_with_ai(data)

        # Merge financials with AI report
        combined = {**data, **ai_res}
        save_stock_report(combined)
        print(f"   ✅ Saved to stocks.db | Score: {ai_res.get('ai_score')}/10 | Verdict: {ai_res.get('ai_verdict')}")

    print("\n" + "=" * 60)
    print("🎉 NIGHT SHIFT COMPLETE! All reports stored in SQLite (stocks.db).")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_night_shift()
