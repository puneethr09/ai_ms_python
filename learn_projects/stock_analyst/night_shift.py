import os
import csv
import json
import logging
import argparse
import httpx
from db import init_db, save_stock_report
from fetcher import fetch_indian_stock_data
from ai_analyst import calculate_grounded_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("night_shift")

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions")

INPUT_DIR_PATHS = [
    "/home/puneeth/repo/stock_fundamental/input",
    "../../stock_fundamental/input",
    "input"
]

UNIVERSE_FILES = {
    "nifty50": "Indian_stocks_nifty_50.csv",
    "nifty200": "Indian_stocks_nifty_200.csv",
    "nifty500": "Indian_stocks_nifty_500.csv",
    "midcap": "Indian_stocks_nifty_midcap_100.csv",
    "smallcap": "Indian_stocks_nifty_smallcap_250.csv"
}

def load_tickers_from_csv(universe_key: str = "nifty50") -> list:
    """Reads ticker symbols from the user's input Excel/CSV directory."""
    filename = UNIVERSE_FILES.get(universe_key, "Indian_stocks_nifty_50.csv")
    csv_path = None

    for base_dir in INPUT_DIR_PATHS:
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            csv_path = candidate
            break

    if not csv_path:
        logger.warning(f"Could not find {filename} in {INPUT_DIR_PATHS}. Falling back to default list.")
        return ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "TATAMOTORS.NS"]

    tickers = []
    try:
        with open(csv_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = row.get("Ticker") or row.get("Symbol") or row.get("ticker")
                if sym:
                    sym = sym.strip()
                    if not sym.endswith(".NS") and not sym.endswith(".BO"):
                        sym = f"{sym}.NS"
                    tickers.append(sym)
        logger.info(f"Loaded {len(tickers)} tickers from {csv_path}")
        return tickers
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
        return ["TCS.NS", "RELIANCE.NS", "INFY.NS"]

def analyze_company_with_ai(financials: dict) -> dict:
    """Generates CFA-grade grounded analysis using local Llama model."""
    pe = financials.get("pe_ratio", 0.0)
    roe = financials.get("roe", 0.0)
    de = financials.get("debt_to_equity", 0.0)
    margin = financials.get("operating_margin") or financials.get("profit_margin", 0.0)
    fcf = financials.get("free_cashflow_cr", 0.0)
    price = financials.get("current_price", 0.0)
    sector = financials.get("sector", "Indian Markets")
    company_name = financials.get("company_name", financials["ticker"])

    score = calculate_grounded_score(pe, roe, de, margin)

    user_prompt = (
        f"Analyze {company_name} ({financials['ticker']}) in the {sector} sector:\n"
        f"• Stock Price: ₹{price} | P/E: {pe}x | ROE: {roe}% | Debt/Equity: {de}\n"
        f"• Operating Margin: {margin}% | Free Cash Flow: ₹{fcf} Cr\n\n"
        f"Grounding requirement: Reference these exact numbers. Explain valuation thesis, economic moat, and top 2 risks."
    )

    system_prompt = (
        "You are an expert equity research AI on Raspberry Pi 5. Return STRICT JSON only:\n"
        "{\n"
        f"  \"ai_score\": {score},\n"
        "  \"ai_verdict\": \"2-3 factual sentences explaining valuation appeal based on ROE and P/E.\",\n"
        "  \"moat_analysis\": \"2 sentences on competitive pricing power and moat durability.\",\n"
        "  \"top_risks\": \"2 specific risks regarding debt, margins, or market cyclicality.\"\n"
        "}"
    )

    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.15,
        "max_tokens": 250
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            res = client.post(LLAMA_SERVER_URL, json=payload)
            res.raise_for_status()
            content = res.json()["choices"][0]["message"]["content"].strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            parsed = json.loads(content)
            parsed["ai_score"] = score
            return parsed
    except Exception as e:
        logger.debug(f"AI call failed: {e}")
        return {
            "ai_score": score,
            "ai_verdict": f"{company_name} carries an ROE of {roe}% with {pe}x P/E and {de} debt-to-equity ratio.",
            "moat_analysis": f"Operating margin of {margin}% reflects industry competitive positioning in {sector}.",
            "top_risks": "Margin compression and macro interest rate headwinds."
        }

def run_night_shift(universe_key: str = "nifty50", limit: int = None):
    """Executes the overnight batch processing pipeline."""
    init_db()
    tickers = load_tickers_from_csv(universe_key)
    if limit:
        tickers = tickers[:limit]

    print("\n" + "=" * 65)
    print(f"🌙 THE NIGHT SHIFT: Running Indian Stock Fundamental Batch Analysis")
    print(f"📁 Universe: {universe_key.upper()} ({len(tickers)} companies)")
    print("=" * 65)

    for i, sym in enumerate(tickers, start=1):
        print(f"\n[{i}/{len(tickers)}] 📊 Fetching & Analyzing {sym}...")
        data = fetch_indian_stock_data(sym)
        if not data:
            continue

        ai_res = analyze_company_with_ai(data)
        combined = {**data, **ai_res}
        save_stock_report(combined)
        print(f"   ✅ Saved {data['company_name']} | Score: {ai_res.get('ai_score')}/10")
        print(f"   💡 Verdict: {ai_res.get('ai_verdict')[:80]}...")

    print("\n" + "=" * 65)
    print(f"🎉 NIGHT SHIFT COMPLETE! {len(tickers)} company reports saved in SQLite (stocks.db).")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Night Shift Stock Batch Analyst")
    parser.add_argument("--universe", default="nifty50", choices=["nifty50", "nifty200", "nifty500", "midcap", "smallcap"], help="Universe CSV to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers to analyze")
    args = parser.parse_args()

    run_night_shift(args.universe, args.limit)
