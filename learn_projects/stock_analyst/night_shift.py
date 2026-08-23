import os
import glob
import csv
import json
import logging
import argparse
import httpx
from datetime import datetime, timedelta
from db import init_db, save_stock_report, get_stock_report
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
    "all": "*.csv",
    "nifty50": "Indian_stocks_nifty_50.csv",
    "nifty200": "Indian_stocks_nifty_200.csv",
    "nifty500": "Indian_stocks_nifty_500.csv",
    "midcap": "Indian_stocks_nifty_midcap_100.csv",
    "smallcap": "Indian_stocks_nifty_smallcap_250.csv",
    "largemidcap": "Indian_stocks_nifty_large_midcap_250.csv"
}

def load_deduplicated_tickers(universe_key: str = "all") -> list:
    """
    Scans CSV files from the input directory and returns a clean,
    DEDUPLICATED list of NSE ticker symbols.
    """
    pattern = UNIVERSE_FILES.get(universe_key, "*.csv")
    found_files = []

    for base_dir in INPUT_DIR_PATHS:
        if os.path.exists(base_dir):
            if universe_key == "all":
                found_files = glob.glob(os.path.join(base_dir, "*.csv"))
            else:
                candidate = os.path.join(base_dir, pattern)
                if os.path.exists(candidate):
                    found_files = [candidate]
            if found_files:
                break

    if not found_files:
        logger.warning(f"No CSV files found matching '{pattern}' in {INPUT_DIR_PATHS}. Using fallback.")
        return ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS", "TATAMOTORS.NS"]

    unique_tickers_dict = {}  # { ticker_symbol: company_name }

    for csv_file in sorted(found_files):
        try:
            with open(csv_file, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sym = row.get("Ticker") or row.get("Symbol") or row.get("ticker")
                    comp_name = row.get("Company Name") or row.get("company_name") or sym
                    if sym:
                        sym = sym.strip()
                        if not sym.endswith(".NS") and not sym.endswith(".BO"):
                            sym = f"{sym}.NS"
                        if sym not in unique_tickers_dict:
                            unique_tickers_dict[sym] = comp_name
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")

    deduped_list = list(unique_tickers_dict.keys())
    logger.info(f"Loaded {len(deduped_list)} UNIQUE deduplicated tickers from {len(found_files)} CSV file(s).")
    return deduped_list

def is_report_fresh(ticker: str, max_age_hours: int = 24) -> bool:
    """Checks if a fresh analysis already exists in stocks.db to prevent duplicate compute."""
    report = get_stock_report(ticker)
    if not report or not report.get("updated_at"):
        return False

    try:
        updated_dt = datetime.strptime(report["updated_at"], "%Y-%m-%d %H:%M:%S")
        if (datetime.now() - updated_dt) < timedelta(hours=max_age_hours):
            return True
    except Exception:
        pass
    return False

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

def run_night_shift(universe_key: str = "all", limit: int = None, force: bool = False):
    """Executes the overnight batch processing pipeline over deduplicated stocks."""
    init_db()
    tickers = load_deduplicated_tickers(universe_key)
    if limit:
        tickers = tickers[:limit]

    print("\n" + "=" * 70)
    print(f"🌙 THE NIGHT SHIFT: Running Indian Stock Fundamental Batch Analysis")
    print(f"📁 Universe: {universe_key.upper()} ({len(tickers)} Unique Companies - Zero Duplicates)")
    print("=" * 70)

    analyzed_count = 0
    skipped_count = 0

    for i, sym in enumerate(tickers, start=1):
        # Skip if analyzed in last 24h unless force is True
        if not force and is_report_fresh(sym):
            skipped_count += 1
            print(f"[{i}/{len(tickers)}] ⏩ [SKIP] {sym:<15} (Fresh analysis exists in stocks.db)")
            continue

        print(f"\n[{i}/{len(tickers)}] 📊 Fetching & Analyzing {sym}...")
        data = fetch_indian_stock_data(sym)
        if not data:
            continue

        ai_res = analyze_company_with_ai(data)
        combined = {**data, **ai_res}
        save_stock_report(combined)
        analyzed_count += 1
        print(f"   ✅ Saved {data['company_name']} | Score: {ai_res.get('ai_score')}/10")
        print(f"   💡 Verdict: {str(ai_res.get('ai_verdict'))[:80]}...")

    print("\n" + "=" * 70)
    print(f"🎉 NIGHT SHIFT COMPLETE!")
    print(f"   • Newly Analyzed: {analyzed_count}")
    print(f"   • Already Fresh / Skipped: {skipped_count}")
    print(f"   • Total Active Reports in stocks.db: {analyzed_count + skipped_count}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Night Shift Deduplicated Stock Batch Analyst")
    parser.add_argument("--universe", default="all", choices=["all", "nifty50", "nifty200", "nifty500", "midcap", "smallcap", "largemidcap"], help="Universe to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tickers to analyze")
    parser.add_argument("--force", action="store_true", help="Force re-analysis even if report was updated in last 24h")
    args = parser.parse_args()

    run_night_shift(args.universe, args.limit, args.force)
