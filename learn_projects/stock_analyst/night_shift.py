import os
import glob
import csv
import json
import logging
import argparse
from datetime import datetime, timedelta
from db import init_db, save_stock_report, get_stock_report
from fetcher import fetch_indian_stock_data
from ai_analyst import query_live_edge_ai

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("night_shift")

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
        return ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]

    unique_tickers_dict = {}

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
        if not force and is_report_fresh(sym):
            skipped_count += 1
            print(f"[{i}/{len(tickers)}] ⏩ [SKIP] {sym:<15} (Fresh analysis exists in stocks.db)")
            continue

        print(f"\n[{i}/{len(tickers)}] 📊 Fetching & Analyzing {sym}...")
        data = fetch_indian_stock_data(sym)
        if not data or not data.get("current_price"):
            print(f"   ⚠️ Could not fetch financial metrics for {sym}. Skipping.")
            continue

        ai_res = query_live_edge_ai(data)
        combined = {**data, **ai_res}
        save_stock_report(combined)
        analyzed_count += 1
        score_display = f"{ai_res.get('ai_score')}/10" if ai_res.get('ai_score') is not None else "N/A"
        print(f"   ✅ Saved {data['company_name']} | Score: {score_display}")
        print(f"   💡 Verdict: {str(ai_res.get('ai_verdict'))[:100]}...")

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
