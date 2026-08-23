import sys
import argparse
from db import init_db, get_stock_report, get_all_reports

def print_single_report(report: dict):
    """Formats and prints a detailed single stock fundamental analysis."""
    score = report.get("ai_score", 0)
    score_badge = "🟢 HIGH CONVICTION" if score >= 8 else ("🟡 MODERATE" if score >= 6 else "🔴 CAUTION")

    print("\n" + "=" * 65)
    print(f"📈 [INSTANT DAY-SHIFT VALUATION] {report['company_name']} ({report['ticker']})")
    print("=" * 65)
    print(f"💰 Current Price:  ₹{report['current_price']} | Sector: {report['sector']}")
    print(f"📊 Key Metrics:    P/E: {report['pe_ratio']} | D/E: {report['debt_to_equity']} | ROE: {report['roe']}%")
    print(f"🏆 AI Health Score: {score}/10  ({score_badge})")
    print("-" * 65)
    print(f"🔍 AI Verdict:\n   {report['ai_verdict']}\n")
    print(f"🏰 Moat Analysis:\n   {report['moat_analysis']}\n")
    print(f"⚠️ Top Risks to Watch:\n   {report['top_risks']}")
    print("-" * 65)
    print(f"⏰ Analysis Timestamp: {report['updated_at']} (Pre-computed in 0.01s!)")
    print("=" * 65 + "\n")

def print_summary_board():
    """Prints a fast scoreboard of all pre-computed stocks."""
    reports = get_all_reports()
    if not reports:
        print("\n❌ No stock reports found in stocks.db. Run 'python night_shift.py' first!\n")
        return

    print("\n" + "=" * 80)
    print("🌅 MORNING INDIAN STOCK INTELLIGENCE DASHBOARD (Pre-Computed via Pi 5)")
    print("=" * 80)
    print(f"{'TICKER':<15} {'PRICE (₹)':<12} {'P/E':<8} {'SCORE':<8} {'VERDICT':<35}")
    print("-" * 80)

    for r in reports:
        print(f"{r['ticker']:<15} ₹{r['current_price']:<11} {r['pe_ratio']:<8} {r['ai_score']}/10   {r['ai_verdict'][:35]}...")

    print("=" * 80)
    print("💡 Type 'python day_shift.py <TICKER>' for full deep-dive analysis.\n")

def main():
    init_db()
    parser = argparse.ArgumentParser(description="Instant Daytime Stock Fundamental Query Tool")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol (e.g. TCS, RELIANCE, TATAMOTORS)")
    args = parser.parse_args()

    if args.ticker:
        report = get_stock_report(args.ticker)
        if report:
            print_single_report(report)
        else:
            print(f"\n❌ No pre-computed report found for '{args.ticker}'.")
            print(f"💡 Run 'python night_shift.py' to generate analysis for this stock!\n")
    else:
        print_summary_board()

if __name__ == "__main__":
    main()
