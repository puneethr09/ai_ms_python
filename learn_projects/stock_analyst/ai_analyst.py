import os
import re
import json
import logging
import sqlite3
import urllib.request

try:
    from src.fetcher import fetch_indian_stock_data
except ImportError:
    try:
        from fetcher import fetch_indian_stock_data
    except ImportError:
        fetch_indian_stock_data = lambda ticker: {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edge_ai_analyst")

DB_PATHS = [
    "/app/data/stocks.db",
    "/home/puneeth/repo/stock_fundamental/data/stocks.db",
    "/home/puneeth/repo/ai_ms_python/learn_projects/stock_analyst/stocks.db"
]

LLAMA_HOST_URLS = [
    "http://127.0.0.1:8080/v1/chat/completions",
    "http://raspberrypi:8080/v1/chat/completions",
    "http://localhost:8080/v1/chat/completions",
    "http://100.79.28.51:8080/v1/chat/completions"
]

def calculate_institutional_score(financials: dict, dorsey_data: dict = None) -> int:
    """
    Multi-Factor institutional scoring model (1-10) incorporating:
    - ROE & Capital Efficiency (0 to +3)
    - Debt/Equity & Solvency (0 to +2)
    - Operating Margin & Moat (0 to +1)
    - Valuation Multiple / Margin of Safety (-2 to +2)
    - Piotroski 9-Point F-Score (+1 or -1)
    - Sloan Earnings Quality (+1 or -1)
    - DuPont Return Engine (+1)
    """
    score = 5

    pe = financials.get("pe_ratio")
    roe = financials.get("roe")
    de = financials.get("debt_to_equity")
    margin = financials.get("operating_margin")

    # 1. Capital Efficiency (ROE)
    if roe is not None:
        if roe >= 30.0:
            score += 3
        elif roe >= 18.0:
            score += 2
        elif roe >= 10.0:
            score += 1
        elif roe < 5.0:
            score -= 2

    # 2. Balance Sheet Solvency (Debt/Equity)
    if de is not None:
        if de < 15.0:
            score += 2
        elif de < 50.0:
            score += 1
        elif de > 120.0:
            score -= 2

    # 3. Pricing Power & Moat (Operating Margin)
    if margin is not None:
        if margin >= 22.0:
            score += 1
        elif margin < 6.0:
            score -= 1

    # 4. Valuation Multiple (P/E)
    if pe is not None:
        if 0 < pe <= 20.0:
            score += 1
        elif pe > 65.0:
            score -= 2

    # 5. Quant Overlays from Dorsey/Quant data
    if dorsey_data and isinstance(dorsey_data, dict):
        p_score = dorsey_data.get("piotroski_f_score", {}).get("score")
        if p_score is not None:
            if p_score >= 7: score += 1
            elif p_score <= 4: score -= 1

        sloan_stat = dorsey_data.get("sloan_accrual", {}).get("status")
        if sloan_stat == "EXCELLENT": score += 1
        elif sloan_stat == "WARNING": score -= 1

        dup_type = dorsey_data.get("dupont_analysis", {}).get("driver_type")
        if dup_type in ["MOAT", "EFFICIENCY"]: score += 1

        mos = dorsey_data.get("valuation", {}).get("combined", {}).get("margin_of_safety")
        if mos is not None:
            if mos >= 20.0: score += 1
            elif mos <= -25.0: score -= 1

    return max(1, min(10, score))

def get_cached_ai_report(ticker: str) -> dict:
    """Retrieves pre-computed analysis from stocks.db if available."""
    for path in DB_PATHS:
        if os.path.exists(path):
            try:
                conn = sqlite3.connect(path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                clean_sym = ticker.replace(".NS", "").replace(".BO", "").strip()
                cursor.execute(
                    "SELECT * FROM stock_reports WHERE (ticker LIKE ? OR ticker = ?) AND ai_verdict IS NOT NULL AND ai_verdict != ''",
                    (f"{clean_sym}%", ticker)
                )
                row = cursor.fetchone()
                conn.close()
                if row:
                    return dict(row)
            except Exception as e:
                logger.debug(f"DB read error: {e}")
    return None

def _extract_llm_fields(raw_text: str) -> dict:
    """Extracts structured fields from LLM response (JSON and Markdown)."""
    raw_text = raw_text.strip()

    # Try JSON
    try:
        clean = raw_text
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()
        parsed = json.loads(clean)
        if isinstance(parsed, dict) and "ai_verdict" in parsed:
            return {
                "ai_verdict": str(parsed.get("ai_verdict", "")).strip(),
                "moat_analysis": str(parsed.get("moat_analysis", "")).strip(),
                "top_risks": str(parsed.get("top_risks", "")).strip()
            }
    except Exception:
        pass

    # Try Markdown Section Headers
    verdict, moat, risks = "", "", ""
    v_match = re.search(r"\*\*AI Verdict:?\*\*\s*(.+?)(?=\n\s*\*\*|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
    if v_match:
        verdict = v_match.group(1).strip()

    m_match = re.search(r"\*\*Moat Analysis:?\*\*\s*(.+?)(?=\n\s*\*\*|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
    if m_match:
        moat = m_match.group(1).strip()

    r_match = re.search(r"\*\*Top Risks:?\*\*\s*(.+)", raw_text, re.DOTALL | re.IGNORECASE)
    if r_match:
        risks = r_match.group(1).strip()

    if verdict or moat or risks:
        return {
            "ai_verdict": verdict or raw_text[:300],
            "moat_analysis": moat or "Established competitive positioning within the sector.",
            "top_risks": risks or "Sector cyclicality and valuation sensitivity."
        }

    return {
        "ai_verdict": raw_text[:350],
        "moat_analysis": "Competitive positioning grounded in domestic market presence.",
        "top_risks": "Market cyclicality, margin volatility, and macroeconomic sensitivity."
    }

def query_live_edge_ai(financials: dict, dorsey_data: dict = None) -> dict:
    """Generates CFA-grade institutional equity research using the local LLM on Raspberry Pi 5."""
    ticker = financials.get("ticker", "UNKNOWN")
    company_name = financials.get("company_name", ticker)
    pe = financials.get("pe_ratio")
    roe = financials.get("roe")
    de = financials.get("debt_to_equity")
    margin = financials.get("operating_margin")
    fcf = financials.get("free_cashflow_cr")
    price = financials.get("current_price")
    sector = financials.get("sector", "Indian Markets")
    industry = financials.get("industry", sector)

    score = calculate_institutional_score(financials, dorsey_data=dorsey_data)

    pe_str = f"{pe}x" if pe is not None else "N/A"
    roe_str = f"{roe}%" if roe is not None else "N/A"
    de_str = f"{de}" if de is not None else "N/A"
    margin_str = f"{margin}%" if margin is not None else "N/A"
    fcf_str = f"₹{fcf:,.1f} Cr" if fcf is not None else "N/A"
    price_str = f"₹{price}" if price is not None else "N/A"

    # Extract quant overlays for prompt context
    pio_str = "N/A"
    dup_str = "N/A"
    sloan_str = "N/A"
    comb_str = "N/A"

    if dorsey_data and isinstance(dorsey_data, dict):
        p_val = dorsey_data.get("piotroski_f_score", {}).get("score")
        if p_val is not None:
            pio_str = f"{p_val}/9 ({dorsey_data.get('piotroski_f_score', {}).get('rating', '')})"
        dup_summary = dorsey_data.get("dupont_analysis", {}).get("summary")
        if dup_summary:
            dup_str = dup_summary
        sloan_val = dorsey_data.get("sloan_accrual", {}).get("assessment")
        if sloan_val:
            sloan_str = f"{sloan_val} ({dorsey_data.get('sloan_accrual', {}).get('accrual_ratio_pct', 0)}% accrual)"
        c_val = dorsey_data.get("valuation", {}).get("combined", {}).get("combined_value")
        mos_val = dorsey_data.get("valuation", {}).get("combined", {}).get("margin_of_safety")
        if c_val is not None:
            comb_str = f"₹{c_val} (Margin of Safety: {mos_val:+.1f}%)"

    user_prompt = (
        f"Institutional Equity Research Prompt for {company_name} ({ticker}):\n"
        f"Sector: {sector} | Industry: {industry}\n"
        f"• Market Price: {price_str} | Trailing P/E: {pe_str}\n"
        f"• Capital Efficiency (ROE): {roe_str} | Operating Margin: {margin_str}\n"
        f"• Balance Sheet Solvency (Debt/Equity): {de_str} | Free Cash Flow: {fcf_str}\n"
        f"• Piotroski F-Score: {pio_str} | DuPont ROE Engine: {dup_str}\n"
        f"• Sloan Earnings Quality: {sloan_str} | Combined Intrinsic Value: {comb_str}\n\n"
        f"Please write a top-tier institutional equity research assessment:\n"
        f"**AI Verdict:** 2-3 analytical sentences evaluating whether current price ({price_str}) offers an asymmetric risk/reward entry relative to combined intrinsic value ({comb_str}), Piotroski score ({pio_str}), and cash flow generation.\n"
        f"**Moat Analysis:** 2-3 insightful sentences evaluating its economic moat and DuPont return driver ({dup_str}) sustaining its {margin_str} operating margin in {industry}.\n"
        f"**Top Risks:** Exactly 2 detailed bullet points highlighting the biggest balance sheet, customer, or macro headwinds."
    )

    system_prompt = (
        "You are a Lead Equity Research Analyst at a premier investment firm analyzing Indian equities. "
        "Provide rigorous, concise, numbers-grounded financial analysis. Reference exact data points provided."
    )

    payload = json.dumps({
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.15,
        "max_tokens": 380
    }).encode("utf-8")

    for url in LLAMA_HOST_URLS:
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=25.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                raw_content = res_data["choices"][0]["message"]["content"]
                extracted = _extract_llm_fields(raw_content)
                extracted["ai_score"] = score
                return extracted
        except Exception as e:
            logger.debug(f"AI host {url} failed: {e}")
            continue

    # Deterministic fallback
    return {
        "ai_score": score,
        "ai_verdict": f"Combined intrinsic value stands at {comb_str} with {pe_str} P/E and {roe_str} ROE (Piotroski: {pio_str}). Local LLM inference busy.",
        "moat_analysis": f"Operating margin of {margin_str} backed by {dup_str}.",
        "top_risks": f"• Leverage & Solvency: Debt-to-Equity ratio of {de_str}.\n• Margin Sensitivity: Exposure to operating cost inflation and sector cyclicality."
    }

def _save_to_cache(financials: dict, ai_result: dict):
    """Saves a live AI result back to stocks.db so subsequent visits are instant."""
    try:
        db_path = None
        for p in DB_PATHS:
            parent = os.path.dirname(p)
            if os.path.exists(parent):
                db_path = p
                break
        if not db_path:
            return

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_reports (
                ticker TEXT PRIMARY KEY, company_name TEXT, sector TEXT,
                current_price REAL, pe_ratio REAL, debt_to_equity REAL, roe REAL,
                ai_score INTEGER, ai_verdict TEXT, moat_analysis TEXT,
                top_risks TEXT, updated_at TIMESTAMP
            )
        """)
        from datetime import datetime
        cursor.execute("""
            INSERT OR REPLACE INTO stock_reports
            (ticker, company_name, sector, current_price, pe_ratio, debt_to_equity, roe,
             ai_score, ai_verdict, moat_analysis, top_risks, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            financials.get("ticker"), financials.get("company_name"), financials.get("sector"),
            financials.get("current_price"), financials.get("pe_ratio"),
            financials.get("debt_to_equity"), financials.get("roe"),
            ai_result.get("ai_score"),
            str(ai_result.get("ai_verdict", "")),
            str(ai_result.get("moat_analysis", "")),
            str(ai_result.get("top_risks", "")),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()
        logger.info(f"Cached AI report for {financials.get('ticker')} to {db_path}")
    except Exception as e:
        logger.debug(f"Cache save failed: {e}")

def get_stock_ai_intelligence(ticker: str, company_name: str = None, dorsey_data: dict = None) -> dict:
    """Main entry point for Flask web views."""
    # 1. Check pre-computed cache first (instant: 0.001s)
    cached = get_cached_ai_report(ticker)
    if cached and cached.get("ai_verdict"):
        return cached

    # 2. Fetch live metrics and run live analysis
    financials = fetch_indian_stock_data(ticker)
    if not financials or not financials.get("current_price"):
        return {
            "ai_score": None,
            "ai_verdict": f"Live financial data currently syncing for {ticker}.",
            "moat_analysis": "Established market presence in domestic industry.",
            "top_risks": "Macroeconomic and sector cyclicality."
        }

    result = query_live_edge_ai(financials, dorsey_data=dorsey_data)

    # 3. Auto-cache the result
    _save_to_cache(financials, result)

    return result
