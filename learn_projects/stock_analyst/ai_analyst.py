import os
import re
import json
import logging
import sqlite3
import urllib.request
import urllib.error

logger = logging.getLogger("ai_analyst")

try:
    from src.fetcher import fetch_indian_stock_data
except ImportError:
    from fetcher import fetch_indian_stock_data

LLAMA_HOST_URLS = [
    os.getenv("LLAMA_SERVER_URL", "http://172.17.0.1:8080/v1/chat/completions"),
    "http://127.0.0.1:8080/v1/chat/completions",
    "http://host.docker.internal:8080/v1/chat/completions"
]

DB_PATHS = [
    "/app/data/stocks.db",
    "/home/puneeth/repo/stock_fundamental/data/stocks.db",
    "/home/puneeth/repo/ai_ms_python/learn_projects/stock_analyst/stocks.db"
]

def calculate_institutional_score(financials: dict) -> int:
    """
    Computes an objective, multi-factor institutional score (1-10)
    combining Capital Efficiency, Balance Sheet Solvency, Pricing Power, and Valuation.
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
    """
    Extracts structured fields from LLM response (supports JSON and Markdown formats).
    """
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

def query_live_edge_ai(financials: dict) -> dict:
    """
    Generates high-conviction, CFA-grade institutional equity research
    using the local LLM on Raspberry Pi 5.
    """
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

    score = calculate_institutional_score(financials)

    pe_str = f"{pe}x" if pe is not None else "N/A"
    roe_str = f"{roe}%" if roe is not None else "N/A"
    de_str = f"{de}" if de is not None else "N/A"
    margin_str = f"{margin}%" if margin is not None else "N/A"
    fcf_str = f"₹{fcf:,.1f} Cr" if fcf is not None else "N/A"
    price_str = f"₹{price}" if price is not None else "N/A"

    user_prompt = (
        f"Institutional Equity Research Prompt for {company_name} ({ticker}):\n"
        f"Sector: {sector} | Industry: {industry}\n"
        f"• Market Price: {price_str} | Trailing P/E: {pe_str}\n"
        f"• Capital Efficiency (ROE): {roe_str} | Operating Margin: {margin_str}\n"
        f"• Balance Sheet Solvency (Debt/Equity): {de_str} | Free Cash Flow: {fcf_str}\n\n"
        f"Please write a top-tier institutional equity research assessment:\n"
        f"**AI Verdict:** 2-3 analytical sentences evaluating whether the current P/E ({pe_str}) offers an asymmetric risk/reward entry given its {roe_str} ROE and free cash flow generation.\n"
        f"**Moat Analysis:** 2-3 insightful sentences evaluating its economic moat (high switching costs, cost leadership, brand intangibles, or network effects) that sustain its {margin_str} operating margin in {industry}.\n"
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

    # Clean deterministic fallback if LLM is offline
    return {
        "ai_score": score,
        "ai_verdict": f"Fundamental valuation multiple stands at {pe_str} P/E with {roe_str} Return on Equity and {de_str} Debt/Equity leverage. (Real-time local LLM inference currently busy/offline).",
        "moat_analysis": f"Operating margin of {margin_str} reflects competitive standing in the {industry} sector.",
        "top_risks": f"• Leverage & Solvency: Debt-to-Equity ratio of {de_str}.\n• Margin Sensitivity: Exposure to operating cost inflation and industry cyclicality."
    }

def _save_to_cache(financials: dict, ai_result: dict):
    """Saves a live AI result back to stocks.db so subsequent visits are instant."""
    try:
        import sqlite3
        from datetime import datetime
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

    result = query_live_edge_ai(financials)

    # 3. Auto-cache the result so next visit loads in 0.001s
    _save_to_cache(financials, result)

    return result
