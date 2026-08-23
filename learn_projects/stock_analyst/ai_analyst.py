import os
import re
import json
import logging
import sqlite3
import urllib.request
import urllib.error
from fetcher import fetch_indian_stock_data

logger = logging.getLogger("ai_analyst")

LLAMA_HOST_URLS = [
    os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions"),
    "http://172.17.0.1:8080/v1/chat/completions",
    "http://host.docker.internal:8080/v1/chat/completions"
]

DB_PATHS = [
    "/app/data/stocks.db",
    "/home/puneeth/repo/stock_fundamental/data/stocks.db",
    "/home/puneeth/repo/ai_ms_python/learn_projects/stock_analyst/stocks.db"
]

def calculate_grounded_score(pe: float, roe: float, debt_to_equity: float, op_margin: float) -> int:
    """
    Computes a baseline fundamental score (1-10) based on verified metrics.
    Gracefully handles None for missing values.
    """
    score = 5

    # ROE Capital Efficiency
    if roe is not None:
        if roe >= 25.0:
            score += 2
        elif roe >= 15.0:
            score += 1
        elif roe < 5.0:
            score -= 1

    # Debt-to-Equity Solvency
    if debt_to_equity is not None:
        if debt_to_equity < 15.0:
            score += 2
        elif debt_to_equity < 50.0:
            score += 1
        elif debt_to_equity > 150.0:
            score -= 2

    # Operating Margin Pricing Power
    if op_margin is not None:
        if op_margin >= 20.0:
            score += 1
        elif op_margin < 8.0:
            score -= 1

    # Valuation Multiple
    if pe is not None:
        if 0 < pe <= 20.0:
            score += 1
        elif pe > 60.0:
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
                cursor.execute("SELECT * FROM stock_reports WHERE ticker LIKE ?", (f"{clean_sym}%",))
                row = cursor.fetchone()
                conn.close()
                if row:
                    return dict(row)
            except Exception as e:
                logger.debug(f"DB read error: {e}")
    return None

def _extract_llm_fields(raw_text: str) -> dict:
    """
    Parses LLM output whether returned as JSON or Markdown headers.
    """
    raw_text = raw_text.strip()

    # 1. Try JSON parsing
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

    # 2. Try Markdown Header Extraction
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
            "moat_analysis": moat or "N/A",
            "top_risks": risks or "N/A"
        }

    # Fallback to raw text summary
    return {
        "ai_verdict": raw_text[:300],
        "moat_analysis": "N/A",
        "top_risks": "N/A"
    }

def query_live_edge_ai(financials: dict) -> dict:
    """
    Calls the local LLM on Raspberry Pi 5 with real verified metrics.
    """
    ticker = financials.get("ticker", "UNKNOWN")
    company_name = financials.get("company_name", ticker)
    pe = financials.get("pe_ratio")
    roe = financials.get("roe")
    de = financials.get("debt_to_equity")
    margin = financials.get("operating_margin")
    fcf = financials.get("free_cashflow_cr")
    price = financials.get("current_price")
    sector = financials.get("sector", "N/A")
    industry = financials.get("industry", "N/A")

    score = calculate_grounded_score(pe, roe, de, margin)

    pe_str = f"{pe}x" if pe is not None else "N/A"
    roe_str = f"{roe}%" if roe is not None else "N/A"
    de_str = f"{de}" if de is not None else "N/A"
    margin_str = f"{margin}%" if margin is not None else "N/A"
    fcf_str = f"₹{fcf} Cr" if fcf is not None else "N/A"
    price_str = f"₹{price}" if price is not None else "N/A"

    user_prompt = (
        f"Company: {company_name} ({ticker})\n"
        f"Sector: {sector} | Industry: {industry}\n"
        f"• Price: {price_str} | P/E: {pe_str} | ROE: {roe_str}\n"
        f"• Debt/Equity: {de_str} | Operating Margin: {margin_str} | Free Cash Flow: {fcf_str}\n\n"
        f"Please provide:\n"
        f"**AI Verdict:** 2 sentences on valuation and capital efficiency.\n"
        f"**Moat Analysis:** 2 sentences on pricing power and competitive moat in {industry}.\n"
        f"**Top Risks:** 2 concise bullet points on balance sheet or market headwinds."
    )

    system_prompt = (
        "You are an equity research analyst. Analyze the company based strictly on the provided data. "
        "Do not invent metrics."
    )

    payload = json.dumps({
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 300
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

    return {
        "ai_score": score,
        "ai_verdict": f"P/E: {pe_str}, ROE: {roe_str}, Debt/Equity: {de_str}. (Live LLM evaluation unavailable)",
        "moat_analysis": f"Operating margin is {margin_str} in {industry}.",
        "top_risks": f"Debt/Equity: {de_str}, Margin: {margin_str}."
    }

def get_stock_ai_intelligence(ticker: str, company_name: str = None, dorsey_data: dict = None) -> dict:
    """Main entry point for Flask web views."""
    cached = get_cached_ai_report(ticker)
    if cached and cached.get("ai_verdict"):
        return cached

    financials = fetch_indian_stock_data(ticker)
    if not financials or not financials.get("current_price"):
        return {
            "ai_score": None,
            "ai_verdict": f"Financial data currently unavailable for {ticker}.",
            "moat_analysis": "N/A",
            "top_risks": "N/A"
        }

    return query_live_edge_ai(financials)
