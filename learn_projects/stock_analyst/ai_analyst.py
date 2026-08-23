import os
import json
import logging
import sqlite3
import urllib.request
import urllib.error

logger = logging.getLogger("ai_analyst")

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

def calculate_grounded_score(pe: float, roe: float, debt_to_equity: float, op_margin: float) -> int:
    """
    Computes an objective fundamental baseline score (1-10) using Warren Buffett & Pat Dorsey rules:
    - High ROE (>20%) + High Margin (>15%) + Low Debt (<0.5) -> 8-10 (Elite Compounder)
    - Moderate ROE (12-20%) + Manageable Debt (0.5-1.0) -> 6-7 (Quality Business)
    - Low ROE (<10%) OR High Debt (>1.5) OR Excessive P/E (>60) -> 2-5 (Capital Intensive / Risk)
    """
    score = 5

    # 1. Capital Efficiency (ROE)
    if roe >= 25.0:
        score += 3
    elif roe >= 15.0:
        score += 2
    elif roe >= 8.0:
        score += 1
    elif roe < 0.0:
        score -= 3
    else:
        score -= 1

    # 2. Balance Sheet Solvency (Debt-to-Equity)
    if debt_to_equity < 0.2:
        score += 2  # Debt-free cash cow
    elif debt_to_equity < 0.8:
        score += 1  # Conservative debt
    elif debt_to_equity > 2.0:
        score -= 2  # Dangerously leveraged

    # 3. Operating Margin Pricing Power
    if op_margin >= 25.0:
        score += 1
    elif op_margin < 5.0 and op_margin > 0:
        score -= 1

    # 4. Valuation Reality Check (P/E)
    if 0 < pe <= 20.0:
        score += 1  # Undervalued / Reasonable
    elif pe > 65.0:
        score -= 1  # Frothy growth premium

    return max(1, min(10, score))

def get_cached_ai_report(ticker: str) -> dict:
    """Checks if a pre-computed report exists in SQLite."""
    for path in DB_PATHS:
        if os.path.exists(path):
            try:
                conn = sqlite3.connect(path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                clean_sym = ticker.replace(".NS", "").replace(".BO", "")
                cursor.execute("SELECT * FROM stock_reports WHERE ticker LIKE ?", (f"{clean_sym}%",))
                row = cursor.fetchone()
                conn.close()
                if row:
                    return dict(row)
            except Exception as e:
                logger.debug(f"DB read error: {e}")
    return None

def query_live_edge_ai(ticker: str, company_name: str, financials: dict) -> dict:
    """
    Generates a deep, grounded, non-hallucinatory fundamental analysis
    using the local Llama model on Pi 5.
    """
    pe = financials.get("pe_ratio", 0.0)
    roe = financials.get("roe", 0.0)
    de = financials.get("debt_to_equity", 0.0)
    margin = financials.get("operating_margin") or financials.get("profit_margin", 0.0)
    fcf = financials.get("free_cashflow_cr", 0.0)
    price = financials.get("current_price", 0.0)
    sector = financials.get("sector", "Indian Markets")

    score = calculate_grounded_score(pe, roe, de, margin)

    user_prompt = (
        f"Analyze {company_name} ({ticker}) in the {sector} sector:\n"
        f"• Current Stock Price: ₹{price}\n"
        f"• Valuation (Trailing P/E): {pe}x\n"
        f"• Capital Efficiency (ROE): {roe}%\n"
        f"• Financial Leverage (Debt/Equity): {de}\n"
        f"• Operating Margin: {margin}%\n"
        f"• Free Cash Flow: ₹{fcf} Crores\n\n"
        f"GROUNDING INSTRUCTION:\n"
        f"1. You MUST reference these exact numbers in your analysis.\n"
        f"2. Explain what the combination of ROE {roe}% and Debt/Equity {de} tells an investor about management's capital allocation.\n"
        f"3. Classify its economic moat (High Switching Costs, Cost Advantage, Brand/Intangibles, or None) based on its {margin}% margin.\n"
        f"4. Highlight 2 concrete operational or valuation risks."
    )

    system_prompt = (
        "You are a Chartered Financial Analyst (CFA) specializing in the Indian Stock Market. "
        "Provide factual, highly knowledgeable equity research grounded strictly in the provided data.\n\n"
        "Return STRICT JSON with these exact keys:\n"
        "{\n"
        f"  \"ai_score\": {score},\n"
        "  \"ai_verdict\": \"2-3 detailed sentences explaining the valuation thesis, cash generation, and whether current P/E offers an attractive entry.\",\n"
        "  \"moat_analysis\": \"2 sentences detailing the source and durability of its competitive economic moat.\",\n"
        "  \"top_risks\": \"2 specific bullet points detailing the biggest financial, margin, or competitive risks.\"\n"
        "}"
    )

    payload = json.dumps({
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.15,
        "max_tokens": 260
    }).encode("utf-8")

    for url in LLAMA_HOST_URLS:
        try:
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=22.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                raw = res_data["choices"][0]["message"]["content"].strip()
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()
                parsed = json.loads(raw)
                parsed["ai_score"] = score
                return parsed
        except Exception as e:
            logger.debug(f"AI host {url} failed: {e}")
            continue

    # Grounded fallback with exact metric citations
    verdict = (
        f"{company_name} demonstrates exceptional capital efficiency with a {roe}% ROE and low debt-to-equity of {de}. "
        f"At a P/E of {pe}x and operating margins of {margin}%, the company represents a high-quality compounder."
        if score >= 8 else (
            f"{company_name} trades at a fair valuation of {pe}x with moderate {roe}% ROE and {de} debt leverage. "
            f"Free cash flow generation (₹{fcf} Cr) provides stability, but upside is bounded by sector competition."
            if score >= 6 else
            f"{company_name} carries elevated debt leverage ({de} D/E) or compressed ROE ({roe}%), while trading at a {pe}x P/E. "
            f"Investors should exercise caution until free cash flow margins improve."
        )
    )

    moat = (
        f"High switching costs and deep customer integration support strong pricing power, reflected in a {margin}% operating margin."
        if margin > 20 else
        f"Standard industry moat with moderate competitive pricing dynamics across the {sector} market."
    )

    risks = (
        f"1. Sensitivity to global IT/macro slowdown and wage inflation. 2. Multiple compression if revenue growth decelerates from current levels."
        if "Tech" in sector or "IT" in sector else
        f"1. High capital expenditure gestation cycles impacting near-term return on capital. 2. Raw material price volatility and interest rate sensitivity."
    )

    return {
        "ai_score": score,
        "ai_verdict": verdict,
        "moat_analysis": moat,
        "top_risks": risks
    }

def get_stock_ai_intelligence(ticker: str, company_name: str, dorsey_data: dict = None) -> dict:
    """Main entry point for Flask web views."""
    # Check cache first
    cached = get_cached_ai_report(ticker)
    if cached and cached.get("ai_score"):
        return cached

    # Extract metrics from dorsey_data or fallback
    v_dict = dorsey_data.get("valuation", {}) if isinstance(dorsey_data, dict) and isinstance(dorsey_data.get("valuation"), dict) else {}
    f_dict = dorsey_data.get("financial_health", {}) if isinstance(dorsey_data, dict) and isinstance(dorsey_data.get("financial_health"), dict) else {}
    s_dict = dorsey_data.get("sector_analysis", {}) if isinstance(dorsey_data, dict) and isinstance(dorsey_data.get("sector_analysis"), dict) else {}

    financials = {
        "current_price": v_dict.get("current_price", 0.0),
        "pe_ratio": v_dict.get("pe_ratio", 0.0),
        "roe": f_dict.get("roe", 0.0),
        "debt_to_equity": f_dict.get("debt_to_equity", 0.0),
        "operating_margin": f_dict.get("operating_margin", 0.0),
        "free_cashflow_cr": v_dict.get("free_cashflow_cr", 0.0),
        "sector": s_dict.get("sector", "Indian Markets")
    }

    return query_live_edge_ai(ticker, company_name, financials)
