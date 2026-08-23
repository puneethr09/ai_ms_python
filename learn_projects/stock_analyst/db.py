import sqlite3
import json
from datetime import datetime

DB_PATH = "stocks.db"

def init_db():
    """Initializes the SQLite database schema for stock fundamental intelligence."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_reports (
            ticker TEXT PRIMARY KEY,
            company_name TEXT,
            sector TEXT,
            current_price REAL,
            pe_ratio REAL,
            debt_to_equity REAL,
            roe REAL,
            ai_score INTEGER,
            ai_verdict TEXT,
            moat_analysis TEXT,
            top_risks TEXT,
            updated_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def _stringify_field(val) -> str:
    """Safely converts lists or dicts into clean readable strings for SQLite."""
    if isinstance(val, list):
        return "\n• " + "\n• ".join(str(item) for item in val)
    elif isinstance(val, dict):
        return json.dumps(val, indent=2)
    return str(val) if val is not None else ""

def save_stock_report(data: dict):
    """Inserts or updates a stock report in SQLite with safe type sanitization."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    verdict_str = _stringify_field(data.get("ai_verdict"))
    moat_str = _stringify_field(data.get("moat_analysis"))
    risks_str = _stringify_field(data.get("top_risks"))

    cursor.execute("""
        INSERT OR REPLACE INTO stock_reports 
        (ticker, company_name, sector, current_price, pe_ratio, debt_to_equity, roe, ai_score, ai_verdict, moat_analysis, top_risks, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("ticker"),
        data.get("company_name"),
        data.get("sector"),
        data.get("current_price"),
        data.get("pe_ratio"),
        data.get("debt_to_equity"),
        data.get("roe"),
        data.get("ai_score"),
        verdict_str,
        moat_str,
        risks_str,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def get_stock_report(ticker: str) -> dict:
    """Retrieves the latest pre-computed report for a ticker."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM stock_reports WHERE ticker = ? OR ticker LIKE ?", (ticker, f"{ticker}%"))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_reports() -> list:
    """Retrieves all reports for a morning summary board."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, company_name, current_price, pe_ratio, ai_score, ai_verdict, updated_at FROM stock_reports ORDER BY ai_score DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
