import os
import sqlite3
import json
from datetime import datetime

# Primary shared database path (mounted into Docker web app)
DEFAULT_SHARED_PATH = "/home/puneeth/repo/stock_fundamental/data/stocks.db"
LOCAL_PATH = "stocks.db"

DB_PATH = DEFAULT_SHARED_PATH if os.path.exists(os.path.dirname(DEFAULT_SHARED_PATH)) else LOCAL_PATH

def get_db_connection():
    """Returns a connection to the primary SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database schema for stock fundamental intelligence."""
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db_connection()
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
    init_db()
    conn = get_db_connection()
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
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM stock_reports WHERE ticker = ? OR ticker LIKE ?", (ticker, f"{ticker}%"))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_reports() -> list:
    """Retrieves all reports for a morning summary board."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, company_name, current_price, pe_ratio, ai_score, ai_verdict, updated_at FROM stock_reports ORDER BY ai_score DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
