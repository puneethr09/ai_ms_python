"""
Dorsey Runner - Main Integration & Quantitative Orchestrator

Executes the complete Dorsey Protocol & Institutional Quant Screen:
- Special Situations & Anomaly Detection (HoldCo, SOTP, Cyclicals)
- Ten-Minute Test (Chapter 12)
- Moat Analysis & Pricing Power (Chapter 3)
- Financial Health, Graduated Red Flags & Piotroski F-Score (Chapters 5-8)
- DuPont 3-Way ROE Decomposition
- Sloan Accrual Ratio (Earnings Quality)
- 52-Week Range & Momentum Context
- Multi-Model Intrinsic Valuation & Margin of Safety (Chapters 9-10)
- 100-Point Rebalanced Scorecard (Chapter 11)
- Graham Value Investing Analysis (Intelligent Investor)
"""

from src.ten_minute_test import TenMinuteTest
from src.dorsey_core.moat import MoatAnalyzer
from src.dorsey_core.financials import FinancialsAnalyzer
from src.dorsey_core.valuation import ValuationAnalyzer
from src.dorsey_core.mistake_detector import MistakeDetector
from src.dorsey_core.scorecard import DorseyScorecard
from src.dorsey_sectors.factory import SectorFactory
from src.graham.intelligent_investor import GrahamAnalyzer
from src.graham.defensive_criteria import DefensiveInvestorScreen
from src.smart_data import SmartDataEngine
from src.anomaly_detector import AnomalyDetector


def run_dorsey_analysis(ticker):
    """
    Executes the full Dorsey, Quant & Anomaly Protocol.
    Returns a unified dictionary for the Flask UI and LLM intelligence card.
    """
    results = {}
    data_engine = SmartDataEngine(ticker)

    # 1. Anomaly & Special Situation Detection
    ad = AnomalyDetector(ticker)
    results["special_situation"] = ad.detect_special_situations()

    # 2. 52-Week Range Context
    results["momentum_52w"] = data_engine.get_52w_position()

    # 3. Ten Minute Test (Chapter 12)
    tm = TenMinuteTest(ticker)
    results["ten_minute_test"] = tm.run_test()

    # 4. Moat Analysis (Chapter 3)
    m = MoatAnalyzer(ticker)
    results["moat_analysis"] = m.analyze_moat()

    # 5. Financial Health & Forensic Quant (Chapters 5-8)
    f = FinancialsAnalyzer(ticker)
    health_result = f.analyze_health()
    results["financial_health"] = health_result
    results["piotroski_f_score"] = health_result.get("piotroski_f_score", {})
    results["dupont_analysis"] = health_result.get("dupont_analysis", {})
    results["sloan_accrual"] = health_result.get("sloan_accrual", {})

    # 6. Valuation (Chapters 9-10 with HoldCo adjustment)
    v = ValuationAnalyzer(ticker)
    results["valuation"] = v.get_valuation_verdict()

    # 7. Mistake Detection (Chapter 2)
    md = MistakeDetector(ticker)
    results["mistake_warnings"] = md.detect_mistakes()

    # 8. Sector Specifics (Chapters 13-26)
    try:
        s = SectorFactory.get_strategy(ticker)
        results["sector_analysis"] = s.analyze()
        results["sector_chapter"] = SectorFactory.get_chapter_reference(ticker)
    except Exception:
        results["sector_analysis"] = {}
        results["sector_chapter"] = "General"

    # 9. Rebalanced Composite Scorecard (Chapter 11)
    scorecard = DorseyScorecard(ticker)
    full_scorecard = scorecard.generate_scorecard()
    results["scorecard"] = full_scorecard

    # 10. Graham Analysis
    try:
        graham = GrahamAnalyzer(ticker)
        graham_result = graham.analyze()
        results["graham_analysis"] = graham_result

        defensive = DefensiveInvestorScreen(ticker)
        defensive_result = defensive.screen()
        results["graham_defensive_screen"] = {
            "passed": defensive_result.get("summary", {}).get("passed", 0),
            "total": defensive_result.get("summary", {}).get("total", 7),
            "verdict": defensive_result.get("summary", {}).get("verdict", "N/A"),
            "criteria": defensive_result.get("criteria", [])
        }
    except Exception as e:
        results["graham_analysis"] = {"error": str(e)}
        results["graham_defensive_screen"] = {"error": str(e)}

    return results


def run_quick_analysis(ticker):
    """Runs a quick analysis with just the scorecard summary."""
    scorecard = DorseyScorecard(ticker)
    return scorecard.get_quick_summary()
