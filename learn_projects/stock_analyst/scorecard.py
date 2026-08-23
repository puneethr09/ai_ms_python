"""
Dorsey Scorecard - Chapter 11 (Composite Investment Decision Engine)

Integrates:
- Ten-Minute Quality Filter (20 pts)
- Moat Analysis & Pricing Power (25 pts)
- Financial Health, Red Flags & Piotroski F-Score (25 pts)
- Combined Intrinsic Valuation & Margin of Safety (30 pts)
"""

try:
    from src.smart_data import SmartDataEngine
    from src.ten_minute_test import TenMinuteTest
    from src.dorsey_core.moat import MoatAnalyzer
    from src.dorsey_core.financials import FinancialsAnalyzer
    from src.dorsey_core.valuation import ValuationAnalyzer
    from src.dorsey_sectors.factory import SectorFactory
except ImportError:
    from smart_data import SmartDataEngine
    from ten_minute_test import TenMinuteTest
    from moat import MoatAnalyzer
    from financials import FinancialsAnalyzer
    from valuation import ValuationAnalyzer
    from dorsey_sectors.factory import SectorFactory


class DorseyScorecard:
    """
    100-point institutional composite scorecard calibrated to balance
    intrinsic valuation with qualitative moat durability and balance sheet forensics.
    """

    def __init__(self, ticker):
        self.ticker = ticker
        self.data_engine = SmartDataEngine(ticker)

    def generate_scorecard(self):
        """Generates comprehensive 100-point Dorsey & Quant scorecard."""
        if not self.data_engine.has_data:
            return {
                "ticker": self.ticker,
                "status": "NO DATA",
                "recommendation": "CANNOT ANALYZE"
            }

        company_name = self.data_engine.info.get("longName", self.ticker)
        current_price = self.data_engine.info.get("currentPrice", 0) or 0
        sector = self.data_engine.info.get("sector", "Unknown")

        # 1. Ten-Minute Test (20 pts)
        ten_min = TenMinuteTest(self.ticker)
        ten_min_result = ten_min.run_test()

        # 2. Moat Analysis (25 pts)
        moat_analyzer = MoatAnalyzer(self.ticker)
        moat_result = moat_analyzer.analyze_moat()

        # 3. Financial Health & Piotroski (25 pts)
        financials = FinancialsAnalyzer(self.ticker)
        health_result = financials.analyze_health()

        # 4. Valuation & Margin of Safety (30 pts)
        valuation = ValuationAnalyzer(self.ticker)
        val_verdict = valuation.get_valuation_verdict()
        combined_val = val_verdict.get("combined", {})
        margin_of_safety = combined_val.get("margin_of_safety", 0)
        val_assessment = combined_val.get("verdict", "FAIRLY VALUED")

        # 5. Sector Specifics
        try:
            sector_strat = SectorFactory.get_strategy(self.ticker)
            sector_result = sector_strat.analyze()
            sector_chapter = SectorFactory.get_chapter_reference(self.ticker)
        except Exception:
            sector_result = {}
            sector_chapter = "General"

        # --- COMPOSITE SCORING (100 Points Total) ---
        total_score = 0.0

        # Component 1: Ten-Minute Test (20 pts max)
        tm_raw = ten_min_result.get("score", 0)
        tm_max = ten_min_result.get("max_score", 8)
        score_tm = (tm_raw / tm_max) * 20.0 if tm_max > 0 else 10.0
        total_score += score_tm

        # Component 2: Moat Rating (25 pts max)
        moat_map = {
            "Wide Moat": 25.0,
            "Narrow Moat": 18.0,
            "Possible Moat": 10.0,
            "No Moat": 0.0
        }
        score_moat = moat_map.get(moat_result.get("moat_rating", "No Moat"), 5.0)
        total_score += score_moat

        # Component 3: Financial Health & Forensic Checks (25 pts max)
        score_health = float(health_result.get("health_score", 15.0))
        total_score += score_health

        # Component 4: Intrinsic Valuation & Margin of Safety (30 pts max)
        if margin_of_safety >= 25.0:
            score_val = 30.0
        elif margin_of_safety >= 10.0:
            score_val = 22.0
        elif margin_of_safety >= -10.0:
            score_val = 14.0
        elif margin_of_safety >= -25.0:
            score_val = 6.0
        else:
            score_val = 0.0
        total_score += score_val

        # --- FINAL RECOMMENDATION ---
        red_count = len(health_result.get("red_flags", []))
        p_score = health_result.get("piotroski_f_score", {}).get("score", 5)

        if total_score >= 78.0 and red_count == 0 and p_score >= 6:
            recommendation = "STRONG BUY"
            confidence = "HIGH"
        elif total_score >= 65.0 and red_count <= 1:
            recommendation = "BUY"
            confidence = "MODERATE-HIGH"
        elif total_score >= 50.0:
            recommendation = "HOLD / WATCHLIST"
            confidence = "MODERATE"
        elif total_score >= 35.0:
            recommendation = "AVOID"
            confidence = "MODERATE"
        else:
            recommendation = "STRONG AVOID"
            confidence = "HIGH"

        scorecard = {
            "ticker": self.ticker,
            "company": company_name,
            "sector": sector,
            "sector_chapter": sector_chapter,
            "current_price": current_price,
            "scores": {
                "ten_minute_test": {
                    "score": round(score_tm, 1),
                    "max": 20,
                    "verdict": ten_min_result.get("overall_verdict", "PASS")
                },
                "moat": {
                    "score": round(score_moat, 1),
                    "max": 25,
                    "rating": moat_result.get("moat_rating", "Unknown")
                },
                "financial_health": {
                    "score": round(score_health, 1),
                    "max": 25,
                    "rating": health_result.get("health_rating", "Unknown"),
                    "red_flags": red_count,
                    "piotroski_f_score": p_score
                },
                "valuation": {
                    "score": round(score_val, 1),
                    "max": 30,
                    "assessment": val_assessment,
                    "margin_of_safety": f"{margin_of_safety:+.1f}%",
                    "combined_intrinsic_value": combined_val.get("combined_value")
                }
            },
            "total_score": round(total_score, 1),
            "max_score": 100,
            "percentage": f"{(total_score/100)*100:.0f}%",
            "recommendation": recommendation,
            "confidence": confidence,
            "dorsey_five_rules_check": {
                "1_do_homework": ten_min_result.get("passed", False),
                "2_find_moats": moat_result.get("moat_rating", "") in ["Wide Moat", "Narrow Moat"],
                "3_margin_of_safety": margin_of_safety > 15.0,
                "4_hold_long_term": moat_result.get("moat_rating", "") == "Wide Moat",
                "5_know_when_to_sell": red_count == 0
            },
            "summary": f"{recommendation} ({total_score:.0f}/100) — {moat_result.get('moat_rating', 'Moat')} with {health_result.get('health_rating', 'Health')} balance sheet. Trading at {margin_of_safety:+.1f}% margin of safety to combined intrinsic value.",
            "key_risks": health_result.get("red_flags", []) + health_result.get("amber_flags", [])
        }

        return scorecard

    def get_quick_summary(self):
        """Returns concise summary for batch screening."""
        sc = self.generate_scorecard()
        return {
            "ticker": sc["ticker"],
            "company": sc["company"],
            "score": f"{sc['total_score']}/{sc['max_score']}",
            "recommendation": sc["recommendation"],
            "moat": sc["scores"]["moat"]["rating"],
            "valuation": sc["scores"]["valuation"]["assessment"],
            "health": sc["scores"]["financial_health"]["rating"],
            "summary": sc["summary"]
        }
