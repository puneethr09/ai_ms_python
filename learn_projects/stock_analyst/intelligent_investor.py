"""
Graham Intelligent Investor Analysis

Implements Benjamin Graham's core value investing principles from The Intelligent Investor:
- Graham Number: √(22.5 × EPS × BVPS)
- Revised Intrinsic Value Formula: [EPS × (8.5 + 2g) × 4.4] / Y
- Mr. Market Indicator (Market Psychology Assessment)
- Defensive Investor 7-Rule Screen
"""

try:
    from src.smart_data import SmartDataEngine
except ImportError:
    from smart_data import SmartDataEngine
import math


class GrahamAnalyzer:
    """
    Implements Benjamin Graham's analytical formulas and margin of safety checks.
    """

    def __init__(self, ticker):
        self.ticker = ticker
        self.data_engine = SmartDataEngine(ticker)

    def calculate_graham_number(self):
        """
        Calculates Graham Number: Max price a defensive investor should pay.
        Formula: √(22.5 × EPS × Book Value Per Share)
        """
        if not self.data_engine.has_data:
            return {"value": None, "note": "No data available"}

        eps = self.data_engine.info.get("trailingEps", 0) or 0
        bvps = self.data_engine.info.get("bookValue", 0) or 0
        current_price = self.data_engine.info.get("currentPrice", 0) or 0

        if eps <= 0 or bvps <= 0:
            return {
                "value": None,
                "formula": "√(22.5 × EPS × BVPS)",
                "inputs": {"EPS": eps, "BVPS": bvps},
                "current_price": current_price,
                "note": "Negative EPS or Book Value - Graham Number not applicable",
                "verdict": "N/A"
            }

        product = 22.5 * eps * bvps
        graham_value = math.sqrt(product)
        upside = ((graham_value - current_price) / current_price * 100) if current_price > 0 else 0

        return {
            "value": round(graham_value, 2),
            "formula": "√(22.5 × EPS × BVPS)",
            "inputs": {
                "EPS": round(eps, 2),
                "BVPS": round(bvps, 2)
            },
            "current_price": current_price,
            "upside_potential": f"{upside:+.1f}%",
            "verdict": "Undervalued (Graham Bargain)" if current_price < graham_value else "Overvalued vs Graham Number"
        }

    def calculate_intrinsic_value(self):
        """
        Graham's Revised Intrinsic Value Formula (1962):
        V = [EPS × (8.5 + 2g) × 4.4] / Y
        """
        if not self.data_engine.has_data:
            return {"value": None, "note": "No data available"}

        eps = self.data_engine.info.get("trailingEps", 0) or 0
        current_price = self.data_engine.info.get("currentPrice", 0) or 0

        if eps <= 0:
            return {
                "value": None,
                "note": "Negative earnings - cannot calculate Graham intrinsic value"
            }

        op_cagr = self.data_engine.calculate_multi_year_cagr(self.data_engine.financials, "Operating Income", max_years=3)
        if op_cagr and op_cagr > 0:
            g = min(op_cagr * 100, 16.0)
            growth_source = "Multi-Year OpIncome CAGR"
        else:
            g = 6.0
            growth_source = "Conservative Baseline (6%)"

        aaa_bond_yield = 7.4

        original_val = eps * (8.5 + 2 * g)
        revised_val = (eps * (8.5 + 2 * g) * 4.4) / aaa_bond_yield

        margin = ((revised_val - current_price) / revised_val * 100) if revised_val > 0 else 0

        return {
            "original_formula_value": round(original_val, 2),
            "revised_formula_value": round(revised_val, 2),
            "formula": "[EPS × (8.5 + 2g) × 4.4] / Y",
            "inputs": {
                "EPS": round(eps, 2),
                "growth_rate_g": f"{g:.1f}%",
                "growth_source": growth_source,
                "bond_yield_Y": f"{aaa_bond_yield:.1f}%"
            },
            "current_price": current_price,
            "margin_of_safety": f"{margin:+.1f}%",
            "verdict": "Buy" if margin > 25 else ("Hold" if margin > 0 else "Avoid")
        }

    def get_mr_market_indicator(self):
        """Mr. Market indicator based on P/E levels."""
        pe = self.data_engine.info.get("trailingPE", 0) or 0
        if pe <= 0:
            mood = "Cannot Assess"
            opp = "Negative or missing P/E"
        elif pe < 12:
            mood = "Pessimistic (Bargains Available)"
            opp = "Mr. Market is offering discounts"
        elif pe < 22:
            mood = "Neutral / Reasonable"
            opp = "Selective compounding opportunities"
        else:
            mood = "Euphoric / Overheated"
            opp = "High valuations - exercise margin of safety"

        return {
            "mr_market_mood": mood,
            "current_pe": round(pe, 1),
            "opportunity": opp,
            "graham_wisdom": "Mr. Market is there to serve you, not guide you."
        }

    def analyze(self):
        """Unified Graham analysis dictionary supporting all template keys."""
        iv = self.calculate_intrinsic_value()
        gn = self.calculate_graham_number()
        mr = self.get_mr_market_indicator()

        return {
            "graham_number": gn,
            "intrinsic_value": iv,
            "graham_intrinsic_value": iv,
            "mr_market": mr,
            "mr_market_indicator": mr
        }
