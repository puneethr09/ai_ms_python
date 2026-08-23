"""
Valuation Analyzer - Chapters 9-10 (Discounted Cash Flow & Relative Valuation)

Implements:
- Multi-Model Intrinsic Valuation (Standard FCF, FCFE for Financials, DDM for Utilities)
- Multi-Year Normalized Growth Calibration & Fundamental Reinvestment Rate
- WACC-Based Scenario Modeling (Conservative, Base, Optimistic)
- AlphaSpread-Style Combined Intrinsic Value (40% DCF + 60% Relative Multiples)
- Holding Company (HoldCo) 55% Discount Adjustment Model
- Greenwald's Earnings Power Value (EPV) & Peter Lynch PEG Valuation
"""

try:
    from src.smart_data import SmartDataEngine
except ImportError:
    from smart_data import SmartDataEngine


class ValuationAnalyzer:
    """
    Comprehensive valuation engine blending fundamental DCF with market-based relative valuation
    and holding company arbitrage detection.
    """

    # Sector Cost of Equity Risk Adjustments (India Market Context)
    SECTOR_COE_ADJUSTMENTS = {
        "technology": 0.0,
        "consumer defensive": -0.015,
        "consumer cyclical": 0.01,
        "financial services": 0.0,
        "real estate": 0.02,
        "utilities": -0.01,
        "energy": 0.005,
        "healthcare": 0.0,
        "industrials": 0.005,
        "basic materials": 0.01,
        "communication services": 0.005,
    }

    # Sector-Specific Valuation Multiples (India Multi-Year Baselines)
    SECTOR_MULTIPLES = {
        "consumer defensive": {"pe": 42, "ev_ebitda": 28, "pb": 10, "ps": 5},
        "consumer cyclical": {"pe": 35, "ev_ebitda": 18, "pb": 5, "ps": 2},
        "financial services": {"pe": 16, "ev_ebitda": None, "pb": 2.2, "ps": 3},
        "technology": {"pe": 27, "ev_ebitda": 18, "pb": 7, "ps": 4},
        "healthcare": {"pe": 35, "ev_ebitda": 20, "pb": 5, "ps": 4},
        "industrials": {"pe": 25, "ev_ebitda": 14, "pb": 4, "ps": 2},
        "energy": {"pe": 12, "ev_ebitda": 7, "pb": 1.5, "ps": 1},
        "basic materials": {"pe": 15, "ev_ebitda": 8, "pb": 2, "ps": 1.5},
        "utilities": {"pe": 18, "ev_ebitda": 10, "pb": 2, "ps": 2},
        "real estate": {"pe": 25, "ev_ebitda": 15, "pb": 2.5, "ps": 3},
        "communication services": {"pe": 30, "ev_ebitda": 12, "pb": 4, "ps": 3},
    }

    DEFAULT_MULTIPLES = {"pe": 23, "ev_ebitda": 12, "pb": 3, "ps": 2.5}

    def __init__(self, ticker):
        self.ticker = ticker
        self.data_engine = SmartDataEngine(ticker)
        # Base Cost of Equity & Long-Term Terminal Growth for India (7.1% Rf + 6.4% ERP)
        self.base_cost_of_equity = 0.135
        self.terminal_growth = 0.050

    def _calculate_wacc(self):
        """Calculates Weighted Average Cost of Capital (WACC)."""
        if not self.data_engine.has_data:
            return self.base_cost_of_equity

        market_cap = self.data_engine.info.get("marketCap", 0) or 0
        total_debt = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Total Debt", 0)
        interest_expense = abs(self.data_engine.get_financials_safe(self.data_engine.financials, "Interest Expense", 0))

        E = market_cap
        D = total_debt
        V = E + D

        if V <= 0 or E <= 0:
            return self.base_cost_of_equity

        sector = self.data_engine.info.get("sector", "").lower()
        sector_adj = self.SECTOR_COE_ADJUSTMENTS.get(sector, 0.0)
        Re = self.base_cost_of_equity + sector_adj

        Rd = min(interest_expense / D, 0.14) if (D > 0 and interest_expense > 0) else 0.08
        Tc = 0.25

        wacc = (E / V * Re) + (D / V * Rd * (1 - Tc))
        return max(0.08, min(wacc, 0.18))

    def _calculate_best_growth(self, growth_cap=0.18):
        """Calculates multi-year normalized CAGR and fundamental reinvestment rate."""
        if not self.data_engine.has_data:
            return 0.08

        growth_candidates = []

        rev_cagr = self.data_engine.calculate_multi_year_cagr(self.data_engine.financials, "Total Revenue", max_years=3)
        if rev_cagr and rev_cagr > 0:
            growth_candidates.append(("Revenue_3Y", rev_cagr))

        op_cagr = self.data_engine.calculate_multi_year_cagr(self.data_engine.financials, "Operating Income", max_years=3)
        if op_cagr and op_cagr > 0:
            growth_candidates.append(("OpIncome_3Y", op_cagr))

        ni_cagr = self.data_engine.calculate_multi_year_cagr(self.data_engine.financials, "Net Income", max_years=3)
        if ni_cagr and ni_cagr > 0:
            growth_candidates.append(("NetIncome_3Y", ni_cagr))

        roe_dec = (self.data_engine.info.get("returnOnEquity") or 0.15)
        payout = (self.data_engine.info.get("payoutRatio") or 0.40)
        retention = max(0.10, min(1.0 - payout, 0.90))
        fund_growth = max(0.04, min(roe_dec * retention, 0.20))
        growth_candidates.append(("Fundamental_ROE_Retention", fund_growth))

        best_name, best_rate = max(growth_candidates, key=lambda x: x[1])
        return max(0.04, min(best_rate, growth_cap))

    def _get_model_inputs(self):
        """Prepares balance sheet, cash flows, and growth parameters for DCF."""
        if not self.data_engine.has_data:
            return None

        sector = self.data_engine.info.get("sector", "").lower()
        industry = self.data_engine.info.get("industry", "").lower()

        is_financial = "financial" in sector or "bank" in industry or "insurance" in industry
        is_utility = "utility" in sector or "utilities" in industry or "energy" in sector

        model_type = "FCF (Standard)"
        equity_mode = False

        if is_financial:
            model_type = "FCFE (Financials)"
            equity_mode = True
            ni = self.data_engine.get_financials_safe(self.data_engine.financials, "Net Income", 0)
            capex = self.data_engine.get_financials_safe(self.data_engine.cashflow, "Capital Expenditure", 0)
            current_fcf = ni + (capex if capex < 0 else -capex)
            if current_fcf <= 0:
                current_fcf = ni * 0.70
        elif is_utility:
            raw_fcf = self.data_engine.calculate_fcf(0)
            ni = self.data_engine.get_financials_safe(self.data_engine.financials, "Net Income", 0)
            current_fcf = raw_fcf if raw_fcf > 0 else ni * 0.75
            model_type = "FCF (Regulated Utilities)"
        else:
            current_fcf = self.data_engine.calculate_fcf(0)
            if current_fcf <= 0:
                ni = self.data_engine.get_financials_safe(self.data_engine.financials, "Net Income", 0)
                current_fcf = ni * 0.80

        shares = self.data_engine.info.get("sharesOutstanding", 0) or 0
        if shares <= 0:
            return None

        debt = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Total Debt", 0)
        cash = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Cash And Cash Equivalents", 0)

        growth_cap = 0.20 if "tech" in sector else (0.14 if is_financial else 0.16)

        return {
            "model_type": model_type,
            "current_fcf": current_fcf,
            "shares": shares,
            "debt": debt,
            "cash": cash,
            "equity_mode": equity_mode,
            "growth_cap": growth_cap
        }

    def _compute_dcf(self, inputs, growth_rate, discount_rate, terminal_rate):
        """Executes a 10-year discounted cash flow computation with terminal value."""
        cf = inputs["current_fcf"]
        pv_fcf = 0.0

        for year in range(1, 11):
            if year > 5:
                rate = growth_rate - ((growth_rate - terminal_rate) * ((year - 5) / 5))
            else:
                rate = growth_rate

            cf = cf * (1 + rate)
            df = (1 + discount_rate) ** year
            pv_fcf += cf / df

        if discount_rate <= terminal_rate:
            terminal_rate = discount_rate - 0.02

        terminal_val = (cf * (1 + terminal_rate)) / (discount_rate - terminal_rate)
        pv_terminal = terminal_val / ((1 + discount_rate) ** 10)
        total_enterprise_value = pv_fcf + pv_terminal

        if inputs["equity_mode"]:
            equity_val = total_enterprise_value
        else:
            equity_val = total_enterprise_value - inputs["debt"] + inputs["cash"]

        return max(0.0, equity_val / inputs["shares"])

    def get_valuation_scenarios(self):
        """Returns 3 DCF scenarios: Conservative, Base, and Optimistic."""
        inputs = self._get_model_inputs()
        if not inputs:
            return {}

        cap = inputs["growth_cap"]
        base_growth = self._calculate_best_growth(cap)
        base_wacc = self._calculate_wacc()

        scenarios = {
            "Conservative": {
                "growth": min(base_growth * 0.85, cap * 0.85),
                "discount": base_wacc + 0.015,
                "terminal": self.terminal_growth - 0.010
            },
            "Base": {
                "growth": base_growth,
                "discount": base_wacc,
                "terminal": self.terminal_growth
            },
            "Optimistic": {
                "growth": min(base_growth * 1.15, cap * 1.15),
                "discount": base_wacc - 0.015,
                "terminal": self.terminal_growth + 0.010
            }
        }

        results = {}
        for name, params in scenarios.items():
            val = self._compute_dcf(inputs, params["growth"], params["discount"], params["terminal"])
            results[name] = {
                "value": round(val, 2),
                "growth_used": round(params["growth"] * 100, 2),
                "discount_used": round(params["discount"] * 100, 2)
            }

        return {
            "scenarios": results,
            "model_type": inputs["model_type"]
        }

    def calculate_relative_value(self):
        """Calculates fair value per share based on comparable sector multiples."""
        if not self.data_engine.has_data:
            return None

        sector = self.data_engine.info.get("sector", "").lower()
        multiples = self.SECTOR_MULTIPLES.get(sector, self.DEFAULT_MULTIPLES)

        eps = self.data_engine.info.get("trailingEps", 0) or 0
        current_price = self.data_engine.info.get("currentPrice", 0) or 0
        shares = self.data_engine.info.get("sharesOutstanding", 0) or 0
        book_value = self.data_engine.info.get("bookValue", 0) or 0

        ebit = self.data_engine.get_financials_safe(self.data_engine.financials, "Operating Income", 0)
        depr = abs(self.data_engine.get_financials_safe(self.data_engine.cashflow, "Depreciation And Amortization", 0))
        ebitda = (ebit or 0) + depr

        debt = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Total Debt", 0)
        cash = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Cash And Cash Equivalents", 0)
        net_debt = debt - cash

        values = []
        methods_used = []

        if eps > 0 and multiples.get("pe"):
            sector_pe = multiples["pe"]
            pe_fair_value = eps * sector_pe
            values.append(pe_fair_value)
            methods_used.append(f"P/E: {sector_pe:.1f}x → ₹{pe_fair_value:.0f}")

        if ebitda > 0 and shares > 0 and multiples.get("ev_ebitda"):
            sector_ev_ebitda = multiples["ev_ebitda"]
            fair_ev = ebitda * sector_ev_ebitda
            fair_equity = fair_ev - net_debt
            ev_fair_value = fair_equity / shares
            if ev_fair_value > 0:
                values.append(ev_fair_value)
                methods_used.append(f"EV/EBITDA: {sector_ev_ebitda}x → ₹{ev_fair_value:.0f}")

        if book_value > 0 and multiples.get("pb") and "financial" in sector:
            sector_pb = multiples["pb"]
            pb_fair_value = book_value * sector_pb
            values.append(pb_fair_value)
            methods_used.append(f"P/B: {sector_pb}x → ₹{pb_fair_value:.0f}")

        if not values:
            return None

        rel_val = sum(values) / len(values)

        return {
            "value": round(rel_val, 2),
            "methods": methods_used,
            "sector": sector.title() if sector else "General",
            "current_price": current_price,
            "upside": round((rel_val - current_price) / current_price * 100, 1) if current_price > 0 else 0
        }

    def get_combined_intrinsic_value(self):
        """
        AlphaSpread-style blended valuation with HoldCo Arbitrage Discount integration:
        Intrinsic Value = (DCF Value × 40%) + (Relative Value × 60%)
        """
        dcf_data = self.get_valuation_scenarios()
        dcf_val = 0
        if dcf_data and "scenarios" in dcf_data:
            dcf_val = dcf_data["scenarios"].get("Base", {}).get("value", 0)

        rel_data = self.calculate_relative_value()
        rel_val = rel_data.get("value", 0) if rel_data else 0
        current_price = self.data_engine.info.get("currentPrice", 0) or 0

        if dcf_val > 0 and rel_val > 0:
            combined = (dcf_val * 0.40) + (rel_val * 0.60)
            weighting = "DCF 40% + Relative 60%"
        elif dcf_val > 0:
            combined = dcf_val
            weighting = "DCF 100%"
        elif rel_val > 0:
            combined = rel_val
            weighting = "Relative 100%"
        else:
            return {"combined_value": current_price, "margin_of_safety": 0, "verdict": "FAIRLY VALUED"}

        # -------------------------------------------------------------
        # HOLDING COMPANY (HOLDCO) DISCOUNT ADJUSTMENT
        # -------------------------------------------------------------
        gross_asset_val = combined
        is_holdco = False
        holdco_discount_pct = 0.0

        try:
            from src.anomaly_detector import AnomalyDetector
            ad = AnomalyDetector(self.ticker)
            situations = ad.detect_special_situations()
            if situations.get("situation_type") == "HOLDING_COMPANY":
                is_holdco = True
                h_adj = situations.get("holdco_adjustment", {})
                factor = h_adj.get("discount_factor", 0.45)
                holdco_discount_pct = h_adj.get("discount_percentage", 55.0)
                combined = gross_asset_val * factor
                weighting = f"HoldCo Adjusted ({holdco_discount_pct:.0f}% Discount) | Gross Asset Value: ₹{gross_asset_val:,.0f}"
        except Exception:
            pass

        margin_of_safety = 0
        verdict = "HOLD"

        if combined > 0 and current_price > 0:
            if current_price < combined:
                margin_of_safety = ((combined - current_price) / combined) * 100
                verdict = "BUY (Undervalued)" if margin_of_safety >= 20 else "HOLD (Fair Value)"
            else:
                premium = ((current_price - combined) / combined) * 100
                verdict = "SELL (Overvalued)" if premium >= 25 else "HOLD (Premium)"

        return {
            "combined_value": round(combined, 2),
            "gross_asset_value": round(gross_asset_val, 2) if is_holdco else None,
            "is_holdco": is_holdco,
            "holdco_discount_pct": holdco_discount_pct if is_holdco else 0.0,
            "dcf_value": round(dcf_val, 2) if dcf_val else None,
            "relative_value": round(rel_val, 2) if rel_val else None,
            "weighting": weighting,
            "current_price": current_price,
            "margin_of_safety": round(margin_of_safety, 1),
            "verdict": verdict,
            "relative_methods": rel_data.get("methods", []) if rel_data else [],
            "model_type": dcf_data.get("model_type", "Standard") if dcf_data else "Relative"
        }

    def get_valuation_verdict(self):
        """Unified valuation summary returning both Combined and DCF Scenarios."""
        combined = self.get_combined_intrinsic_value()
        dcf_scenarios = self.get_valuation_scenarios()

        current_price = self.data_engine.info.get("currentPrice", 0) or 0
        prev_close = self.data_engine.info.get("previousClose", 0) or 0
        change = current_price - prev_close if (prev_close and current_price) else 0
        change_p = (change / prev_close) * 100 if prev_close else 0

        return {
            "current_price": current_price,
            "price_change": round(change, 2),
            "price_change_percent": round(change_p, 2),
            "intrinsic_value": combined.get("combined_value", 0),
            "scenarios": dcf_scenarios.get("scenarios", {}),
            "model_type": dcf_scenarios.get("model_type", "Standard"),
            "margin_of_safety": combined.get("margin_of_safety", 0),
            "verdict": combined.get("verdict", "HOLD"),
            "combined": combined
        }
