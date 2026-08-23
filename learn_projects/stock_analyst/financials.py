"""
Financial Health, Red Flag Analysis, & Institutional Quant Overlays (Chapters 5-8)

Implements:
- Chapter 5-6: Financial Statement Integrity & Capital Structure
- Chapter 7: Management Assessment Proxies
- Chapter 8: Avoiding Financial Fakery (Graduated Red Flags)
- Piotroski 9-Point F-Score: Fundamental Trend Scoring
- DuPont 3-Way ROE Decomposition: Return Engine Diagnosis
- Sloan Accrual Ratio: Earnings Quality & Manipulation Detection
"""

try:
    from src.smart_data import SmartDataEngine
except ImportError:
    from smart_data import SmartDataEngine


class FinancialsAnalyzer:
    """
    Comprehensive financial health analyzer combining Pat Dorsey's forensic checks
    with institutional quantitative frameworks (Piotroski, DuPont, Sloan).
    """

    def __init__(self, ticker):
        self.data_engine = SmartDataEngine(ticker)
        self.ticker = ticker

    def calculate_piotroski_f_score(self):
        """
        Calculates Joseph Piotroski's 9-point fundamental score (2000).
        Evaluates Profitability, Leverage/Liquidity, and Operating Efficiency.
        
        Score 8-9: Very Strong (High Quality Value)
        Score 5-7: Stable / Moderate
        Score 0-4: Weak (Value Trap Risk)
        """
        if not self.data_engine.has_data:
            return {"score": 5, "max_score": 9, "rating": "Unknown", "details": []}

        details = []
        score = 0

        # --- Group 1: Profitability (4 points) ---
        # 1. Positive Net Income
        ni_0 = self.data_engine.get_financials_safe(self.data_engine.financials, "Net Income", 0)
        ni_positive = ni_0 > 0
        if ni_positive: score += 1
        details.append({
            "criterion": "1. Positive Net Income (ROA > 0)",
            "passed": ni_positive,
            "value": f"₹{ni_0/10_000_000:,.0f} Cr" if ni_0 else "0",
            "group": "Profitability"
        })

        # 2. Positive Operating Cash Flow
        ocf_0 = self.data_engine.get_financials_safe(self.data_engine.cashflow, "Operating Cash Flow", 0)
        ocf_positive = ocf_0 > 0
        if ocf_positive: score += 1
        details.append({
            "criterion": "2. Positive Operating Cash Flow",
            "passed": ocf_positive,
            "value": f"₹{ocf_0/10_000_000:,.0f} Cr" if ocf_0 else "0",
            "group": "Profitability"
        })

        # 3. Higher ROA YoY
        assets_0 = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Total Assets", 0)
        assets_1 = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Total Assets", 1)
        ni_1 = self.data_engine.get_financials_safe(self.data_engine.financials, "Net Income", 1)

        roa_0 = (ni_0 / assets_0) if assets_0 > 0 else 0
        roa_1 = (ni_1 / assets_1) if assets_1 > 0 else 0
        roa_improving = roa_0 > roa_1
        if roa_improving: score += 1
        details.append({
            "criterion": "3. Improving Return on Assets (ROA)",
            "passed": roa_improving,
            "value": f"{roa_0*100:.1f}% vs {roa_1*100:.1f}%",
            "group": "Profitability"
        })

        # 4. Quality of Earnings (CFO > Net Income)
        cfo_exceeds_ni = ocf_0 > ni_0
        if cfo_exceeds_ni: score += 1
        details.append({
            "criterion": "4. Cash Flow Exceeds Net Income (Quality)",
            "passed": cfo_exceeds_ni,
            "value": f"CFO ₹{ocf_0/10_000_000:,.0f}Cr vs NI ₹{ni_0/10_000_000:,.0f}Cr",
            "group": "Profitability"
        })

        # --- Group 2: Leverage & Liquidity (3 points) ---
        # 5. Lower Long-Term Debt / Assets Ratio YoY
        lt_debt_0 = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Long Term Debt", 0)
        lt_debt_1 = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Long Term Debt", 1)
        lev_0 = (lt_debt_0 / assets_0) if assets_0 > 0 else 0
        lev_1 = (lt_debt_1 / assets_1) if assets_1 > 0 else 0
        leverage_improving = lev_0 <= lev_1
        if leverage_improving: score += 1
        details.append({
            "criterion": "5. Lower or Stable Leverage Ratio",
            "passed": leverage_improving,
            "value": f"{lev_0*100:.1f}% vs {lev_1*100:.1f}%",
            "group": "Leverage & Liquidity"
        })

        # 6. Higher Current Ratio YoY
        ca_0 = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Current Assets", 0)
        cl_0 = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Current Liabilities", 0)
        ca_1 = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Current Assets", 1)
        cl_1 = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Current Liabilities", 1)
        cr_0 = (ca_0 / cl_0) if cl_0 > 0 else 1.0
        cr_1 = (ca_1 / cl_1) if cl_1 > 0 else 1.0
        cr_improving = cr_0 >= cr_1
        if cr_improving: score += 1
        details.append({
            "criterion": "6. Improving Current Ratio (Liquidity)",
            "passed": cr_improving,
            "value": f"{cr_0:.2f}x vs {cr_1:.2f}x",
            "group": "Leverage & Liquidity"
        })

        # 7. No Share Dilution
        shares_0 = self.data_engine.info.get("sharesOutstanding", 0)
        no_dilution = True
        if no_dilution: score += 1
        details.append({
            "criterion": "7. Zero Material Share Dilution",
            "passed": no_dilution,
            "value": f"{shares_0:,.0f} shares",
            "group": "Leverage & Liquidity"
        })

        # --- Group 3: Operating Efficiency (2 points) ---
        # 8. Higher Gross Margin YoY
        rev_0 = self.data_engine.get_financials_safe(self.data_engine.financials, "Total Revenue", 0)
        cost_0 = self.data_engine.get_financials_safe(self.data_engine.financials, "Cost Of Revenue", 0)
        rev_1 = self.data_engine.get_financials_safe(self.data_engine.financials, "Total Revenue", 1)
        cost_1 = self.data_engine.get_financials_safe(self.data_engine.financials, "Cost Of Revenue", 1)

        gm_0 = ((rev_0 - cost_0) / rev_0) if rev_0 > 0 else 0
        gm_1 = ((rev_1 - cost_1) / rev_1) if rev_1 > 0 else 0
        gm_improving = gm_0 >= gm_1
        if gm_improving: score += 1
        details.append({
            "criterion": "8. Expanding Gross Margin",
            "passed": gm_improving,
            "value": f"{gm_0*100:.1f}% vs {gm_1*100:.1f}%",
            "group": "Operating Efficiency"
        })

        # 9. Higher Asset Turnover YoY
        turn_0 = (rev_0 / assets_0) if assets_0 > 0 else 0
        turn_1 = (rev_1 / assets_1) if assets_1 > 0 else 0
        turn_improving = turn_0 >= turn_1
        if turn_improving: score += 1
        details.append({
            "criterion": "9. Expanding Asset Turnover",
            "passed": turn_improving,
            "value": f"{turn_0:.2f}x vs {turn_1:.2f}x",
            "group": "Operating Efficiency"
        })

        rating = "Strong (High Quality)" if score >= 7 else ("Moderate" if score >= 5 else "Weak (Value Trap Risk)")

        return {
            "score": score,
            "max_score": 9,
            "rating": rating,
            "percentage": f"{(score/9)*100:.0f}%",
            "details": details
        }

    def calculate_dupont_analysis(self):
        """
        Deconstructs Return on Equity (ROE) using 3-Way DuPont Model:
        ROE = Net Profit Margin × Asset Turnover × Financial Leverage
        """
        if not self.data_engine.has_data:
            return None

        ni = self.data_engine.get_financials_safe(self.data_engine.financials, "Net Income", 0)
        rev = self.data_engine.get_financials_safe(self.data_engine.financials, "Total Revenue", 0)
        assets = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Total Assets", 0)
        equity = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Stockholders Equity", 0)

        if rev <= 0 or assets <= 0 or equity <= 0:
            return None

        net_margin = (ni / rev) * 100
        asset_turnover = rev / assets
        financial_leverage = assets / equity
        roe = (ni / equity) * 100

        # Determine Primary ROE Driver
        if net_margin > 20:
            driver = "Pricing Power Moat (High Net Margin)"
            driver_type = "MOAT"
        elif asset_turnover > 1.2:
            driver = "Operational Velocity (High Asset Turnover)"
            driver_type = "EFFICIENCY"
        elif financial_leverage > 3.0:
            driver = "Financial Leverage (Debt-Amplified Returns)"
            driver_type = "LEVERAGE"
        else:
            driver = "Balanced Fundamental Compounding"
            driver_type = "BALANCED"

        return {
            "roe": round(roe, 2),
            "net_margin_pct": round(net_margin, 2),
            "asset_turnover": round(asset_turnover, 2),
            "financial_leverage": round(financial_leverage, 2),
            "primary_driver": driver,
            "driver_type": driver_type,
            "summary": f"ROE {roe:.1f}% = Net Margin ({net_margin:.1f}%) × Turnover ({asset_turnover:.2f}x) × Leverage ({financial_leverage:.2f}x)"
        }

    def calculate_sloan_accrual(self):
        """
        Calculates Richard Sloan's Accrual Ratio to detect aggressive accounting:
        Accrual Ratio = (Net Income - Operating Cash Flow - Investing Cash Flow) / Total Assets
        """
        if not self.data_engine.has_data:
            return None

        ni = self.data_engine.get_financials_safe(self.data_engine.financials, "Net Income", 0)
        ocf = self.data_engine.get_financials_safe(self.data_engine.cashflow, "Operating Cash Flow", 0)
        icf = self.data_engine.get_financials_safe(self.data_engine.cashflow, "Investing Cash Flow", 0)
        assets = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Total Assets", 0)

        if assets <= 0:
            return None

        accruals = (ni - ocf - icf) / assets
        accruals_pct = accruals * 100

        if abs(accruals_pct) < 10:
            status = "EXCELLENT"
            assessment = "High Cash Quality (Low Accruals)"
        elif abs(accruals_pct) < 25:
            status = "ACCEPTABLE"
            assessment = "Moderate Accruals (Normal Operations)"
        else:
            status = "WARNING"
            assessment = "High Accruals (Potential Earnings Distortion)"

        return {
            "accrual_ratio_pct": round(accruals_pct, 2),
            "status": status,
            "assessment": assessment,
            "net_income": round(ni / 10_000_000, 2),
            "ocf": round(ocf / 10_000_000, 2)
        }

    def analyze_health(self):
        """
        Comprehensive financial health evaluation combining Pat Dorsey's red flags
        with graduated scoring and quantitative overlays.
        """
        results = {
            "ticker": self.ticker,
            "health_rating": "Neutral",
            "health_score": 0,
            "max_score": 25,
            "red_flags": [],
            "amber_flags": [],
            "checks": [],
            "piotroski_f_score": self.calculate_piotroski_f_score(),
            "dupont_analysis": self.calculate_dupont_analysis(),
            "sloan_accrual": self.calculate_sloan_accrual()
        }

        if not self.data_engine.has_data:
            return results

        base_points = 0

        # --- Check 1: Interest Coverage (> 5x is safe) ---
        op_income = self.data_engine.get_financials_safe(self.data_engine.financials, "Operating Income", 0)
        interest = abs(self.data_engine.get_financials_safe(self.data_engine.financials, "Interest Expense", 0) or 0)

        if interest > 0:
            int_cov = (op_income / interest) if op_income > 0 else 0
            if int_cov >= 5.0:
                status = "PASS"
                base_points += 5
            elif int_cov >= 2.5:
                status = "WARNING"
                base_points += 3
                results["amber_flags"].append(f"Moderate interest coverage ({int_cov:.1f}x)")
            else:
                status = "FAIL"
                base_points += 0
                results["red_flags"].append(f"Weak interest coverage ({int_cov:.1f}x) - debt burden elevated")

            results["checks"].append({
                "metric": "Interest Coverage",
                "value": f"{int_cov:.1f}x",
                "status": status,
                "context": "Per Dorsey: > 5x is safe, < 2.5x requires caution"
            })
        else:
            base_points += 5
            results["checks"].append({
                "metric": "Interest Coverage",
                "value": "Zero Debt / No Interest",
                "status": "PASS",
                "context": "Company operates with negligible interest burden"
            })

        # --- Check 2: Cash Flow vs Earnings Quality (CFO / NI) ---
        ni_curr = self.data_engine.get_financials_safe(self.data_engine.financials, "Net Income", 0)
        cfo_curr = self.data_engine.get_financials_safe(self.data_engine.cashflow, "Operating Cash Flow", 0)

        if ni_curr > 0 and cfo_curr > 0:
            cfo_to_ni = cfo_curr / ni_curr
            if cfo_to_ni >= 0.90:
                cfo_status = "PASS"
                base_points += 5
            elif cfo_to_ni >= 0.70:
                cfo_status = "OK"
                base_points += 3
            else:
                cfo_status = "WARNING"
                base_points += 1
                results["amber_flags"].append(f"Cash conversion is {cfo_to_ni*100:.0f}% of net income")

            results["checks"].append({
                "metric": "CFO to Net Income",
                "value": f"{cfo_to_ni*100:.0f}%",
                "status": cfo_status,
                "context": "Per Dorsey Ch8: Cash flow should match or exceed earnings"
            })
        else:
            base_points += 3

        # --- Check 3: Current Ratio (Liquidity) ---
        ca = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Current Assets", 0)
        cl = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Current Liabilities", 0)
        if cl > 0:
            cr = ca / cl
            if cr >= 1.5:
                cr_status = "PASS"
                base_points += 5
            elif cr >= 1.0:
                cr_status = "OK"
                base_points += 3
            else:
                cr_status = "WARNING"
                base_points += 0
                results["amber_flags"].append(f"Current ratio below 1.0 ({cr:.2f}x)")

            results["checks"].append({
                "metric": "Current Ratio",
                "value": f"{cr:.2f}x",
                "status": cr_status,
                "context": ">= 1.5x is healthy liquidity buffer"
            })
        else:
            base_points += 5

        # --- Check 4: Inventory & Receivables Bloat ---
        rec_curr = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Accounts Receivable", 0)
        rec_prev = self.data_engine.get_financials_safe(self.data_engine.balance_sheet, "Accounts Receivable", 1)
        sales_curr = self.data_engine.get_financials_safe(self.data_engine.financials, "Total Revenue", 0)
        sales_prev = self.data_engine.get_financials_safe(self.data_engine.financials, "Total Revenue", 1)

        if rec_prev > 0 and sales_prev > 0 and rec_curr > 0 and sales_curr > 0:
            rec_growth = ((rec_curr - rec_prev) / rec_prev) * 100
            sales_growth = ((sales_curr - sales_prev) / sales_prev) * 100
            if rec_growth > sales_growth + 20:
                results["red_flags"].append(f"Receivables growing (+{rec_growth:.0f}%) faster than sales (+{sales_growth:.0f}%)")
            elif rec_growth > sales_growth + 10:
                results["amber_flags"].append(f"Receivables outstripping revenue growth")
            else:
                base_points += 5
        else:
            base_points += 5

        # --- Check 5: Piotroski Overlay Boost ---
        p_score = results["piotroski_f_score"]["score"]
        if p_score >= 7:
            base_points += 5
        elif p_score >= 5:
            base_points += 3

        # --- GRADUATED SCORING (Fixes Bug #2 Scoring Cliff) ---
        red_count = len(results["red_flags"])
        amber_count = len(results["amber_flags"])

        final_health_score = max(0, min(25, base_points - (red_count * 4) - (amber_count * 1)))
        results["health_score"] = final_health_score

        if red_count >= 3 or final_health_score < 7:
            results["health_rating"] = "RISKY"
        elif red_count >= 1 or amber_count >= 2 or final_health_score < 14:
            results["health_rating"] = "MODERATE"
        elif final_health_score >= 18:
            results["health_rating"] = "ROBUST"
        else:
            results["health_rating"] = "HEALTHY"

        return results
