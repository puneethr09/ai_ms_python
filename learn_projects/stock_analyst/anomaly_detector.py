"""
Autonomous Anomaly & Special Situation Engine

Identifies and adjusts for structural distortions in Indian equity analysis:
1. Holding Companies (HoldCo Discount Model)
2. Multi-Segment Conglomerates (SOTP Recognition)
3. Commodity & Cyclicals (Shiller CAPE Normalized Mid-Cycle Earnings)
4. Non-Operating One-Time Exceptional Gain Stripping
5. New-Age High-Growth Internet Platforms (P/S Multiple Model)
"""

try:
    from src.smart_data import SmartDataEngine
except ImportError:
    from smart_data import SmartDataEngine


class AnomalyDetector:
    """
    Forensic classifier for detecting special corporate structures,
    holding company arbitrage, cyclical earnings peaks, and accounting distortions.
    """

    # Verified Indian Holding Companies (HoldCos)
    HOLDING_COMPANIES = {
        "BBTC.NS": {"parent_group": "Wadia Group", "key_asset": "Britannia Industries (~50.5%)", "holdco_discount": 0.55},
        "BAJAJHLDNG.NS": {"parent_group": "Bajaj Group", "key_asset": "Bajaj Auto & Bajaj Finserv", "holdco_discount": 0.55},
        "MAHSCOOTER.NS": {"parent_group": "Bajaj Group", "key_asset": "Bajaj Finance & Finserv", "holdco_discount": 0.60},
        "TATAINVEST.NS": {"parent_group": "Tata Group", "key_asset": "Tata Sons & Tata Group Entities", "holdco_discount": 0.60},
        "GRASIM.NS": {"parent_group": "Aditya Birla", "key_asset": "UltraTech Cement (~57%) & AB Capital", "holdco_discount": 0.45},
        "PILANIINVS.NS": {"parent_group": "Birla Group", "key_asset": "Century Textiles & Grasim", "holdco_discount": 0.65},
        "JSWHL.NS": {"parent_group": "JSW Group", "key_asset": "JSW Steel & JSW Energy", "holdco_discount": 0.60},
        "SUNDARMHLD.NS": {"parent_group": "TVS Group", "key_asset": "Sundaram Finance Group", "holdco_discount": 0.55},
    }

    # Multi-Segment Conglomerates (Requiring SOTP context)
    CONGLOMERATES = {
        "RELIANCE.NS": {
            "name": "Reliance Industries",
            "segments": ["O2C Energy & Petrochemicals", "Jio Platforms (Digital/Telecom)", "Reliance Retail (Consumer)"],
            "sotp_note": "Energy multiples (12x P/E) understate Jio (30x) and Retail (35x) high-growth digital segments."
        },
        "LT.NS": {
            "name": "Larsen & Toubro",
            "segments": ["Core EPC Infrastructure", "LTIMindtree (IT Services)", "LTTS (Engineering Tech)", "L&T Finance"],
            "sotp_note": "Core infrastructure business is blended with high-multiple listed IT subsidiaries."
        },
        "ITC.NS": {
            "name": "ITC Limited",
            "segments": ["Cigarettes / Tobacco (High Margin Cash Cow)", "FMCG Others", "Paperboards & Packaging", "Agri Business"],
            "sotp_note": "Cash-generative cigarette monopoly subsidizes high-growth FMCG and packaging investments."
        },
        "TATAMOTORS.NS": {
            "name": "Tata Motors",
            "segments": ["Jaguar Land Rover (JLR Global Luxury)", "India Passenger Vehicles (EV)", "Commercial Vehicles"],
            "sotp_note": "JLR luxury cash flows have distinct currency and margin dynamics from domestic commercial vehicles."
        }
    }

    # Cyclical / Commodity Sectors
    CYCLICAL_SECTORS = ["basic materials", "metals", "mining", "steel", "chemical", "fertilizer", "shipping", "cement"]

    def __init__(self, ticker):
        self.ticker = ticker
        self.data_engine = SmartDataEngine(ticker)

    def detect_special_situations(self):
        """
        Executes comprehensive anomaly detection.
        Returns dictionary of active situations, adjustments, and warnings.
        """
        results = {
            "ticker": self.ticker,
            "has_special_situation": False,
            "situation_type": "STANDARD_OPERATING_COMPANY",
            "badge_title": None,
            "badge_color": "info",
            "description": None,
            "holdco_adjustment": None,
            "conglomerate_sotp": None,
            "cyclical_normalization": None,
            "exceptional_gain_flag": None,
            "new_age_platform": None
        }

        if not self.data_engine.has_data:
            return results

        norm_sym = self.ticker.upper().strip()
        if not norm_sym.endswith(".NS") and not norm_sym.endswith(".BO"):
            norm_sym += ".NS"

        sector = (self.data_engine.info.get("sector", "") or "").lower()
        industry = (self.data_engine.info.get("industry", "") or "").lower()

        # -------------------------------------------------------------
        # 1. HOLDING COMPANY (HOLDCO) DETECTION
        # -------------------------------------------------------------
        if norm_sym in self.HOLDING_COMPANIES or "holding" in (self.data_engine.info.get("longName", "").lower()):
            holdco_info = self.HOLDING_COMPANIES.get(norm_sym, {
                "parent_group": "Corporate Group",
                "key_asset": "Investments in operating subsidiaries",
                "holdco_discount": 0.55
            })
            discount_pct = holdco_info["holdco_discount"] * 100.0

            results["has_special_situation"] = True
            results["situation_type"] = "HOLDING_COMPANY"
            results["badge_title"] = f"🏢 HOLDING COMPANY ({discount_pct:.0f}% HoldCo Discount Applied)"
            results["badge_color"] = "warning"
            results["description"] = f"Holding company for {holdco_info['parent_group']}. Major underlying value lies in {holdco_info['key_asset']}. Market applies a standard {discount_pct:.0f}% HoldCo discount."
            results["holdco_adjustment"] = {
                "is_holdco": True,
                "discount_factor": 1.0 - holdco_info["holdco_discount"],
                "discount_percentage": discount_pct,
                "key_asset": holdco_info["key_asset"]
            }
            return results

        # -------------------------------------------------------------
        # 2. CONGLOMERATE SOTP DETECTION
        # -------------------------------------------------------------
        if norm_sym in self.CONGLOMERATES or "conglomerate" in industry:
            cong_info = self.CONGLOMERATES.get(norm_sym, {
                "name": self.data_engine.info.get("longName", norm_sym),
                "segments": ["Multi-segment operations"],
                "sotp_note": "Multi-segment operations warrant Sum-of-the-Parts (SOTP) evaluation."
            })
            results["has_special_situation"] = True
            results["situation_type"] = "CONGLOMERATE_SOTP"
            results["badge_title"] = "🌐 MULTI-SEGMENT CONGLOMERATE (SOTP Premium Profile)"
            results["badge_color"] = "primary"
            results["description"] = f"{cong_info['sotp_note']} Key pillars: {', '.join(cong_info['segments'])}."
            results["conglomerate_sotp"] = cong_info
            return results

        # -------------------------------------------------------------
        # 3. COMMODITY & CYCLICAL CAPE NORMALIZATION
        # -------------------------------------------------------------
        is_cyclical = any(c in sector or c in industry for c in self.CYCLICAL_SECTORS)
        if is_cyclical:
            # Check 3-year EBIT volatility
            ebit_0 = self.data_engine.get_financials_safe(self.data_engine.financials, "Operating Income", 0)
            ebit_1 = self.data_engine.get_financials_safe(self.data_engine.financials, "Operating Income", 1)
            ebit_2 = self.data_engine.get_financials_safe(self.data_engine.financials, "Operating Income", 2)
            valid_ebits = [e for e in [ebit_0, ebit_1, ebit_2] if e > 0]

            if len(valid_ebits) >= 2:
                avg_ebit = sum(valid_ebits) / len(valid_ebits)
                ebit_ratio = (ebit_0 / avg_ebit) if avg_ebit > 0 else 1.0

                if ebit_ratio > 1.40:
                    cycle_status = "Peak Earnings (P/E may look deceptively low)"
                    color = "danger"
                elif ebit_ratio < 0.70:
                    cycle_status = "Trough Earnings (Opportunity if balance sheet is solvent)"
                    color = "success"
                else:
                    cycle_status = "Mid-Cycle Earnings"
                    color = "info"

                results["has_special_situation"] = True
                results["situation_type"] = "CYCLICAL_COMMODITY"
                results["badge_title"] = f"🌋 COMMODITY CYCLICAL: {cycle_status}"
                results["badge_color"] = color
                results["description"] = f"Commodity sector company ({industry.title()}). Current EBIT is {ebit_ratio:.2f}x of 3-year normalized average (₹{avg_ebit/10_000_000:,.0f} Cr)."
                results["cyclical_normalization"] = {
                    "is_cyclical": True,
                    "cycle_status": cycle_status,
                    "normalized_ebit_cr": round(avg_ebit / 10_000_000, 2),
                    "current_vs_avg_ratio": round(ebit_ratio, 2)
                }
                return results

        # -------------------------------------------------------------
        # 4. ONE-TIME EXCEPTIONAL GAIN DETECTION
        # -------------------------------------------------------------
        pretax = self.data_engine.get_financials_safe(self.data_engine.financials, "Pretax Income", 0)
        other_inc = abs(self.data_engine.get_financials_safe(self.data_engine.financials, "Other Non Operating Income", 0) or 0)

        if pretax > 0 and (other_inc / pretax) > 0.25:
            results["has_special_situation"] = True
            results["situation_type"] = "EXCEPTIONAL_NON_OPERATING_INCOME"
            results["badge_title"] = "🧨 ONE-TIME EXTRAORDINARY GAIN DETECTED"
            results["badge_color"] = "warning"
            results["description"] = f"Non-operating income (₹{other_inc/10_000_000:,.0f} Cr) accounts for {(other_inc/pretax)*100:.0f}% of pretax profit. Valuation uses core operating earnings to avoid distortion."
            results["exceptional_gain_flag"] = {
                "other_income_cr": round(other_inc / 10_000_000, 2),
                "pct_of_pretax": round((other_inc / pretax) * 100, 1)
            }
            return results

        # -------------------------------------------------------------
        # 5. NEW-AGE INTERNET / HYPER-GROWTH PLATFORM
        # -------------------------------------------------------------
        eps = self.data_engine.info.get("trailingEps", 0) or 0
        rev_cagr = self.data_engine.calculate_multi_year_cagr(self.data_engine.financials, "Total Revenue", max_years=3)

        if eps <= 0 and rev_cagr and rev_cagr > 0.20:
            results["has_special_situation"] = True
            results["situation_type"] = "NEW_AGE_GROWTH_PLATFORM"
            results["badge_title"] = "🚀 NEW-AGE GROWTH PLATFORM (High Revenue Velocity)"
            results["badge_color"] = "primary"
            results["description"] = f"Fast-scaling digital platform with {rev_cagr*100:.1f}% revenue CAGR. Evaluated on Price-to-Sales (P/S) and operating leverage rather than GAAP P/E."
            results["new_age_platform"] = {
                "revenue_cagr_pct": round(rev_cagr * 100, 1),
                "is_scaling": True
            }
            return results

        return results
