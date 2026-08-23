import sys

def main():
    target_path = sys.argv[1] if len(sys.argv) > 1 else '/home/puneeth/repo/stock_fundamental/templates/results.html'
    with open(target_path, 'r') as f:
        text = f.read()

    start_marker = '<!-- ============================================================ -->\n    <!-- MASTER COMPOSITE VERDICT HERO (ONE UNIFIED DEFINITIVE SCORE) -->'
    end_marker = '<!-- ============================================================ -->\n    <!-- MISTAKE WARNINGS -->'

    idx1 = text.find(start_marker)
    idx2 = text.find(end_marker)

    if idx1 == -1 or idx2 == -1:
        print(f"Markers not found! idx1={idx1}, idx2={idx2}")
        return

    unified_top = '''<!-- ============================================================ -->
    <!-- MASTER COMPOSITE VERDICT HERO (ONE UNIFIED DEFINITIVE SCORE) -->
    <!-- ============================================================ -->
    {% if scorecard %}
    <div class="card mb-4 shadow-lg animate-fade-in {% if scorecard.recommendation == 'STRONG BUY' %}border-glow-green{% elif 'AVOID' in scorecard.recommendation %}border-glow-red{% else %}glow-blue{% endif %}" style="border-width: 2px; border-radius: 14px; background: linear-gradient(145deg, #181825, #11111b);">
        <div class="card-header d-flex justify-content-between align-items-center py-3 px-4" style="background: rgba(255, 255, 255, 0.03); border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
            <div>
                <span class="badge badge-pill badge-primary px-3 py-1 mr-2" style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.5px;">
                    <i class="fas fa-crown text-warning mr-1"></i> MASTER COMPOSITE SCORE
                </span>
                <span class="text-muted" style="font-size: 0.82rem;">Unified Dorsey + Graham + Forensic Quant Synthesis</span>
            </div>
            {% if momentum_52w and momentum_52w.status != 'N/A' %}
            <div>
                <span class="badge badge-pill {% if 'Low' in momentum_52w.status %}badge-success{% elif 'High' in momentum_52w.status %}badge-info{% else %}badge-secondary{% endif %}" style="font-size: 0.80rem;">
                    <i class="fas fa-chart-area mr-1"></i> 52W Range: {{ momentum_52w.range_position_pct }}% ({{ momentum_52w.status }})
                </span>
            </div>
            {% endif %}
        </div>
        <div class="card-body py-4 px-4">
            <div class="row align-items-center">
                <!-- Single Master Score Circle -->
                <div class="col-lg-3 col-md-4 text-center mb-3 mb-md-0 border-right" style="border-color: rgba(255,255,255,0.08) !important;">
                    <div class="score-circle {% if scorecard.total_score >= 65 %}{% elif scorecard.total_score >= 50 %}moderate{% else %}poor{% endif %} mx-auto mb-2" style="width: 110px; height: 110px;">
                        <div class="text-center">
                            <span style="font-size: 2.3rem; font-weight: 800; line-height: 1;">{{ scorecard.total_score|round|int }}</span>
                            <span class="d-block text-muted" style="font-size: 0.8rem; font-weight: 600;">/ 100</span>
                        </div>
                    </div>
                    <h3 class="mb-0 {% if scorecard.recommendation == 'STRONG BUY' %}text-success{% elif scorecard.recommendation == 'BUY' %}text-info{% elif 'AVOID' in scorecard.recommendation %}text-danger{% else %}text-warning{% endif %}" style="font-size: 1.4rem; font-weight: 800; letter-spacing: 0.5px;">
                        {{ scorecard.recommendation }}
                    </h3>
                    <small class="text-muted" style="font-size: 0.8rem;">Confidence: <span class="text-white">{{ scorecard.confidence }}</span></small>
                </div>

                <!-- 4 Master Pillars Breakdown -->
                <div class="col-lg-9 col-md-8 pl-md-4">
                    <div class="row">
                        <!-- Pillar 1: Valuation -->
                        <div class="col-6 col-lg-3 mb-2">
                            <div class="p-2 rounded text-center" style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);">
                                <div class="small text-muted font-weight-bold mb-1"><i class="fas fa-tag text-success mr-1"></i>Valuation</div>
                                <div class="font-weight-bold" style="font-size: 1.15rem;">{{ scorecard.scores.valuation.score|round|int }}<span class="text-muted" style="font-size: 0.75rem;"> / {{ scorecard.scores.valuation.max }}</span></div>
                                <small class="{% if 'BUY' in scorecard.scores.valuation.assessment %}text-success{% elif 'SELL' in scorecard.scores.valuation.assessment %}text-danger{% else %}text-warning{% endif %}" style="font-size: 0.75rem; font-weight: 600;">{{ scorecard.scores.valuation.assessment }}</small>
                            </div>
                        </div>
                        <!-- Pillar 2: Moat -->
                        <div class="col-6 col-lg-3 mb-2">
                            <div class="p-2 rounded text-center" style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);">
                                <div class="small text-muted font-weight-bold mb-1"><i class="fas fa-shield-alt text-info mr-1"></i>Moat Power</div>
                                <div class="font-weight-bold" style="font-size: 1.15rem;">{{ scorecard.scores.moat.score|round|int }}<span class="text-muted" style="font-size: 0.75rem;"> / {{ scorecard.scores.moat.max }}</span></div>
                                <small class="text-info" style="font-size: 0.75rem; font-weight: 600;">{{ scorecard.scores.moat.rating }}</small>
                            </div>
                        </div>
                        <!-- Pillar 3: Health -->
                        <div class="col-6 col-lg-3 mb-2">
                            <div class="p-2 rounded text-center" style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);">
                                <div class="small text-muted font-weight-bold mb-1"><i class="fas fa-heartbeat text-danger mr-1"></i>Health</div>
                                <div class="font-weight-bold" style="font-size: 1.15rem;">{{ scorecard.scores.financial_health.score|round|int }}<span class="text-muted" style="font-size: 0.75rem;"> / {{ scorecard.scores.financial_health.max }}</span></div>
                                <small class="{% if scorecard.scores.financial_health.rating == 'ROBUST' or scorecard.scores.financial_health.rating == 'HEALTHY' %}text-success{% elif scorecard.scores.financial_health.rating == 'RISKY' %}text-danger{% else %}text-warning{% endif %}" style="font-size: 0.75rem; font-weight: 600;">{{ scorecard.scores.financial_health.rating }}</small>
                            </div>
                        </div>
                        <!-- Pillar 4: 10-Min Screen -->
                        <div class="col-6 col-lg-3 mb-2">
                            <div class="p-2 rounded text-center" style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);">
                                <div class="small text-muted font-weight-bold mb-1"><i class="fas fa-clock text-warning mr-1"></i>10-Min Screen</div>
                                <div class="font-weight-bold" style="font-size: 1.15rem;">{{ scorecard.scores.ten_minute_test.score|round|int }}<span class="text-muted" style="font-size: 0.75rem;"> / {{ scorecard.scores.ten_minute_test.max }}</span></div>
                                <small class="{% if scorecard.scores.ten_minute_test.verdict == 'PASS' %}text-success{% else %}text-warning{% endif %}" style="font-size: 0.75rem; font-weight: 600;">{{ scorecard.scores.ten_minute_test.verdict }}</small>
                            </div>
                        </div>
                    </div>
                    <!-- Master Summary Sentence -->
                    <div class="mt-3 p-2 rounded" style="background: rgba(99, 102, 241, 0.08); border-left: 3px solid #6366f1;">
                        <span style="font-size: 0.88rem; color: #e2e8f0; line-height: 1.4;">
                            <i class="fas fa-info-circle text-primary mr-1"></i> {{ scorecard.summary }}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    {% endif %}

    <!-- ============================================================ -->
    <!-- SPECIAL SITUATION & STRUCTURAL ANOMALY CONTEXT -->
    <!-- ============================================================ -->
    {% if special_situation and special_situation.has_special_situation %}
    <div class="alert alert-{{ special_situation.badge_color }} mb-4 p-3 shadow-sm animate-fade-in" style="border-radius: 10px; border-left: 4px solid #f59e0b; background: rgba(245, 158, 11, 0.08); color: #fde68a;">
        <div class="d-flex align-items-center mb-1">
            <strong style="font-size: 0.95rem; letter-spacing: 0.5px;"><i class="fas fa-lightbulb mr-2"></i>{{ special_situation.badge_title }}</strong>
        </div>
        <div style="font-size: 0.88rem; line-height: 1.45; color: #f3f4f6;">
            {{ special_situation.description }}
            {% if special_situation.holdco_adjustment and valuation.combined.gross_asset_value %}
            <div class="mt-2 pt-2 border-top" style="border-color: rgba(255,255,255,0.12) !important;">
                <strong>Gross Asset Value (GAV):</strong> <span class="text-white">₹{{ valuation.combined.gross_asset_value }}</span> &bull; 
                <strong>HoldCo Market Target ({{ special_situation.holdco_adjustment.discount_percentage|round|int }}% Disc):</strong> <span class="text-warning font-weight-bold">₹{{ valuation.combined.combined_value }}</span>
            </div>
            {% endif %}
        </div>
    </div>
    {% endif %}

    <!-- ============================================================ -->
    <!-- CFA-GRADE EDGE AI QUALITATIVE RESEARCH SYNTHESIS -->
    <!-- ============================================================ -->
    {% if ai_report and ai_report.ai_verdict %}
    <div class="card mb-4 shadow-lg" style="border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 14px; background: linear-gradient(145deg, #181825, #11111b); color: #cdd6f4; overflow: hidden;">
        <div class="card-header d-flex justify-content-between align-items-center py-2 px-4" style="background: rgba(99, 102, 241, 0.08); border-bottom: 1px solid rgba(99, 102, 241, 0.18);">
            <div class="d-flex align-items-center">
                <span class="mr-2" style="font-size: 1.25rem;">🧠</span>
                <div>
                    <strong style="color: #c7d2fe; font-size: 0.98rem; letter-spacing: 0.5px;">AUTONOMOUS CFA-GRADE EQUITY RESEARCH SYNTHESIS</strong>
                </div>
            </div>
            <div>
                <span class="badge badge-pill" style="font-size: 0.76rem; padding: 4px 10px; background: rgba(99, 102, 241, 0.20); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.35);">
                    <i class="fas fa-microchip mr-1"></i> Cortex-A76 Local Engine &bull; Private
                </span>
            </div>
        </div>
        <div class="card-body py-3 px-4">
            <!-- Verdict -->
            <div class="mb-3 p-3" style="background: rgba(99, 102, 241, 0.05); border-radius: 10px; border-left: 3px solid #6366f1;">
                <div class="text-uppercase small mb-1" style="color: #a5b4fc; font-weight: 700; letter-spacing: 0.8px; font-size: 0.78rem;">
                    <i class="fas fa-chart-line mr-1"></i> Executive Valuation Verdict &amp; Thesis
                </div>
                <div style="font-size: 0.95rem; font-weight: 500; color: #f8fafc; line-height: 1.5;">
                    {{ ai_report.ai_verdict }}
                </div>
            </div>

            <!-- Two Columns for Moat and Risks -->
            <div class="row">
                <div class="col-md-6 mb-2 mb-md-0">
                    <div class="p-3" style="background: rgba(16, 185, 129, 0.04); border-radius: 10px; border-left: 3px solid #10b981; height: 100%;">
                        <div class="text-uppercase small mb-1" style="color: #34d399; font-weight: 700; letter-spacing: 0.8px; font-size: 0.78rem;">
                            <i class="fas fa-chess-rook mr-1"></i> Economic Moat &amp; Pricing Power
                        </div>
                        <div style="color: #e2e8f0; line-height: 1.45; font-size: 0.88rem;">
                            {{ ai_report.moat_analysis }}
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="p-3" style="background: rgba(244, 63, 94, 0.04); border-radius: 10px; border-left: 3px solid #f43f5e; height: 100%;">
                        <div class="text-uppercase small mb-1" style="color: #fb7185; font-weight: 700; letter-spacing: 0.8px; font-size: 0.78rem;">
                            <i class="fas fa-exclamation-circle mr-1"></i> Critical Risks &amp; Downside Catalysts
                        </div>
                        <div style="color: #e2e8f0; line-height: 1.45; font-size: 0.88rem;">
                            {{ ai_report.top_risks | replace('\n', '<br>') | safe }}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    {% endif %}

    '''

    new_text = text[:idx1] + unified_top + text[idx2:]

    with open(target_path, 'w') as f:
        f.write(new_text)

    print("Successfully injected Special Situation banner and unified top!")

if __name__ == '__main__':
    main()
