html_path = "/home/puneeth/repo/stock_fundamental/templates/results.html"

with open(html_path, "r") as f:
    content = f.read()

# Locate where the AI card starts and ends
start_marker = "<!-- ============================================================ -->\n    <!-- EDGE AI"
end_marker = "<!-- ============================================================ -->\n    <!-- DORSEY SCORECARD - THE VERDICT -->"

if start_marker in content:
    pre = content.split(start_marker)[0]
    post = content.split(end_marker)[1]
else:
    pre = content.split(end_marker)[0]
    post = content.split(end_marker)[1]

new_ai_card = """    <!-- ============================================================ -->
    <!-- EDGE AI AUTONOMOUS VALUATION CARD (Raspberry Pi 5 ARM NEON) -->
    <!-- ============================================================ -->
    {% if ai_report and ai_report.ai_verdict %}
    <div class="card mb-4 shadow-lg" style="border: 2px solid #6366f1; border-radius: 14px; background: linear-gradient(145deg, #181825, #11111b); color: #cdd6f4; overflow: hidden;">
        <div class="card-header d-flex justify-content-between align-items-center py-3 px-4" style="background: rgba(99, 102, 241, 0.12); border-bottom: 1px solid rgba(99, 102, 241, 0.25);">
            <div class="d-flex align-items-center">
                <span class="mr-3" style="font-size: 1.5rem;">🧠</span>
                <div>
                    <strong style="color: #c7d2fe; font-size: 1.15rem; letter-spacing: 0.5px; display: block;">INSTITUTIONAL EDGE AI INTELLIGENCE</strong>
                    <small style="color: #818cf8; font-size: 0.78rem;">Local Cortex-A76 ARM NEON Engine &bull; Private &amp; Offline</small>
                </div>
            </div>
            <div>
                {% if ai_report.ai_score %}
                    {% if ai_report.ai_score >= 8 %}
                        <span class="badge badge-pill" style="font-size: 0.95rem; padding: 7px 16px; background: #10b981; color: #ffffff; font-weight: 700; box-shadow: 0 0 12px rgba(16, 185, 129, 0.4);">
                            <i class="fas fa-shield-alt mr-1"></i> AI Score: {{ ai_report.ai_score }}/10 &bull; High Conviction
                        </span>
                    {% elif ai_report.ai_score >= 5 %}
                        <span class="badge badge-pill" style="font-size: 0.95rem; padding: 7px 16px; background: #f59e0b; color: #ffffff; font-weight: 700; box-shadow: 0 0 12px rgba(245, 158, 11, 0.4);">
                            <i class="fas fa-balance-scale mr-1"></i> AI Score: {{ ai_report.ai_score }}/10 &bull; Balanced
                        </span>
                    {% else %}
                        <span class="badge badge-pill" style="font-size: 0.95rem; padding: 7px 16px; background: #f43f5e; color: #ffffff; font-weight: 700; box-shadow: 0 0 12px rgba(244, 63, 94, 0.4);">
                            <i class="fas fa-exclamation-triangle mr-1"></i> AI Score: {{ ai_report.ai_score }}/10 &bull; Caution
                        </span>
                    {% endif %}
                {% endif %}
            </div>
        </div>
        <div class="card-body py-4 px-4">
            <!-- Verdict -->
            <div class="mb-4 p-3" style="background: rgba(99, 102, 241, 0.06); border-radius: 10px; border-left: 4px solid #6366f1;">
                <div class="text-uppercase small mb-1" style="color: #a5b4fc; font-weight: 700; letter-spacing: 0.8px; font-size: 0.8rem;">
                    <i class="fas fa-chart-line mr-1"></i> Executive Valuation Verdict &amp; Thesis
                </div>
                <div style="font-size: 1.05rem; font-weight: 500; color: #f8fafc; line-height: 1.55;">
                    {{ ai_report.ai_verdict }}
                </div>
            </div>

            <!-- Two Columns for Moat and Risks -->
            <div class="row">
                <div class="col-md-6 mb-3 mb-md-0">
                    <div class="p-3" style="background: rgba(16, 185, 129, 0.05); border-radius: 10px; border-left: 4px solid #10b981; height: 100%;">
                        <div class="text-uppercase small mb-2" style="color: #34d399; font-weight: 700; letter-spacing: 0.8px; font-size: 0.8rem;">
                            <i class="fas fa-chess-rook mr-1"></i> Economic Moat &amp; Pricing Power
                        </div>
                        <div style="color: #e2e8f0; line-height: 1.5; font-size: 0.93rem;">
                            {{ ai_report.moat_analysis }}
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="p-3" style="background: rgba(244, 63, 94, 0.05); border-radius: 10px; border-left: 4px solid #f43f5e; height: 100%;">
                        <div class="text-uppercase small mb-2" style="color: #fb7185; font-weight: 700; letter-spacing: 0.8px; font-size: 0.8rem;">
                            <i class="fas fa-exclamation-circle mr-1"></i> Critical Risks &amp; Downside Catalysts
                        </div>
                        <div style="color: #e2e8f0; line-height: 1.5; font-size: 0.93rem;">
                            {{ ai_report.top_risks | replace('\n', '<br>') | safe }}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    {% endif %}

    <!-- ============================================================ -->
    <!-- DORSEY SCORECARD - THE VERDICT -->
    <!-- ============================================================ -->"""

full_updated = pre + new_ai_card + post
with open(html_path, "w") as f:
    f.write(full_updated)

print("✅ templates/results.html successfully upgraded with institutional UI!")
