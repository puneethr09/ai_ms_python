html_path = "/home/puneeth/repo/stock_fundamental/templates/results.html"

with open(html_path, "r") as f:
    content = f.read()

target = """    <!-- ============================================================ -->
    <!-- DORSEY SCORECARD - THE VERDICT -->
    <!-- ============================================================ -->"""

ai_card = """    <!-- ============================================================ -->
    <!-- EDGE AI AUTONOMOUS VALUATION CARD (Raspberry Pi 5 ARM NEON) -->
    <!-- ============================================================ -->
    {% if ai_report %}
    <div class="card mb-4 shadow-sm" style="border: 2px solid #6366f1; border-radius: 12px; background: linear-gradient(145deg, #1e1e2e, #181825); color: #cdd6f4;">
        <div class="card-header d-flex justify-content-between align-items-center py-3" style="background: rgba(99, 102, 241, 0.15); border-bottom: 1px solid rgba(99, 102, 241, 0.3);">
            <div class="d-flex align-items-center">
                <span class="mr-2" style="font-size: 1.3rem;">🧠</span>
                <strong style="color: #a5b4fc; font-size: 1.1rem; letter-spacing: 0.5px;">EDGE AI FUNDAMENTAL INTELLIGENCE</strong>
                <span class="badge badge-info ml-2" style="background-color: #4f46e5; font-size: 0.75rem;">ARM Cortex-A76 Local LLM</span>
            </div>
            <div>
                <span class="badge badge-pill" style="font-size: 0.95rem; padding: 6px 14px; background: #6366f1; color: white;">
                    AI Score: {{ ai_report.ai_score }}/10
                </span>
            </div>
        </div>
        <div class="card-body py-3">
            <div class="mb-3">
                <div class="text-muted small text-uppercase" style="color: #94a3b8 !important; font-weight: 600; letter-spacing: 0.5px;">💡 AI Valuation Verdict</div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #f8fafc; margin-top: 2px;">{{ ai_report.ai_verdict }}</div>
            </div>
            <div class="row">
                <div class="col-md-6 mb-2 mb-md-0">
                    <div class="p-3" style="background: rgba(255, 255, 255, 0.03); border-radius: 8px; border-left: 3px solid #10b981; height: 100%;">
                        <div style="color: #34d399; font-weight: 700; font-size: 0.9rem;">🏰 Competitive Moat & Pricing Power</div>
                        <div class="small mt-1" style="color: #cbd5e1; line-height: 1.4;">{{ ai_report.moat_analysis }}</div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="p-3" style="background: rgba(255, 255, 255, 0.03); border-radius: 8px; border-left: 3px solid #f43f5e; height: 100%;">
                        <div style="color: #fb7185; font-weight: 700; font-size: 0.9rem;">⚠️ Critical Debt & Market Risks</div>
                        <div class="small mt-1" style="color: #cbd5e1; line-height: 1.4;">{{ ai_report.top_risks }}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    {% endif %}

    <!-- ============================================================ -->
    <!-- DORSEY SCORECARD - THE VERDICT -->
    <!-- ============================================================ -->"""

if target in content:
    with open(html_path, "w") as f:
        f.write(content.replace(target, ai_card, 1))
    print("✅ results.html successfully upgraded with Edge AI card UI!")
else:
    print("⚠️ Target not found in results.html")
