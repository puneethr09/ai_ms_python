import sys

def main():
    target_path = sys.argv[1] if len(sys.argv) > 1 else '/home/puneeth/repo/stock_fundamental/templates/results.html'
    with open(target_path, 'r') as f:
        text = f.read()

    target = '<h3 class="text-center mb-3 {% if financial_health.health_rating == \'ROBUST\' %}text-success{% elif financial_health.health_rating == \'RISKY\' %}text-danger{% else %}text-warning{% endif %}">\n                        {{ financial_health.health_rating }}\n                    </h3>'

    quant_card = '''<h3 class="text-center mb-3 {% if financial_health.health_rating == 'ROBUST' %}text-success{% elif financial_health.health_rating == 'RISKY' %}text-danger{% else %}text-warning{% endif %}">
                        {{ financial_health.health_rating }}
                    </h3>

                    <!-- Piotroski F-Score Badge -->
                    {% if piotroski_f_score and piotroski_f_score.score is defined %}
                    <div class="mb-3 p-2 rounded" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);">
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <span style="font-size: 0.82rem; font-weight: 600;"><i class="fas fa-chart-line text-info mr-1"></i>Piotroski F-Score:</span>
                            <span class="badge {% if piotroski_f_score.score >= 7 %}badge-success{% elif piotroski_f_score.score >= 5 %}badge-warning{% else %}badge-danger{% endif %}" style="font-size: 0.80rem;">
                                {{ piotroski_f_score.score }}/9 ({{ piotroski_f_score.rating }})
                            </span>
                        </div>
                        <div class="progress" style="height: 5px;">
                            <div class="progress-bar {% if piotroski_f_score.score >= 7 %}bg-success{% elif piotroski_f_score.score >= 5 %}bg-warning{% else %}bg-danger{% endif %}" role="progressbar" style="width: {{ (piotroski_f_score.score / 9) * 100 }}%;"></div>
                        </div>
                    </div>
                    {% endif %}

                    <!-- DuPont ROE Breakdown -->
                    {% if dupont_analysis and dupont_analysis.roe is defined %}
                    <div class="mb-3 p-2 rounded" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); font-size: 0.80rem;">
                        <div class="d-flex justify-content-between mb-1">
                            <span style="font-weight: 600;"><i class="fas fa-layer-group text-warning mr-1"></i>DuPont ROE Engine:</span>
                            <span class="badge badge-info">{{ dupont_analysis.driver_type }}</span>
                        </div>
                        <div class="text-muted" style="font-size: 0.78rem; line-height: 1.3;">
                            <strong>{{ dupont_analysis.primary_driver }}</strong><br>
                            Margin: <span class="text-white font-weight-bold">{{ dupont_analysis.net_margin_pct }}%</span> &bull; 
                            Turnover: <span class="text-white font-weight-bold">{{ dupont_analysis.asset_turnover }}x</span> &bull; 
                            Leverage: <span class="text-white font-weight-bold">{{ dupont_analysis.financial_leverage }}x</span>
                        </div>
                    </div>
                    {% endif %}

                    <!-- Sloan Accrual Quality -->
                    {% if sloan_accrual and sloan_accrual.status is defined %}
                    <div class="mb-3 p-2 rounded" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); font-size: 0.80rem;">
                        <div class="d-flex justify-content-between mb-1">
                            <span style="font-weight: 600;"><i class="fas fa-balance-scale text-primary mr-1"></i>Earnings Quality (Sloan):</span>
                            <span class="badge {% if sloan_accrual.status == 'EXCELLENT' %}badge-success{% elif sloan_accrual.status == 'ACCEPTABLE' %}badge-info{% else %}badge-danger{% endif %}">
                                {{ sloan_accrual.status }}
                            </span>
                        </div>
                        <small class="text-muted">{{ sloan_accrual.assessment }} (Accrual: {{ sloan_accrual.accrual_ratio_pct }}%)</small>
                    </div>
                    {% endif %}'''

    if target in text:
        text = text.replace(target, quant_card)
        with open(target_path, 'w') as f:
            f.write(text)
        print('Successfully injected quant cards into results.html!')
    else:
        print('Target string not found in results.html')

if __name__ == '__main__':
    main()
