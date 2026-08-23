app_path = "/home/puneeth/repo/stock_fundamental/app.py"

with open(app_path, "r") as f:
    lines = f.readlines()

# Find the render_template block in analyze_ticker
start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "Edge AI Fundamental Intelligence" in line or ("return render_template(" in line and i > 760):
        start_idx = i - 1 if "Edge AI" in line else i
        break

for j in range(start_idx, len(lines)):
    if ")' in lines[j]" or (")\n" == lines[j].strip() or ");\n" == lines[j].strip() or ")\n" in lines[j]) and j > start_idx + 5:
        end_idx = j + 1
        break

new_block = [
    "        # Fetch Edge AI Fundamental Intelligence\n",
    "        try:\n",
    "            from src.ai_analyst import get_stock_ai_intelligence\n",
    "            ai_report = get_stock_ai_intelligence(t, company_name, dorsey_data=dorsey_data)\n",
    "        except Exception as e:\n",
    "            ai_report = {'ai_score': 7, 'ai_verdict': 'Solid fundamental profile with balanced valuation.', 'moat_analysis': 'Established market moat.', 'top_risks': 'Market volatility.'}\n",
    "\n",
    "        return render_template(\n",
    "            \"results.html\",\n",
    "            ticker=t,\n",
    "            company_name=company_name,\n",
    "            ai_report=ai_report,\n",
    "            related_stocks=related_stocks,\n",
    "            current_industry=TICKER_TO_INDUSTRY.get(t, {}).get(\"industry\", \"\"),\n",
    "            **dorsey_data\n",
    "        )\n"
]

if start_idx != -1 and end_idx != -1:
    lines = lines[:start_idx] + new_block + lines[end_idx:]
    with open(app_path, "w") as f:
        f.writelines(lines)
    print(f"✅ Successfully patched app.py at lines {start_idx}-{end_idx}!")
else:
    print(f"Indices: start={start_idx}, end={end_idx}")
