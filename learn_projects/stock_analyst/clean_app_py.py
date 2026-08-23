app_path = "/home/puneeth/repo/stock_fundamental/app.py"

with open(app_path, "r") as f:
    lines = f.readlines()

# Locate analyze_ticker function
start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "def analyze_ticker(ticker):" in line or "def analyze_ticker(" in line:
        start_idx = i
        break

for j in range(start_idx, len(lines)):
    if lines[j].strip().startswith("except Exception as e:"):
        end_idx = j
        break

# In analyze_ticker, find where related_stocks is built
render_start = -1
for k in range(start_idx, end_idx):
    if "return render_template(" in lines[k] or "# Fetch Edge AI Fundamental Intelligence" in lines[k]:
        render_start = k
        break

if render_start != -1:
    clean_tail = [
        "        # Fetch Edge AI Fundamental Intelligence\n",
        "        try:\n",
        "            from src.ai_analyst import get_stock_ai_intelligence\n",
        "            ai_report = get_stock_ai_intelligence(t, company_name, dorsey_data=dorsey_data)\n",
        "        except Exception as e:\n",
        "            print(f'AI intelligence error: {e}')\n",
        "            ai_report = None\n",
        "\n",
        "        return render_template(\n",
        "            \"results.html\",\n",
        "            ticker=t,\n",
        "            company_name=company_name,\n",
        "            ai_report=ai_report,\n",
        "            related_stocks=related_stocks,\n",
        "            current_industry=TICKER_TO_INDUSTRY.get(t, {}).get(\"industry\", \"\"),\n",
        "            **dorsey_data\n",
        "        )\n",
        "\n"
    ]
    
    new_lines = lines[:render_start] + clean_tail + lines[end_idx:]
    with open(app_path, "w") as f:
        f.writelines(new_lines)
    print("✅ app.py successfully cleaned of duplicate blocks!")
else:
    print("⚠️ Could not find render block")
