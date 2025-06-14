import pandas as pd
import json

# 设置指令内容
instruction_text = "You are an expert in the field of sports injuries. Please infer the main content of the abstract based on the given paper title."

# 读取 Excel 数据
df = pd.read_excel("D:/a_work/1-phD/project/3-bibliometric/data-sports-injury-medline-26307-20241115/sports-injury-medline-22866-20241115.xlsx")

# 准备 Alpaca 格式数据
alpaca_data = []
for _, row in df.iterrows():
    title = row.get("Article Title", "")
    abstract = row.get("Abstract", "")
    if pd.notnull(title) and pd.notnull(abstract) and str(abstract).strip():
        entry = {
            "instruction": instruction_text,
            "input": str(title).strip(),
            "output": str(abstract).strip()
        }
        alpaca_data.append(entry)

# 写入 JSON 文件（符合 Alpaca 格式）
output_path = "sports_injury_alpaca_dataset.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for item in alpaca_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

