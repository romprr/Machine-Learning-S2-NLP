import json
import re

# Count text features
with open('handcrafted_features/extractors_text.py', 'r') as f:
    text_content = f.read()
text_feats = len(re.findall(r"df_feats\['feat_(.*?)'\]", text_content))

# Count manual code features
with open('handcrafted_features/extractors_code.py', 'r') as f:
    code_content = f.read()
manual_code_feats = len(re.findall(r"'feat_[a-zA-Z0-9_]+':\s*r'", code_content))

# Count auto-discovered features kept (logic from extractors_code.py)
discovered = 0
try:
    with open('handcrafted_features/discovered_candidates.json', 'r') as f:
        data = json.load(f)
        for tag, cands in data.items():
            discovered += len(cands[:10]) # max_auto_per_tag is 10 by default
except Exception as e:
    print("Error:", e)

print(f"Manual text features: {text_feats}")
print(f"Manual code features: {manual_code_feats}")
print(f"Total manual features: {text_feats + manual_code_feats}")
print(f"Auto-discovered features kept: {discovered}")
