import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./stacking.csv', index_col=0)

plt.figure(figsize=(18, 14))

sns.heatmap(df, annot=True, fmt='g', cmap='Blues', cbar=True, annot_kws={"size": 10})

plt.title("Stacking Ensemble Confusion Matrix", pad=20, fontsize=22, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=18, fontweight='bold')
plt.ylabel('True Label', fontsize=18, fontweight='bold')

plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.tight_layout()
plt.savefig("stacking.png")
plt.show()
