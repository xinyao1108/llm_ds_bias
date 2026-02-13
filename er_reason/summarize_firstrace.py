import pandas as pd

df = pd.read_csv("er_reason.csv")

counts = df["firstrace"].value_counts(dropna=False)
total = len(df)

print(f"Total records: {total}\n")
print("firstrace distribution:")
print("-" * 45)
for val, count in counts.items():
    pct = count / total * 100
    label = str(val) if pd.notna(val) else "(missing)"
    print(f"  {label:<35} {count:>5}  ({pct:.1f}%)")
