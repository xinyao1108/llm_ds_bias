import pandas as pd
import json

df = pd.read_csv("er_reason.csv")

race_map = {
    "White": "white",
    "Asian": "asian",
    "Black or African American": "black",
}

result = {}
for race_label, key in race_map.items():
    subset = df[df["firstrace"] == race_label]
    result[key] = subset.to_dict(orient="records")
    print(f"{key}: {len(subset)} records")

with open("patients_by_race.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

print("\nSaved to patients_by_race.json")
