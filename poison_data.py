import os
import pandas as pd
import numpy as np

data_path = "data/iris.csv"

df = pd.read_csv(data_path)

poison_levels = {
    "poison_5": 0.05,
    "poison_10": 0.10,
    "poison_50": 0.50
}

def poison_data(df, fraction):
    poisoned = df.copy()
    n = int(len(df) * fraction)

    idx = np.random.choice(df.index, size=n, replace=False)

    numeric_cols = poisoned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        noise = np.random.normal(0, poisoned[col].std() * 0.2, size=n)
        poisoned.loc[idx, col] += noise

    if "species" in poisoned.columns:
        poisoned.loc[idx, "species"] = np.random.permutation(poisoned.loc[idx, "species"].values)

    return poisoned

for folder, frac in poison_levels.items():
    folder_path = os.path.join("data", folder)
    os.makedirs(folder_path, exist_ok=True)

    poisoned_df = poison_data(df, frac)
    output_path = os.path.join(folder_path, "iris_poisoned.csv")
    poisoned_df.to_csv(output_path, index=False)

    print(f"Created {output_path} with {int(len(df) * frac)} poisoned rows.")
