from pathlib import Path
import pandas as pd

BASE_DIR = Path("/Users/oceans/Downloads/political-compass-master")

oss_path = BASE_DIR / "summary_oss.csv"
ent_path = BASE_DIR / "summary_entropy.csv"
out_path = BASE_DIR / "summary_oss_entropy_merged.csv"

df_oss = pd.read_csv(oss_path)
df_ent = pd.read_csv(ent_path)

keys = ["Persona_ID", "Model", "Temperature", "Prompt_Strategy"]

df_oss["Temperature"] = pd.to_numeric(df_oss["Temperature"], errors="coerce")
df_ent["Temperature"] = pd.to_numeric(df_ent["Temperature"], errors="coerce")

df_ent = df_ent[keys + ["Confidence", "Entropy"]]

merged = df_oss.merge(df_ent, on=keys, how="left", validate="one_to_one")

cols = [
    "Persona_ID", "Model", "Temperature", "Prompt_Strategy",
    "SD_econ", "Mean_econ", "SD_polit", "Mean_polit", "OSS",
    "Confidence", "Entropy"
]
merged = merged[cols]

merged.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"Saved: {out_path} (rows={len(merged)})")