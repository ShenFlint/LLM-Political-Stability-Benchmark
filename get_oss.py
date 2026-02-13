import re
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path("/Users/oceans/Documents/political-github/score")
OUT_CSV = "summary_oss.csv"

FOLDER_RE = re.compile(r"^score_(cot|zero)_([^_]+)_([0-9]*\.?[0-9]+)$")
PERSONA_RE = re.compile(r"user_(\d+)")

rows = []

for subdir in sorted([p for p in BASE_DIR.iterdir() if p.is_dir()]):
    m = FOLDER_RE.match(subdir.name)
    if not m:
        continue

    prompt_strategy = m.group(1)
    model = m.group(2)
    temperature = float(m.group(3))

    for csv_path in sorted(subdir.glob("*.csv")):
        pm = PERSONA_RE.search(csv_path.name)
        if not pm:
            continue
        persona_id = f"user_{pm.group(1).zfill(2)}"

        df = pd.read_csv(csv_path)

        cols = {c.strip(): c for c in df.columns}
        if "Economic" not in cols or "Social" not in cols:
            raise ValueError(f"Missing Economic/Social columns in {csv_path}")

        econ = pd.to_numeric(df[cols["Economic"]], errors="coerce").dropna()
        soc  = pd.to_numeric(df[cols["Social"]], errors="coerce").dropna()

        sd_econ = float(econ.std(ddof=1)) if len(econ) >= 2 else float("nan")
        mean_econ = float(econ.mean()) if len(econ) >= 1 else float("nan")

        sd_soc = float(soc.std(ddof=1)) if len(soc) >= 2 else float("nan")
        mean_soc = float(soc.mean()) if len(soc) >= 1 else float("nan")

        oss = float(np.sqrt(sd_econ**2 + sd_soc**2)) if np.isfinite(sd_econ) and np.isfinite(sd_soc) else float("nan")

        rows.append({
            "Persona_ID": persona_id,
            "Model": model,
            "Temperature": temperature,
            "Prompt_Strategy": prompt_strategy,
            "SD_econ": sd_econ,
            "Mean_econ": mean_econ,
            "SD_polit": sd_soc,
            "Mean_polit": mean_soc,
            "OSS": oss,
        })

out_df = pd.DataFrame(rows)

if not out_df.empty:
    out_df["Persona_ID_int"] = out_df["Persona_ID"].str.replace("user_", "", regex=False).astype(int)
    out_df = out_df.sort_values(["Persona_ID_int", "Model", "Temperature", "Prompt_Strategy"]).drop(columns=["Persona_ID_int"])

out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"Saved: {OUT_CSV}  (rows={len(out_df)})")