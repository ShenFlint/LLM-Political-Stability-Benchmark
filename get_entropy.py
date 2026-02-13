import os
import pandas as pd
from math import log2

BASE_DIR = "/Users/oceans/Documents/political-github/political_result_1"
OUTPUT_FILE = "summary_entropy.csv"

def entropy(values):
    counts = pd.Series(values).value_counts(normalize=True)
    return -sum(p * log2(p) for p in counts if p > 0)

def compute_confidence(persona_id: str) -> str:
    low_set = {f"user_{i}" for i in range(11, 16)}
    low_set |= {f"user_{i:02d}" for i in range(11, 16)}
    return "low" if persona_id in low_set else "high"

def parse_from_folder(folder_name: str):
    parts = folder_name.split("_")
    if len(parts) < 5:
        return None
    prompt_strategy = parts[2]
    model = parts[3]
    try:
        temperature = float(parts[4])
    except Exception:
        return None
    return prompt_strategy, model, temperature

def compute_entropy_mean_all(base_dir: str = BASE_DIR, output_file: str = OUTPUT_FILE):
    results = []

    for root, _, files in os.walk(base_dir):
        if not files:
            continue

        folder = os.path.basename(root)
        parsed = parse_from_folder(folder)
        if parsed is None:
            continue
        prompt_strategy, model, temperature = parsed

        for file in files:
            if not file.endswith("_answers.csv"):
                continue

            persona_id = file.replace("_answers.csv", "")
            confidence = compute_confidence(persona_id)
            path = os.path.join(root, file)

            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"Read error: {path} ({e})")
                continue

            run_cols = [c for c in df.columns if str(c).startswith("run_")]
            if len(run_cols) == 0:
                print(f"Missing run_* columns: {path}")
                continue

            run_cols = sorted(
                run_cols,
                key=lambda x: int(str(x).split("_")[1]) if "_" in str(x) else 999
            )

            if "question_id" not in df.columns:
                print(f"Missing question_id column: {path}")
                continue

            entropies = []
            for _, row in df.iterrows():
                responses = [row[c] for c in run_cols]
                entropies.append(entropy(responses))

            entropy_mean = sum(entropies) / len(entropies) if entropies else None
            if entropy_mean is None:
                continue

            results.append({
                "Persona_ID": persona_id,
                "Model": model,
                "Temperature": temperature,
                "Prompt_Strategy": prompt_strategy,
                "Confidence": confidence,
                "Entropy": entropy_mean
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Done. Saved to {output_file}")
    print(results_df.head())
    return results_df

if __name__ == "__main__":
    compute_entropy_mean_all()