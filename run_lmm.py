import pandas as pd
import os
import statsmodels.formula.api as smf

OUT_DIR = "lmm_icc"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("summary_oss_entropy_merged.csv")

df["Model"] = df["Model"].astype("category")
df["Prompt_Strategy"] = df["Prompt_Strategy"].astype("category")
df["Persona_ID"] = df["Persona_ID"].astype("category")
df["Temperature"] = df["Temperature"].astype(float)
df["Confidence"] = df["Confidence"].astype("category")

print("Data loaded. Rows:", len(df))
print(df.head())

def run_lmm(dep_var):
    formula = f"{dep_var} ~ Model + Temperature + Prompt_Strategy + Confidence"
    model = smf.mixedlm(formula, data=df, groups=df["Persona_ID"])
    return model.fit(reml=False)

print("\n===== LMM (OSS) =====")
oss_lmm = run_lmm("OSS")
print(oss_lmm.summary())

print("\n===== LMM (Entropy) =====")
ent_lmm = run_lmm("Entropy")
print(ent_lmm.summary())

def compute_icc(lmm_result, name):
    var_persona = float(lmm_result.cov_re.iloc[0, 0])
    var_resid = float(lmm_result.scale)
    icc = var_persona / (var_persona + var_resid)

    print(f"\n===== ICC ({name}) =====")
    print(f"{name} random intercept variance: {var_persona:.6f}")
    print(f"{name} residual variance:        {var_resid:.6f}")
    print(f"{name} ICC:                     {icc:.6f}")

    return var_persona, var_resid, icc

oss_var_p, oss_var_e, oss_icc = compute_icc(oss_lmm, "OSS")
ent_var_p, ent_var_e, ent_icc = compute_icc(ent_lmm, "Entropy")

out_path = os.path.join(OUT_DIR, "lmm_and_icc_results.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("===== OSS Linear Mixed Model Results =====\n")
    f.write(str(oss_lmm.summary()))
    f.write("\n\n")
    f.write("===== Entropy Linear Mixed Model Results =====\n")
    f.write(str(ent_lmm.summary()))
    f.write("\n\n")
    f.write("===== ICC Results =====\n")
    f.write(f"OSS: var_persona={oss_var_p:.6f}, var_resid={oss_var_e:.6f}, ICC={oss_icc:.6f}\n")
    f.write(f"Entropy: var_persona={ent_var_p:.6f}, var_resid={ent_var_e:.6f}, ICC={ent_icc:.6f}\n")

print("\nDone.")
print("Output directory:", OUT_DIR)
print("Output file:", out_path)