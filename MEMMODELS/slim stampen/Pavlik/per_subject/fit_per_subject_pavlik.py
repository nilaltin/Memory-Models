# Fit Pavlik ACT-R model per subject (c = 0.5 fixed)

import os
import sys
import pandas as pd

def main():
    # --- paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))           # .../Pavlik/per_subject
    pavlik_dir = os.path.abspath(os.path.join(script_dir, ".."))      # .../Pavlik
    project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))  # .../slim stampen
    csv_path = os.path.join(project_dir, "slimstampen_all_trials.csv")

    # allow importing model_pavlik.py from Pavlik/
    sys.path.insert(0, pavlik_dir)
    from model_pavlik import fit_pavlik_model


    # --- load data ---
    df = pd.read_csv(csv_path)
    subjects = sorted(df["subj"].unique())

    print("CSV:", csv_path)
    print("N subjects:", len(subjects))
    print("Subjects (first 10):", subjects[:10], "...")

    results = []

    for i, subj in enumerate(subjects, start=1):
        df_s = df[df["subj"] == subj].copy()

        res = fit_pavlik_model(df_s, rt_only_correct=True, rt_in_ms=True)

        phi, tau, s, T0, F, sigma_rt = res.x

        row = {
            "subj": int(subj),
            "n_trials": int(len(df_s)),
            "converged": bool(res.success),
            "NLL": float(res.fun),
            "phi": float(phi),
            "tau": float(tau),
            "s": float(s),
            "T0": float(T0),
            "F": float(F),
            "sigma_rt": float(sigma_rt),
        }
        results.append(row)

        print(
            f"[{i}/{len(subjects)}] subj={subj} "
            f"conv={row['converged']} NLL={row['NLL']:.3f} "
            f"phi={row['phi']:.4f} sigma_rt={row['sigma_rt']:.3f}"
        )

    out_csv = os.path.join(script_dir, "pavlik_per_subject_fits.csv")
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print(out_df.head())


if __name__ == "__main__":
    main()
