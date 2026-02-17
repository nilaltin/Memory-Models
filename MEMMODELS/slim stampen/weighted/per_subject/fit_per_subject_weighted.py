# Fit weighted-trace ACT-R model per subject (individual d's)
import os
import sys
import pandas as pd

# Import weighted model from weighted/pooled/
pooled_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pooled"))
sys.path.insert(0, pooled_dir)
from modelmemorypractice1 import fit_weighted_model

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    csv_path = os.path.join(project_dir, "slimstampen_all_trials.csv")


    # Output file location
    out_csv = os.path.join(script_dir, "per_subject_fits.csv")
    out_log = os.path.join(script_dir, "per_subject_log.txt")

    df = pd.read_csv(csv_path)
    subjects = sorted(df["subj"].unique())

    results = []

    # Optional: log progress to a text file (in addition to terminal)
    with open(out_log, "w") as logf:
        logf.write(f"Loaded data: {csv_path}\n")
        logf.write(f"N subjects: {len(subjects)}\n\n")

        for i, subj in enumerate(subjects, start=1):
            df_s = df[df["subj"] == subj].copy()

            # Fit this subject
            res = fit_weighted_model(df_s, rt_only_correct=True, rt_in_ms=True)

            row = {
                "subj": subj,
                "n_trials": len(df_s),
                "converged": bool(res.success),
                "NLL": float(res.fun),
                "d": float(res.x[0]),
                "w1": float(res.x[1]),
                "tau": float(res.x[2]),
                "s": float(res.x[3]),
                "T0": float(res.x[4]),
                "F": float(res.x[5]),
                "sigma_rt": float(res.x[6]),
            }
            results.append(row)

            msg = (
                f"[{i}/{len(subjects)}] subj={subj} "
                f"converged={row['converged']} "
                f"d={row['d']:.4f} NLL={row['NLL']:.2f}\n"
            )
            print(msg, end="")
            logf.write(msg)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_csv, index=False)

    print(f"\nSaved per-subject fits → {out_csv}")
    print(results_df.head())

if __name__ == "__main__":
    main()
