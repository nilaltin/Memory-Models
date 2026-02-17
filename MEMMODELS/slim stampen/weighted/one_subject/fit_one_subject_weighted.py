# Fit weighted-trace model to ONE subject

import os
import sys
import pandas as pd

def main():

    # this script lives in .../slim stampen/weighted/one_subject/
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # allow importing from weighted/pooled/
    pooled_dir = os.path.abspath(os.path.join(script_dir, "..", "pooled"))
    sys.path.append(pooled_dir)

    from modelmemorypractice1 import fit_weighted_model

    # CSV lives in .../slim stampen/
    project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    csv_path = os.path.join(project_dir, "slimstampen_all_trials.csv")

    df = pd.read_csv(csv_path)

    print("CSV:", csv_path)
    print("Available subjects:", sorted(df["subj"].unique())[:10], "...")

    subject_id = int(input("Enter subject ID to fit: ").strip())

    df_one = df[df["subj"] == subject_id].copy()

    print(f"\n=== WEIGHTED ONE-SUBJECT FIT ===")
    print("Subject:", subject_id)
    print("Trials:", len(df_one))

    res = fit_weighted_model(df_one, rt_only_correct=True, rt_in_ms=True)

    print("\n=== FIT RESULT ===")
    print("Converged:", res.success)
    print("Message:", res.message)
    print("NLL:", res.fun)
    print("Params [d, w1, tau, s, T0, F, sigma_rt]:")
    print(res.x)


if __name__ == "__main__":
    main()
