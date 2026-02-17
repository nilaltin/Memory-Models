import os
import sys
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
pavlik_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(pavlik_dir)

from model_pavlik import fit_pavlik_model, replay_pavlik_actr

project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
csv_path = os.path.join(project_dir, "slimstampen_all_trials.csv")

df = pd.read_csv(csv_path)

default_subj = int(df["subj"].iloc[0])
s_in = input(f"Enter subject ID (e.g., {default_subj}): ").strip()
subject_id = int(s_in) if s_in else default_subj

df_one = df[df["subj"] == subject_id].copy()

print("\n=== PAVLIK ONE-SUBJECT FIT (c = 0.5 fixed) ===")
print("Subject:", subject_id)
print("Trials:", len(df_one))

res = fit_pavlik_model(df_one)

print("\n=== FIT RESULT ===")
print("Converged:", res.success)
print("NLL:", res.fun)
print("Params [phi, tau, s, T0, F, sigma_rt]:")
print(res.x)

phi, tau, s, T0, F, sigma_rt = res.x

pred = replay_pavlik_actr(df_one, phi, tau, s, T0, F)

print("\nPredictions head:")
print(pred.head())
