# Weighted-trace Spacing Model + ACT-R Accuracy + ACT-R RT

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ----------------------------
# 1) Load data
# ----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "..", "slimstampen_all_trials.csv")
df = pd.read_csv(csv_path)


print("Loaded df shape:", df.shape)
print(df.head())

# ----------------------------
# 2) Helper functions
# ----------------------------
def logistic(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def softplus(x):
    x = np.clip(x, -60, 60)
    return np.log1p(np.exp(x))

# ----------------------------
# 3) Replay: weighted traces -> activation -> p(correct) + RT
# ----------------------------
def replay_weighted_actr(
    df: pd.DataFrame,
    d: float,
    w1: float,
    tau: float,
    s: float,
    T0: float,
    F: float,
    A_min: float = -10.0,
    learn_on: tuple = ("study", "test_correct"),
    rt_in_ms: bool = True
) -> pd.DataFrame:
    """
    Replays each (subj,item) through time, producing predictions at each test.

    Activation:
        A(t) = log( sum_i w_i * (t - t_i)^(-d) )

    Weight update (surprisal):
        w_new = w1                       if first trace
        w_new = softplus(-A(t_new))      otherwise

    Accuracy link (whiteboard form):
        p = 1 / (1 + exp((-A + tau)/s))  == logistic((A - tau)/s)

    RT link (whiteboard):
        RT_pred = T0 + F * exp(-A)

    Required df columns:
      subj, item, type, time, trial, isCorrect, RT
    """
    if d <= 0 or w1 <= 0 or s <= 0 or T0 < 0 or F <= 0:
        raise ValueError("Invalid parameter values.")

    work = df.copy()

    # Ensure isCorrect exists and is int (0/1)
    if "isCorrect" not in work.columns:
        raise ValueError("Expected column 'isCorrect' not found.")
    work["isCorrect"] = work["isCorrect"].astype(int)

    # Sort by subject + time + trial
    for col in ["subj", "item", "type", "time", "trial"]:
        if col not in work.columns:
            raise ValueError(f"Expected column '{col}' not found.")

    work = work.sort_values(["subj", "time", "trial"]).reset_index(drop=True)

    # RT -> seconds (ACT-R RT eq is typically in seconds)
    if "RT" in work.columns:
        if rt_in_ms:
            work["RT_sec"] = work["RT"] / 1000.0
        else:
            work["RT_sec"] = work["RT"].astype(float)
    else:
        work["RT_sec"] = np.nan

    state = {}  # (subj,item) -> {"traces":[...], "weights":[...]}

    out_rows = []

    for row in work.itertuples(index=False):
        key = (row.subj, row.item)
        t = float(row.time)
        typ = row.type
        correct = int(row.isCorrect)
        rt_obs = float(row.RT_sec) if not pd.isna(row.RT_sec) else np.nan

        if key not in state:
            state[key] = {"traces": [], "weights": []}

        traces = state[key]["traces"]
        weights = state[key]["weights"]

        # Compute odds = sum w_i * (dt)^(-d)
        odds = 0.0
        for tr, w in zip(traces, weights):
            dt = t - tr
            if dt > 0:
                odds += w * (dt ** (-d))

        # Activation
        if odds > 0:
            A = np.log(odds)
            A = max(A, A_min)
        else:
            A = A_min

        # TEST: produce predictions
        if typ == "test":
            p = logistic((A - tau) / s)
            rt_pred = T0 + F * np.exp(-A)

            out_rows.append({
                "subj": row.subj,
                "item": row.item,
                "trial": row.trial,
                "time": t,
                "A": A,
                "p_correct": p,
                "rt_pred": rt_pred,
                "isCorrect": correct,
                "rt_obs": rt_obs
            })

        # Decide whether this trial lays down a trace
        do_learn = False
        if typ == "study" and "study" in learn_on:
            do_learn = True
        elif typ == "test" and correct == 1 and "test_correct" in learn_on:
            do_learn = True
        elif typ == "test" and correct == 0 and "test_incorrect" in learn_on:
            do_learn = True

        if do_learn:
            if len(traces) == 0:
                w_new = w1
            else:
                # surprisal based on residual activation at time of encoding
                w_new = softplus(-A)

            traces.append(t)
            weights.append(float(w_new))

    return pd.DataFrame(out_rows)

# ----------------------------
# 4) Joint Negative Log Likelihood: Accuracy + RT
# ----------------------------
def nll_joint(theta, df, rt_only_correct=True, rt_in_ms=True):
    """
    theta = [d, w1, tau, s, T0, F, sigma_rt]
      - sigma_rt is lognormal sd on log(RT) residuals
    """
    d, w1, tau, s, T0, F, sigma_rt = theta

    # guards
    if d <= 0 or d > 2:
        return 1e12
    if w1 <= 0 or w1 > 200:
        return 1e12
    if s <= 1e-3 or s > 10:
        return 1e12
    if T0 < 0 or T0 > 5:
        return 1e12
    if F <= 0 or F > 50:
        return 1e12
    if sigma_rt <= 1e-3 or sigma_rt > 2:
        return 1e12

    pred = replay_weighted_actr(
        df,
        d=d, w1=w1,
        tau=tau, s=s,
        T0=T0, F=F,
        learn_on=("study", "test_correct"),
        rt_in_ms=rt_in_ms
    )

    # Accuracy NLL (Bernoulli)
    y = pred["isCorrect"].to_numpy(dtype=int)
    p = np.clip(pred["p_correct"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    nll_acc = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    # RT NLL (LogNormal around rt_pred)
    rt_obs = pred["rt_obs"].to_numpy(dtype=float)
    rt_pred = pred["rt_pred"].to_numpy(dtype=float)

    valid = np.isfinite(rt_obs) & np.isfinite(rt_pred) & (rt_obs > 0) & (rt_pred > 0)

    if rt_only_correct:
        valid = valid & (pred["isCorrect"].to_numpy(dtype=int) == 1)

    if valid.any():
        z = (np.log(rt_obs[valid]) - np.log(rt_pred[valid])) / sigma_rt
        nll_rt = 0.5 * np.sum(z**2) + np.sum(np.log(rt_obs[valid] * sigma_rt * np.sqrt(2*np.pi)))
    else:
        nll_rt = 0.0

    return nll_acc + nll_rt

# ----------------------------
# 5) Fit wrapper
# ----------------------------
def fit_weighted_model(df, rt_only_correct=True, rt_in_ms=True):
    # Initial guesses:
    # d, w1, tau, s, T0, F, sigma_rt
    x0 = np.array([0.30, 5.0, 0.0, 1.0, 0.30, 0.50, 0.35])

    bounds = [
        (0.01, 1.50),   # d
        (1e-3, 200.0),  # w1
        (-10, 10),      # tau
        (0.05, 5.0),    # s
        (0.0, 2.0),     # T0 (sec)
        (1e-3, 10.0),   # F  (sec)
        (0.05, 1.5)     # sigma_rt
    ]

    res = minimize(
        nll_joint,
        x0=x0,
        args=(df, rt_only_correct, rt_in_ms),
        method="L-BFGS-B",
        bounds=bounds
    )
    return res

# ----------------------------
# 6) Run fit
# ----------------------------
if __name__ == "__main__":

    print("Working directory:", os.getcwd())
    print("Files here:", os.listdir()[:15])  # show first few files

    res = fit_weighted_model(df, rt_only_correct=True, rt_in_ms=True)

    print("\n=== FIT RESULT ===")
    print("Converged:", res.success)
    print("Message:", res.message)
    print("NLL:", res.fun)
    print("Params [d, w1, tau, s, T0, F, sigma_rt]:")
    print(res.x)

    # ----------------------------
    # 7) Generate predictions using fitted params
    # ----------------------------
    pred = replay_weighted_actr(
        df,
        d=res.x[0], w1=res.x[1],
        tau=res.x[2], s=res.x[3],
        T0=res.x[4], F=res.x[5],
        learn_on=("study","test_correct"),
        rt_in_ms=True
    )

    print("\nPredictions head:")
    print(pred.head())

    print("\nRT obs (sec) summary (correct only):")
    print(pred.loc[pred.isCorrect == 1, "rt_obs"].describe())

    print("\nRT pred (sec) summary:")
    print(pred["rt_pred"].describe())
