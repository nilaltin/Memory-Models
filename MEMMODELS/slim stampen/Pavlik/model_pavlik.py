# model_pavlik.py
# Pavlik ACT-R model with fixed c = 0.5

import numpy as np
import pandas as pd
from scipy.optimize import minimize

C_CONST = 0.5  


# ----------------------------
# helpers
# ----------------------------
def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_log(x, eps=1e-12):
    return np.log(np.clip(x, eps, None))


# ----------------------------
# replay
# ----------------------------
def replay_pavlik_actr(
    df,
    phi,
    tau,
    s,
    T0,
    F,
    learn_on=("study", "test_correct"),
    rt_in_ms=True,
    A_min=-10,
    A_clip=10,
):
    work = df.copy()
    work["isCorrect"] = work["isCorrect"].astype(int)
    work = work.sort_values(["time", "trial"]).reset_index(drop=True)

    if "RT" in work.columns:
        if rt_in_ms:
            work["RT_sec"] = work["RT"] / 1000.0
        else:
            work["RT_sec"] = work["RT"].astype(float)
    else:
        work["RT_sec"] = np.nan

    traces_by_item = {}

    out = []

    for row in work.itertuples(index=False):
        item = int(row.item)
        t = float(row.time)
        typ = str(row.type)
        correct = int(row.isCorrect)
        rt_obs = float(row.RT_sec) if not pd.isna(row.RT_sec) else np.nan

        if item not in traces_by_item:
            traces_by_item[item] = []

        traces = traces_by_item[item]

        odds = 0.0
        for (t_i, d_i) in traces:
            dt = t - t_i
            if dt > 0:
                odds += dt ** (-d_i)

        if odds > 0:
            A = np.log(odds)
            A = max(A, A_min)
        else:
            A = A_min

        if typ == "test":
            p = logistic((A - tau) / s)
            rt_pred = T0 + F * np.exp(-A)

            out.append(
                {
                    "item": item,
                    "trial": int(row.trial),
                    "time": t,
                    "A": A,
                    "p_correct": float(p),
                    "rt_pred": float(rt_pred),
                    "isCorrect": correct,
                    "rt_obs": rt_obs,
                }
            )

        do_learn = False
        if typ == "study" and "study" in learn_on:
            do_learn = True
        elif typ == "test" and correct == 1 and "test_correct" in learn_on:
            do_learn = True

        if do_learn:
            A_res = float(np.clip(A, A_min, A_clip))
            d_new = phi + C_CONST * np.exp(A_res)
            d_new = float(np.clip(d_new, 0.01, 5.0))

            traces.append((t, d_new))

    return pd.DataFrame(out)


# ----------------------------
# likelihood
# ----------------------------
def nll_pavlik_joint(theta, df, rt_only_correct=True, rt_in_ms=True):
    phi, tau, s, T0, F, sigma_rt = theta

    pred = replay_pavlik_actr(
        df,
        phi=phi,
        tau=tau,
        s=s,
        T0=T0,
        F=F,
        learn_on=("study", "test_correct"),
        rt_in_ms=rt_in_ms,
    )

    if len(pred) == 0:
        return 1e12

    y = pred["isCorrect"].to_numpy(dtype=int)
    p = np.clip(pred["p_correct"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)

    nll_acc = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    rt_obs = pred["rt_obs"].to_numpy(dtype=float)
    rt_pred = pred["rt_pred"].to_numpy(dtype=float)

    valid = (rt_obs > 0) & (rt_pred > 0)
    if rt_only_correct:
        valid &= (y == 1)

    if valid.sum() == 0:
        return float(nll_acc)

    z = (safe_log(rt_obs[valid]) - safe_log(rt_pred[valid])) / sigma_rt
    nll_rt = 0.5 * np.sum(z**2)

    return float(nll_acc + nll_rt)


# ----------------------------
# fit
# ----------------------------
def fit_pavlik_model(df, rt_only_correct=True, rt_in_ms=True):
    x0 = np.array([0.5, 0.0, 1.0, 0.3, 0.5, 0.3])
    bounds = [
        (0.01, 2.0),   # phi
        (-10, 10),     # tau
        (0.05, 5.0),   # s
        (0.0, 5.0),    # T0
        (0.01, 10.0),  # F
        (0.05, 2.0),   # sigma_rt
    ]

    obj = lambda th: nll_pavlik_joint(th, df, rt_only_correct, rt_in_ms)

    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
    return res
