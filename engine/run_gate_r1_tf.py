"""Run Gate R1 on a chosen timeframe with frozen per-freq windows + target H.

Usage:
    python engine/run_gate_r1_tf.py 1h   # H=4 (4-hour barrier)
    python engine/run_gate_r1_tf.py 4h   # H=4 (16-hour barrier)

Per-freq windows (owner rule: never reuse another frequency's calibration):
- mom windows, trend_eff, breakout, atr, vol are tuned per timeframe.
Target A (barrier hit): price hits +2*ATR before -1*ATR within next H bars.
Report: walk-forward overall AUC, day-clustered AUC, month-block bootstrap CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).resolve().parent.parent / "data"
FEATS = ["mom_s", "mom_m", "mom_l", "trend_eff_N", "breakout", "kalman_slope", "vol_gate"]

# per-freq frozen windows: (mom_s, mom_m, mom_l, eff, breakout, atr, vol)  [bars]
TF_CONF = {
    "1h": dict(mom_windows=(4, 12, 48), eff=20, breakout=20, atr=20, vol=200, H=4),
    "4h": dict(mom_windows=(4, 12, 48), eff=20, breakout=20, atr=20, vol=200, H=4),
}


def load_features(df, conf):
    from engine.features import atr as _atr, build_features, kalman_slope
    from engine.target import barrier_hit
    a = _atr(df, conf["atr"])
    f = build_features(
        df,
        mom_windows=conf["mom_windows"],
        eff_window=conf["eff"],
        breakout_window=conf["breakout"],
        atr_window=conf["atr"],
        vol_window=conf["vol"],
    )
    lab = barrier_hit(df, a, H=conf["H"], up_mult=2.0, down_mult=-1.0)
    X = f.dropna()
    y = lab.reindex(X.index)
    m = y.notna()
    return X[m], y[m]


def month_block_ci(y_true, y_score, index, n_boot=800, alpha=0.05, max_per_month=2000):
    rng = np.random.default_rng(0)
    didx = pd.DatetimeIndex(index)
    months = didx.to_period("M").unique()
    obs_by_month = {m: didx[didx.to_period("M") == m] for m in months}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.choice(months, size=len(months), replace=True)
        pieces = []
        for m in sel:
            obs = obs_by_month[m]
            if len(obs) > max_per_month:
                obs = rng.choice(obs, size=max_per_month, replace=False)
            pieces.append(obs)
        idx = np.concatenate(pieces)
        yt = y_true.loc[idx]
        ys = y_score.loc[idx]
        if yt.nunique() < 2:
            draws[b] = np.nan
            continue
        try:
            draws[b] = roc_auc_score(yt, ys)
        except ValueError:
            draws[b] = np.nan
    draws = draws[np.isfinite(draws)]
    return np.quantile(draws, alpha / 2), np.quantile(draws, 1 - alpha / 2), draws.mean()


def main(tf):
    conf = TF_CONF[tf]
    df = pd.read_parquet(DATA / f"xau_{tf}.parquet")
    df.columns = [str(c).lower() for c in df.columns]
    print(f"=== Gate R1 on {tf} | {len(df)} bars | H={conf['H']} | base rate of +2ATR-first ===")
    X, y = load_features(df, conf)
    base = y.mean()
    print(f"rows {len(X):,} | base rate {base:.3f}")

    n = len(X)
    n_folds = 4
    boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
    embargo = conf["H"] + 1
    fold_aucs, day_aucs = [], []
    for f in range(1, n_folds):
        ts, te = boundaries[f], boundaries[f + 1]
        train_end = ts - embargo
        Xtr, ytr = X.iloc[:train_end], y.iloc[:train_end]
        Xte, yte = X.iloc[ts:te], y.iloc[ts:te]
        if ytr.nunique() < 2 or yte.nunique() < 2 or len(Xtr) < 1000:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        yscore = pd.Series(clf.predict_proba(Xte)[:, 1], index=Xte.index)
        fold_aucs.append(roc_auc_score(yte, yscore))
        # day-clustered AUC
        didx = pd.DatetimeIndex(Xte.index)
        for day, pos in pd.Series(np.arange(len(Xte)), index=didx).groupby(didx.normalize()):
            yt, ys = yte.loc[pos.index], yscore.loc[pos.index]
            if yt.nunique() >= 2:
                try:
                    day_aucs.append(roc_auc_score(yt, ys))
                except ValueError:
                    pass
        print(f"  fold {f}: train {len(Xtr):,} test {len(Xte):,} | overall AUC {fold_aucs[-1]:.4f}")

    print(f"\nwalk-forward overall AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    da = np.array(day_aucs)
    print(f"day-clustered AUC: mean {da.mean():.4f} | N days {len(da)}")

    # month-block CI on pooled
    yt_all = pd.concat([y.iloc[boundaries[f]:boundaries[f+1]] for f in range(1, n_folds)])
    # recompute scores for pooled
    scores = []
    for f in range(1, n_folds):
        ts, te = boundaries[f], boundaries[f + 1]
        Xtr, ytr = X.iloc[:ts - embargo], y.iloc[:ts - embargo]
        Xte, yte = X.iloc[ts:te], y.iloc[ts:te]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        scores.append(pd.Series(clf.predict_proba(Xte)[:, 1], index=Xte.index))
    ys_all = pd.concat(scores)
    lo, hi, m = month_block_ci(yt_all, ys_all, ys_all.index)
    print(f"month-block 95% CI [{lo:.4f},{hi:.4f}] mean {m:.4f} | excludes 0.5: {'YES' if lo>0.5 else 'NO'}")


if __name__ == "__main__":
    tf = sys.argv[1] if len(sys.argv) > 1 else "1h"
    main(tf)
