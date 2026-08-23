"""AUC vs timeframe for REGIME Gate R1 (Target A barrier-hit), with bootstrap CI.

Per-timeframe windows are FIXED IN WALL-CLOCK so the plot is apples-to-apples:
  mom_s/mom_m/mom_l = 1h / 5h / 1d ; eff = 5h ; breakout = 1d ; atr = 5h ; vol = 1d
  barrier horizon H = 4h (wall-clock), converted to bars per timeframe (min 1).
Then: walk-forward logistic on the FULL 7-feature set, pooled OOS, month-block
bootstrap 95% CI. Plots AUC (and the momentum-only baseline) vs timeframe.

Toy-model checks (same pipeline on synthetic):
  - pure sinusoidal price: does the method find (spurious) structure?
  - trend/chop drift-switching: the hypothesis-fit signal the method MUST recover.
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
MOM_FEATS = ["mom_s", "mom_m", "mom_l"]

TIMEFRAMES = ["1m", "15m", "30m", "1h", "4h"]
MIN_PER_BAR = {"1m": 1, "15m": 15, "30m": 30, "1h": 60, "4h": 240}


def bar_windows(tf):
    """Wall-clock windows -> bars for this timeframe (min 1)."""
    m = MIN_PER_BAR[tf]
    def bars(minutes):
        return max(1, int(round(minutes / m)))
    return dict(
        mom_windows=(bars(60), bars(300), bars(1440)),   # 1h/5h/1d
        eff=bars(300), breakout=bars(1440), atr=bars(300), vol=bars(1440),
        H=max(1, int(round(240 / m))),                    # barrier horizon 4h wall-clock
    )


def load(df, conf):
    from engine.features import atr, build_features
    from engine.target import barrier_hit
    a = atr(df, conf["atr"])
    f = build_features(df, mom_windows=conf["mom_windows"], eff_window=conf["eff"],
                       breakout_window=conf["breakout"], atr_window=conf["atr"],
                       vol_window=conf["vol"])
    lab = barrier_hit(df, a, H=conf["H"], up_mult=2.0, down_mult=-1.0)
    X = f.dropna()
    y = lab.reindex(X.index)
    m = y.notna()
    return X[m], y[m]


def walkforward_oos(X, y, cols, n_folds=4):
    """Fit logistic per fold (expanding train, purged/embargoed), return pooled OOS."""
    n = len(X)
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    ys, scores = [], []
    for f in range(1, n_folds):
        ts, te = bounds[f], bounds[f + 1]
        embargo = 5
        Xtr, ytr = X.iloc[:ts - embargo][cols], y.iloc[:ts - embargo]
        Xte, yte = X.iloc[ts:te][cols], y.iloc[ts:te]
        if ytr.nunique() < 2 or yte.nunique() < 2 or len(Xtr) < 1000:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        ys.append(yte)
        scores.append(pd.Series(clf.predict_proba(Xte)[:, 1], index=Xte.index))
    if not ys:
        return None
    return pd.concat(ys), pd.concat(scores)


def month_bootstrap_ci(y_true, y_score, n_boot=800, alpha=0.05, max_per_month=2000):
    rng = np.random.default_rng(0)
    didx = pd.DatetimeIndex(y_score.index)
    months = didx.to_period("M").unique()
    obs_by_month = {m: didx[didx.to_period("M") == m] for m in months}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.choice(months, size=len(months), replace=True)
        pieces = []
        for m in sel:
            o = obs_by_month[m]
            if len(o) > max_per_month:
                o = rng.choice(o, size=max_per_month, replace=False)
            pieces.append(o)
        idx = np.concatenate(pieces)
        yt, ys = y_true.loc[idx], y_score.loc[idx]
        if yt.nunique() < 2:
            draws[b] = np.nan; continue
        try:
            draws[b] = roc_auc_score(yt, ys)
        except ValueError:
            draws[b] = np.nan
    draws = draws[np.isfinite(draws)]
    return np.quantile(draws, alpha/2), np.quantile(draws, 1-alpha/2), draws.mean()


def run_tf(tf):
    conf = bar_windows(tf)
    df = pd.read_parquet(DATA / f"xau_{tf}.parquet")
    df.columns = [str(c).lower() for c in df.columns]
    X, y = load(df, conf)
    r = walkforward_oos(X, y, FEATS)
    r_mom = walkforward_oos(X, y, MOM_FEATS)
    if r is None:
        return None
    yt, ys = r
    auc = roc_auc_score(yt, ys)
    lo, hi, m = month_bootstrap_ci(yt, ys)
    mom_auc = roc_auc_score(*r_mom) if r_mom else np.nan
    return dict(tf=tf, n=len(X), H=conf["H"], base=y.mean(), auc=auc, lo=lo, hi=hi, mean=m,
                mom_auc=mom_auc)


# ---------------- toy models ----------------

def make_sine(n=200000, period=2000, amp=1.0):
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    close = 100 + amp * np.sin(2 * np.pi * np.arange(n) / period)
    return pd.DataFrame({"close": close}, index=idx)


def make_trend_chop(n=200000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    rets = np.empty(n)
    for b in range(n // 2000 + 1):
        lo, hi = b * 2000, min((b + 1) * 2000, n)
        if b % 2 == 0:
            rets[lo:hi] = rng.normal(0.0003, 0.0005, hi - lo)   # mild trend
        else:
            rets[lo:hi] = rng.normal(0, 0.0012, hi - lo)         # chop
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"close": close}, index=idx)


def run_toy(df, name):
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    # use 1h-equivalent windows on this synthetic (as if 1h bars)
    conf = dict(mom_windows=(4, 12, 48), eff=20, breakout=20, atr=20, vol=200, H=4)
    X, y = load(df, conf)
    r = walkforward_oos(X, y, FEATS)
    if r is None:
        print(f"  {name}: no valid folds"); return
    yt, ys = r
    auc = roc_auc_score(yt, ys)
    lo, hi, m = month_bootstrap_ci(yt, ys)
    print(f"  {name}: AUC {auc:.3f} | month-block CI [{lo:.3f},{hi:.3f}] | n {len(yt):,}")


def main():
    results = []
    for tf in TIMEFRAMES:
        r = run_tf(tf)
        if r:
            results.append(r)
            print(f"{tf:>4}: bars={r['n']:>8,} H={r['H']} base={r['base']:.3f} "
                  f"AUC={r['auc']:.4f} CI[{r['lo']:.4f},{r['hi']:.4f}] mom_auc={r['mom_auc']:.4f}")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tf_order = [r["tf"] for r in results]
    aucs = [r["auc"] for r in results]
    los = [r["auc"] - r["lo"] for r in results]
    his = [r["hi"] - r["auc"] for r in results]
    mom = [r["mom_auc"] for r in results]
    xs = np.arange(len(tf_order))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(xs, aucs, yerr=[los, his], fmt="o-", capsize=5, label="Full regime set", color="tab:blue")
    ax.plot(xs, mom, "s--", label="Momentum only", color="tab:orange")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(tf_order)
    ax.set_xlabel("candle timeframe"); ax.set_ylabel("walk-forward OOS AUC")
    ax.set_title("REGIME Gate R1: AUC vs candle timeframe (month-block bootstrap CI)")
    ax.legend(); ax.grid(alpha=0.3)
    out = DATA.parent / "results" / "auc_vs_timeframe.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")

    print("\n=== Toy-model checks (same pipeline) ===")
    run_toy(make_sine(), "pure sine (periodic, no trend regime)")
    run_toy(make_trend_chop(), "trend/chop drift-switching (hypothesis-fit)")


if __name__ == "__main__":
    main()
