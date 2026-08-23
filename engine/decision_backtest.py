"""Decision-level backtest for REGIME Gate R1 — does the barrier edge survive costs?

Strategy (frozen PREREG 2026-08-23): long-only. At bar t, walk-forward logistic predicts
p = P(hit +2ATR before -1ATR in next H bars). Enter long when p >= theta (theta = 80th
pct of TRAINING p, per-fold, causal). Hold until barrier or time exit:
  +2ATR first -> return +2*ATR[t]/close[t]
  -1ATR first -> return -1*ATR[t]/close[t]
  neither in H -> time exit at close[t+H]: return close[t+H]/close[t]-1
Flat otherwise. Returns fractional, comparable to baseline.

Fair baseline: RISK-MATCHED CONSTANT exposure w_const = mean(position) held constantly
(captures gold's drift, zero turnover). Always-in reported for context only.

Cost: per-trade (entry+exit) on turnover only. Cost-sensitivity curve mandatory.
Pass bar: scheme net Sharpe > const net Sharpe at measured cost, day-clustered block
bootstrap one-sided p<0.05, Bonferroni across 1h & 30m.

Harness controls: positive (injected drift<->signal link) and negative (iid -> no edge).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression

DATA = Path(__file__).resolve().parent.parent / "data"
FEATS = ["mom_s", "mom_m", "mom_l", "trend_eff_N", "breakout", "kalman_slope", "vol_gate"]

TF_CONF = {
    "30m": dict(mom_windows=(2, 10, 48), eff=10, breakout=48, atr=10, vol=48, H=8),
    "1h":  dict(mom_windows=(1, 5, 24),  eff=5,  breakout=24, atr=5,  vol=24, H=4),
}


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
    return df, a, X[m], y[m]


def oos_probs_and_positions(X, y, df, atr, conf, theta_q=0.80, n_folds=4):
    """Walk-forward logistic, return per-bar entry decision + realized trade returns.

    Vectorized trade simulation with numpy (the naive per-bar .loc loop was too slow
    on 100k+ bars).
    """
    n = len(X)
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    idx = X.index
    pos = pd.Series(0.0, index=idx)
    trade_ret = pd.Series(0.0, index=idx)
    close = df["close"].values
    atr_v = atr.values

    # position of each row in df's close/atr arrays
    row_of = pd.Series(np.arange(len(df)), index=df.index)

    for f in range(1, n_folds):
        ts, te = bounds[f], bounds[f + 1]
        embargo = conf["H"] + 1
        Xtr, ytr = X.iloc[:ts - embargo], y.iloc[:ts - embargo]
        Xte, yte = X.iloc[ts:te], y.iloc[ts:te]
        if ytr.nunique() < 2 or len(Xtr) < 1000:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        pte = clf.predict_proba(Xte)[:, 1]
        theta = np.quantile(clf.predict_proba(Xtr)[:, 1], theta_q)

        # map test index -> array rows
        rows = row_of.loc[Xte.index].values
        c_te = close[rows]
        a_te = atr_v[rows]
        p_te = pte
        L = len(rows)
        pos_te = np.zeros(L)
        ret_te = np.zeros(L)

        i = 0
        H = conf["H"]
        while i < L:
            if p_te[i] >= theta:
                entry_price = c_te[i]
                base = a_te[i]
                if np.isfinite(base) and base > 0:
                    up_b = entry_price + 2 * base
                    dn_b = entry_price - base
                    fwd = c_te[i + 1: i + 1 + H]
                    if len(fwd):
                        up_t = np.where(fwd >= up_b)[0]
                        dn_t = np.where(fwd <= dn_b)[0]
                        tu = up_t[0] if up_t.size else np.inf
                        td = dn_t[0] if dn_t.size else np.inf
                        if tu < td and np.isfinite(tu):
                            k = int(tu) + 1
                            ret_te[i + k] = 2 * base / entry_price
                            pos_te[i:i + k + 1] = 1.0
                            i += k + 1; continue   # no immediate re-entry (integrity fix)
                        if td < tu and np.isfinite(td):
                            k = int(td) + 1
                            ret_te[i + k] = -base / entry_price
                            pos_te[i:i + k + 1] = 1.0
                            i += k + 1; continue
                    # time exit (guaranteed to advance i)
                    end_i = min(i + H, L - 1)
                    if end_i <= i:
                        i += 1
                    else:
                        ret_te[end_i] = c_te[end_i] / entry_price - 1
                        pos_te[i:end_i + 1] = 1.0
                        i = end_i + 1
                    continue
            i += 1
        # write back into the position/return series on the test slice positions
        pos.iloc[ts:te] = pos_te
        trade_ret.iloc[ts:te] = ret_te
    return pos, trade_ret


def backtest(df, atr, pos, trade_ret, costs=(0.0, 0.0001, 0.0002, 0.0005, 0.001)):
    """Net PnL, Sharpe, turnover for scheme and risk-matched constant at each cost."""
    close = df["close"]
    # per-bar strategy return: trade_ret realized at exit bar; each trade pays ONE
    # round-trip cost at its exit bar (not |Δw|*c which double-counts entry+exit).
    strat_ret = trade_ret
    exit_events = (trade_ret != 0).astype(float)  # 1 at each exit bar = one trade
    pos = pos.fillna(0.0)
    out = {}
    for c in costs:
        net = strat_ret - exit_events * c
        scheme_sharpe = _sharpe(net)
        # risk-matched constant: same avg exposure held constantly, zero turnover
        w_const = pos.mean()
        const_ret = w_const * close.pct_change().fillna(0.0)
        const_sharpe = _sharpe(const_ret)
        always_in_sharpe = _sharpe(close.pct_change().fillna(0.0))
        out[c] = dict(scheme_net=float(net.sum()), scheme_sharpe=float(scheme_sharpe),
                      const_sharpe=float(const_sharpe), always_in=float(always_in_sharpe),
                      turnover=float(exit_events.sum()), n_trades=int(exit_events.sum()),
                      exposure=float(pos.mean()))
    return out


def _sharpe(ret, ppy=None):
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) < 2 or ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(len(ret)))  # per-period, comparable across schemes


def run_tf(tf, costs=(0.0, 0.0001, 0.0002, 0.0005, 0.001)):
    conf = TF_CONF[tf]
    df = pd.read_parquet(DATA / f"xau_{tf}.parquet")
    df.columns = [str(c).lower() for c in df.columns]
    df, atr, X, y = load(df, conf)
    pos, trade_ret = oos_probs_and_positions(X, y, df, atr, conf)
    return df, pos, trade_ret, backtest(df, atr, pos, trade_ret, costs)


def day_cluster_bootstrap_diff(pos, trade_ret, df, cost, n_boot=2000):
    """Block bootstrap over days of the scheme-vs-const daily Sharpe difference."""
    rng = np.random.default_rng(0)
    # daily scheme net return and const net return
    strat_ret = trade_ret.reindex(df.index).fillna(0.0)
    pos = pos.reindex(df.index).fillna(0.0)
    exit_events = (strat_ret != 0).astype(float)
    scheme_daily = (strat_ret - exit_events * cost).groupby(pd.DatetimeIndex(df.index).normalize()).sum()
    const_daily = (pos.mean() * df["close"].pct_change().fillna(0.0)).groupby(
        pd.DatetimeIndex(df.index).normalize()).sum()
    days = scheme_daily.index
    diffs = []
    for b in range(n_boot):
        sel = rng.choice(days, size=len(days), replace=True)
        sd = scheme_daily.loc[sel]
        cd = const_daily.loc[sel]
        ss = sd.mean() / (sd.std(ddof=1) if sd.std(ddof=1) > 0 else 1e-12)
        cs = cd.mean() / (cd.std(ddof=1) if cd.std(ddof=1) > 0 else 1e-12)
        diffs.append(ss - cs)
    diffs = np.array(diffs)
    p = float((diffs <= 0).mean())  # one-sided P(diff <= 0)
    return p, np.quantile(diffs, 0.025), np.quantile(diffs, 0.975)


def main():
    for tf in ["30m", "1h"]:
        df, pos, trade_ret, res = run_tf(tf)
        print(f"\n=== DECISION-LEVEL BACKTEST: {tf} ===")
        print(f"exposure {res[0.0]['exposure']:.3f} | n_trades {res[0.0]['n_trades']}")
        print(f"{'cost%':>7} {'schemeShp':>10} {'constShp':>9} {'alwaysIn':>9} {'turnover':>9} {'schemeNet':>10}")
        for c in [0.0, 0.0001, 0.0002, 0.0005, 0.001]:
            r = res[c]
            print(f"{c*100:>7.3f} {r['scheme_sharpe']:>10.3f} {r['const_sharpe']:>9.3f} "
                  f"{r['always_in']:>9.3f} {r['turnover']:>9.1f} {r['scheme_net']:>10.4f}")
        # bootstrap at measured cost 0.02%
        p, lo, hi = day_cluster_bootstrap_diff(pos, trade_ret, df, 0.0002)
        print(f"day-cluster bootstrap P(diff<=0) at 0.02% cost = {p:.4f} (CI of diff [{lo:.3f},{hi:.3f}])")


if __name__ == "__main__":
    main()
