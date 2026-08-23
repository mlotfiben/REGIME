"""Parallel decision variants for REGIME Gate R1 (1h) — attack turnover two ways.

Variant R — Re_t/vol-regime gate (reuse existing gauge):
    barrier strategy, but only enter when causal Re_t >= its rolling median (the
    laminar/trending half). Tests the owner's hypothesis that the existing Re_t gauge
    cuts turnover enough to flip economics.

Variant M — meta-labeling (Lopez de Prado Ch.3.6/50):
    secondary logistic predicts P(trade profitable after cost | primary features +
    Re_t + session + vol_gate); trade only top-confidence quantile (meta_theta =
    80th pctile of TRAINING meta-p, causal).

Both: long-only barrier strategy at 1h, cost-sensitivity curve, risk-matched constant
baseline, day-cluster bootstrap P(scheme worse). Same protocol as the base test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

DATA = Path(__file__).resolve().parent.parent / "data"
FEATS = ["mom_s", "mom_m", "mom_l", "trend_eff_N", "breakout", "kalman_slope", "vol_gate",
         "fracdiff_price", "fracdiff_ret", "session"]
CONF = dict(mom_windows=(1, 5, 24), eff=5, breakout=24, atr=5, vol=24, H=4)


def load(df):
    from engine.enhanced_features import build_enhanced_features
    from engine.features import atr
    from engine.target import barrier_hit
    a = atr(df, CONF["atr"])
    f = build_enhanced_features(df)
    lab = barrier_hit(df, a, H=CONF["H"], up_mult=2.0, down_mult=-1.0)
    common = f.dropna().index.intersection(lab.dropna().index)
    X = f.loc[common, FEATS]
    y = lab.loc[common]
    return df, a, X, y


def causal_reynolds(df, window=20):
    """Canonical causal Re_t (COLORSLOPE engine/regime.py, trailing-ATR no-lookahead)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "color_regime", str(Path.home() / "COLORSLOPE/engine/regime.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    return pd.Series(mod.compute_reynolds(close, high, low, window), index=df.index)


def session_feature(idx):
    """London/NY session as a causal feature (hour of day only, no future info)."""
    h = idx.hour
    return pd.Series(((h >= 7) & (h <= 16)).astype(float), index=idx)  # approx London+NY


def walkforward_entries(X, y, df, atr, re_t, variant, theta_q=0.80, meta_q=0.80, n_folds=4):
    """Return per-bar entry decision (bool Series) and realized trade returns.

    variant='R' -> Re_t gate; variant='M' -> meta-labeling.
    """
    n = len(X)
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    idx = X.index
    pos = pd.Series(0.0, index=idx)
    trade_ret = pd.Series(0.0, index=idx)
    close = df["close"].values
    atr_v = atr.values
    row_of = pd.Series(np.arange(len(df)), index=df.index)

    # causal Re_t per row and session
    re_arr = re_t.reindex(df.index).values
    sess_arr = session_feature(df.index).values

    for f in range(1, n_folds):
        ts, te = bounds[f], bounds[f + 1]
        embargo = CONF["H"] + 1
        Xtr, ytr = X.iloc[:ts - embargo], y.iloc[:ts - embargo]
        Xte, yte = X.iloc[ts:te], y.iloc[ts:te]
        if ytr.nunique() < 2 or len(Xtr) < 1000:
            continue

        # primary barrier model
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        p_te = clf.predict_proba(Xte)[:, 1]
        theta = np.quantile(clf.predict_proba(Xtr)[:, 1], theta_q)

        rows_tr = row_of.loc[Xtr.index].values
        rows_te = row_of.loc[Xte.index].values
        re_tr = re_arr[rows_tr]
        re_te = re_arr[rows_te]

        if variant == "R":
            # gate: enter only when Re_t >= median of TRAINING Re_t (laminar half)
            thr = np.nanmedian(re_tr)
            entry_te = (p_te >= theta) & (re_te >= thr)
        else:  # M: meta-labeling
            # build secondary label on TRAIN: is primary trade (p>=theta) profitable?
            p_tr = clf.predict_proba(Xtr)[:, 1]
            meta_y_tr = []
            meta_X_tr = []
            c_tr = close[rows_tr]
            a_tr = atr_v[rows_tr]
            for i in range(len(rows_tr)):
                if p_tr[i] >= theta and np.isfinite(a_tr[i]) and a_tr[i] > 0:
                    ep = c_tr[i]; base = a_tr[i]
                    up_b, dn_b = ep + 2 * base, ep - base
                    fw = c_tr[i + 1: i + 1 + CONF["H"]]
                    if len(fw):
                        tu = np.where(fw >= up_b)[0]
                        td = np.where(fw <= dn_b)[0]
                        tu = tu[0] if tu.size else np.inf
                        td = td[0] if td.size else np.inf
                        if tu < td and np.isfinite(tu):
                            prof = 2 * base / ep > 0.0002
                        elif td < tu and np.isfinite(td):
                            prof = (-base / ep) > 0.0002
                        else:
                            prof = (c_tr[min(i + CONF["H"], len(c_tr) - 1)] / ep - 1) > 0.0002
                    else:
                        prof = False
                    meta_X_tr.append(np.concatenate([Xtr.iloc[i].values,
                                                     [re_tr[i], sess_arr[rows_tr[i]]]]))
                    meta_y_tr.append(1.0 if prof else 0.0)
            if len(meta_y_tr) < 100 or len(set(meta_y_tr)) < 2:
                entry_te = (p_te >= theta)
            else:
                mclf = LogisticRegression(max_iter=1000)
                mclf.fit(np.array(meta_X_tr), np.array(meta_y_tr))
                meta_theta = np.quantile(mclf.predict_proba(np.array(meta_X_tr))[:, 1], meta_q)
                mX_te = np.column_stack([Xte.values,
                                         re_te, sess_arr[rows_te]])
                m_te = mclf.predict_proba(mX_te)[:, 1]
                entry_te = (p_te >= theta) & (m_te >= meta_theta)

        # simulate trades from entry_te
        L = len(rows_te)
        pos_te = np.zeros(L)
        ret_te = np.zeros(L)
        c_te = close[rows_te]
        a_te = atr_v[rows_te]
        i = 0
        H = CONF["H"]
        while i < L:
            if entry_te[i]:
                ep, base = c_te[i], a_te[i]
                if np.isfinite(base) and base > 0:
                    up_b, dn_b = ep + 2 * base, ep - base
                    fw = c_te[i + 1: i + 1 + H]
                    if len(fw):
                        tu = np.where(fw >= up_b)[0]
                        td = np.where(fw <= dn_b)[0]
                        tu = tu[0] if tu.size else np.inf
                        td = td[0] if td.size else np.inf
                        if tu < td and np.isfinite(tu):
                            k = int(tu) + 1
                            ret_te[i + k] = 2 * base / ep
                            pos_te[i:i + k + 1] = 1.0
                            i += k + 1; continue   # no immediate re-entry (integrity fix)
                        if td < tu and np.isfinite(td):
                            k = int(td) + 1
                            ret_te[i + k] = -base / ep
                            pos_te[i:i + k + 1] = 1.0
                            i += k + 1; continue
                    end_i = min(i + H, L - 1)
                    if end_i > i:
                        ret_te[end_i] = c_te[end_i] / ep - 1
                        pos_te[i:end_i + 1] = 1.0
                        i = end_i + 1
                        continue
            i += 1
        pos.iloc[ts:te] = pos_te
        trade_ret.iloc[ts:te] = ret_te
    return pos, trade_ret


def sharpe(ret):
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) < 2 or ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(len(ret)))


def bootstrap_diff(pos, trade_ret, df, cost, n_boot=1500):
    rng = np.random.default_rng(0)
    strat_ret = trade_ret.reindex(df.index).fillna(0.0)
    pos = pos.reindex(df.index).fillna(0.0)
    exit_events = (strat_ret != 0).astype(float)
    sched = (strat_ret - exit_events * cost).groupby(pd.DatetimeIndex(df.index).normalize()).sum()
    cst = (pos.mean() * df["close"].pct_change().fillna(0.0)).groupby(
        pd.DatetimeIndex(df.index).normalize()).sum()
    days = sched.index
    diffs = []
    for _ in range(n_boot):
        sel = rng.choice(days, size=len(days), replace=True)
        sd, cd = sched.loc[sel], cst.loc[sel]
        ss = sd.mean() / (sd.std(ddof=1) if sd.std(ddof=1) > 0 else 1e-12)
        cs = cd.mean() / (cd.std(ddof=1) if cd.std(ddof=1) > 0 else 1e-12)
        diffs.append(ss - cs)
    diffs = np.array(diffs)
    return float((diffs <= 0).mean()), np.quantile(diffs, 0.025), np.quantile(diffs, 0.975)


def main():
    df = pd.read_parquet(DATA / "xau_1h.parquet")
    df.columns = [str(c).lower() for c in df.columns]
    df, atr, X, y = load(df)
    re_t = causal_reynolds(df)
    print(f"loaded {len(df):,} 1h bars | Re_t valid {int(re_t.notna().sum()):,}")

    costs = [0.0, 0.0001, 0.0002, 0.0005, 0.001]
    for variant, name in [("R", "Re_t/vol-regime gate"), ("M", "meta-labeling")]:
        print(f"\n=== Variant {variant}: {name} ===")
        pos, trade_ret = walkforward_entries(X, y, df, atr, re_t, variant)
        strat_ret = trade_ret.reindex(df.index).fillna(0.0)
        pos = pos.reindex(df.index).fillna(0.0)
        exit_events = (strat_ret != 0).astype(float)
        w_const = pos.mean()
        const_sharpe = sharpe(w_const * df["close"].pct_change().fillna(0.0))
        print(f"exposure {w_const:.3f} | n_trades {int(exit_events.sum())}")
        print(f"{'cost%':>7} {'schemeShp':>10} {'constShp':>9} {'schemeNet':>11}")
        for c in costs:
            net = strat_ret - exit_events * c
            print(f"{c*100:>7.3f} {sharpe(net):>10.3f} {const_sharpe:>9.3f} {net.sum():>11.4f}")
        p, lo, hi = bootstrap_diff(pos, trade_ret, df, 0.0002)
        print(f"bootstrap P(scheme worse) at 0.02% = {p:.4f} (CI of diff [{lo:.3f},{hi:.3f}])")


if __name__ == "__main__":
    main()
