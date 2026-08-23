"""Precision-Recall analysis for REGIME barrier model (1h, enhanced features).

The two competing effects: PRECISION (win rate -> higher = fewer losing trades = less
cost waste) vs COST (turnover -> more trades = more cost). The PR curve shows the
precision achievable at each recall level; the economic overlay shows where the
precision/cost competition actually turns net-of-cost positive vs the risk-matched
constant baseline.

For each probability threshold: simulate the long-only barrier strategy, compute
precision & recall (on the +2ATR-first label) and the net-of-cost Sharpe vs the
risk-matched constant at 0.02% round-trip. Reports:
  - precision at 20% recall
  - the recall level where economics flip negative (vs const)
  - full PR curve + economic overlay plot
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from engine.decision_variants import FEATS, CONF, load, causal_reynolds  # reuse

DATA = Path(__file__).resolve().parent.parent / "data"
COST = 0.0002  # measured-cost baseline, round-trip


def walkforward_oos(X, y, n_folds=4):
    """Return pooled OOS predicted P(+2ATR-first) aligned to X.index."""
    n = len(X)
    bounds = np.linspace(0, n, n_folds + 1, dtype=int)
    scores = []
    for f in range(1, n_folds):
        ts, te = bounds[f], bounds[f + 1]
        Xtr, ytr = X.iloc[:ts - CONF["H"] - 1], y.iloc[:ts - CONF["H"] - 1]
        Xte = X.iloc[ts:te]
        if ytr.nunique() < 2 or len(Xtr) < 1000:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        scores.append(pd.Series(clf.predict_proba(Xte)[:, 1], index=Xte.index))
    return pd.concat(scores)


def simulate_trades(df, atr, entry_decision):
    """Long-only barrier trades from a boolean entry_decision (aligned to df.index).
    Returns (pos, trade_ret) and realized outcomes for precision/recall.

    FIX: after a trade exits at bar j, the next bar considered is j+1 — NO immediate
    re-entry at the exit bar (which created overlapping/adjacent trades and inflated
    returns at low thresholds). One open position at a time, one return per exit.
    """
    close = df["close"].values
    atr_v = atr.values
    n = len(df)
    pos = np.zeros(n)
    trade_ret = np.zeros(n)
    hit_up = np.zeros(n)  # 1 if this exit hit +2ATR first (a precision/recall "positive")
    dec = entry_decision.reindex(df.index).fillna(False).values
    H = CONF["H"]
    i = 0
    while i < n:
        if dec[i]:
            ep, base = close[i], atr_v[i]
            if np.isfinite(base) and base > 0:
                up_b, dn_b = ep + 2 * base, ep - base
                fw = close[i + 1: i + 1 + H]
                if len(fw):
                    tu = np.where(fw >= up_b)[0]
                    td = np.where(fw <= dn_b)[0]
                    tu = tu[0] if tu.size else np.inf
                    td = td[0] if td.size else np.inf
                    if tu < td and np.isfinite(tu):
                        k = int(tu) + 1
                        trade_ret[i + k] = 2 * base / ep
                        hit_up[i + k] = 1.0
                        pos[i:i + k + 1] = 1.0
                        i = i + k + 1  # next bar AFTER exit (no immediate re-entry)
                        continue
                    if td < tu and np.isfinite(td):
                        k = int(td) + 1
                        trade_ret[i + k] = -base / ep
                        pos[i:i + k + 1] = 1.0
                        i = i + k + 1
                        continue
                end_i = min(i + H, n - 1)
                if end_i > i:
                    trade_ret[end_i] = close[end_i] / ep - 1
                    pos[i:end_i + 1] = 1.0
                    i = end_i + 1
                    continue
        i += 1
    return pd.Series(pos, index=df.index), pd.Series(trade_ret, index=df.index), pd.Series(hit_up, index=df.index)


def evaluate_threshold(df, atr, y, score, thr):
    """At threshold thr: precision, recall, n_trades, net Sharpe vs const @ COST.

    ECONOMIC precision/recall (unambiguous, trade-level):
      - a "win" = a completed trade with realized return > 0 (net of 0 cost reference)
      - precision = fraction of trades that win
      - recall = wins / total label positives (+2ATR-first bars in the window)
    This is the quantity that competes against cost, not bar-level bookkeeping.
    """
    dec = pd.Series((score >= thr).astype(float), index=score.index)
    pos, trade_ret, hit_up = simulate_trades(df, atr, dec)
    # each completed trade = one nonzero trade_ret (its realized return)
    ret_vals = trade_ret[trade_ret != 0]
    n_trades = int(len(ret_vals))
    wins = int((ret_vals > 0).sum())
    prec = wins / n_trades if n_trades else 0.0
    recall = wins / int(y.sum()) if y.sum() else 0.0
    # economics (0.02% round-trip on turnover = per completed trade)
    strat_ret = trade_ret.reindex(df.index).fillna(0.0)
    pos_s = pos.reindex(df.index).fillna(0.0)
    exit_events = (strat_ret != 0).astype(float)
    net = strat_ret - exit_events * COST
    w_const = pos_s.mean()
    const_ret = w_const * df["close"].pct_change().fillna(0.0)
    return dict(thr=thr, precision=prec, recall=recall, n_trades=n_trades,
                scheme_sharpe=_sharpe(net), const_sharpe=_sharpe(const_ret),
                diff=_sharpe(net) - _sharpe(const_ret))


def _sharpe(ret):
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) < 2 or ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(len(ret)))


def main():
    df = pd.read_parquet(DATA / "xau_1h.parquet")
    df.columns = [str(c).lower() for c in df.columns]
    df, atr, X, y = load(df)
    score = walkforward_oos(X, y)
    common = score.index
    y_c = y.loc[common]

    # sklearn PR curve (for the plot)
    prec_sk, rec_sk, thr_sk = precision_recall_curve(y_c, score.values)

    # economic sweep over thresholds
    thrs = np.quantile(score.values, np.linspace(0.5, 0.99, 40))
    rows = [evaluate_threshold(df, atr, y_c, score, t) for t in thrs]

    print("=== Precision-Recall economic sweep (1h enhanced, cost 0.02%) ===")
    print(f"{'thr':>6} {'precision':>9} {'recall':>7} {'n_trades':>8} {'schemeShp':>9} {'constShp':>8} {'diff':>7}")
    for r in rows:
        print(f"{r['thr']:.3f} {r['precision']:>9.3f} {r['recall']:>7.3f} {r['n_trades']:>8} "
              f"{r['scheme_sharpe']:>9.2f} {r['const_sharpe']:>8.2f} {r['diff']:>+7.3f}")

    # find recall ~0.20
    r20 = min([r for r in rows if r["recall"] <= 0.21], key=lambda r: abs(r["recall"] - 0.20), default=None)
    # find where diff flips (scheme stops beating const)
    pos_rows = [r for r in rows if r["diff"] > 0]
    print("\n=== Key findings ===")
    if r20:
        print(f"At ~20% recall: precision={r20['precision']:.3f}, n_trades={r20['n_trades']}, "
              f"scheme Sharpe {r20['scheme_sharpe']:.2f} vs const {r20['const_sharpe']:.2f} (diff {r20['diff']:+.3f})")
    # best diff point
    best = max(rows, key=lambda r: r["diff"])
    print(f"Best point on frontier: recall={best['recall']:.3f}, precision={best['precision']:.3f}, "
          f"diff={best['diff']:+.3f} (scheme {best['scheme_sharpe']:.2f} vs const {best['const_sharpe']:.2f})")
    # highest recall where diff still positive
    pos_recalls = [r["recall"] for r in pos_rows]
    if pos_recalls:
        print(f"Diff stays positive up to recall={max(pos_recalls):.3f} (beyond that, cost wins)")
    else:
        print("Diff NEVER positive — scheme never beats const at any recall on this frontier")

    # save plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(rec_sk, prec_sk, ".-", color="tab:blue")
    ax1.set_xlabel("recall"); ax1.set_ylabel("precision"); ax1.set_title("Precision-Recall curve (sklearn)")
    ax1.grid(alpha=0.3)
    recs = [r["recall"] for r in rows]; diffs = [r["diff"] for r in rows]
    ax2.plot(recs, diffs, "o-", color="tab:orange")
    ax2.axhline(0, color="gray", ls=":")
    ax2.set_xlabel("recall"); ax2.set_ylabel("net Sharpe diff (scheme - const, 0.02% cost)")
    ax2.set_title("Economic frontier: where precision vs cost compete")
    ax2.grid(alpha=0.3)
    out = DATA.parent / "results" / "precision_recall_econ.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
