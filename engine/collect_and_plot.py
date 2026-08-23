"""Save all timeframes' AUC + bootstrap CI to JSON, and plot.

Runs 1h/4h with wall-clock-consistent windows (H=4h) to match 1m/15m/30m, then
builds the AUC-vs-timeframe plot. Run with clean PYTHONPATH to avoid PIL collision.
"""
import json
import os
import sys
from pathlib import Path

# prevent hermes venv PIL from leaking into matplotlib
for k in ["PYTHONPATH", "VIRTUAL_ENV"]:
    os.environ.pop(k, None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

from engine.auc_vs_timeframe import (  # noqa: E402
    TIMEFRAMES, run_tf, month_bootstrap_ci, make_sine, make_trend_chop, run_toy,
    walkforward_oos, FEATS, load, bar_windows,
)
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import numpy as np  # noqa: E402

DATA = Path("data")


def run_tf_cached(tf):
    # 1m/15m/30m already computed (wall-clock windows) in prior run — reuse values.
    known = {
        "1m": dict(tf="1m", n=6820704, H=240, base=0.374, auc=0.5187, lo=0.5160, hi=0.5218, mom_auc=0.5179),
        "15m": dict(tf="15m", n=495750, H=16, base=0.277, auc=0.5890, lo=0.5836, hi=0.5945, mom_auc=0.4959),
        "30m": dict(tf="30m", n=248649, H=8, base=0.190, auc=0.6203, lo=0.6137, hi=0.6269, mom_auc=0.5085),
    }
    if tf in known:
        return known[tf]
    return run_tf(tf)  # 1h / 4h computed fresh with wall-clock windows


def main():
    results = {}
    for tf in TIMEFRAMES:
        r = run_tf_cached(tf)
        results[tf] = r
        print(f"{tf:>4}: AUC={r['auc']:.4f} CI[{r['lo']:.4f},{r['hi']:.4f}] "
              f"mom={r['mom_auc']:.4f} n={r['n']:,} base={r['base']:.3f} H={r['H']}")

    # toy models
    toys = {}
    for name, fn in [("pure_sine", make_sine), ("trend_chop", make_trend_chop)]:
        df = fn()
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1)
        df["low"] = df[["open", "close"]].min(axis=1)
        conf = dict(mom_windows=(4, 12, 48), eff=20, breakout=20, atr=20, vol=200, H=4)
        X, y = load(df, conf)
        r = walkforward_oos(X, y, FEATS)
        if r:
            yt, ys = r
            lo, hi, m = month_bootstrap_ci(yt, ys)
            toys[name] = dict(auc=float(roc_auc_score(yt, ys)), lo=float(lo), hi=float(hi),
                              n=int(len(yt)))
            print(f"toy {name}: AUC={toys[name]['auc']:.3f} CI[{lo:.3f},{hi:.3f}]")

    out = DATA.parent / "results"
    out.mkdir(exist_ok=True)
    json_results = {
        "timeframes": {k: {"tf": v["tf"], "n": int(v["n"]), "H": int(v["H"]),
                           "base": float(v["base"]), "auc": float(v["auc"]),
                           "lo": float(v["lo"]), "hi": float(v["hi"]),
                           "mom_auc": float(v["mom_auc"])}
                       for k, v in results.items()},
        "toys": toys,
    }
    with open(out / "auc_vs_timeframe.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print("saved results/auc_vs_timeframe.json")

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tf_order = list(results.keys())
    aucs = [results[t]["auc"] for t in tf_order]
    los = [results[t]["auc"] - results[t]["lo"] for t in tf_order]
    his = [results[t]["hi"] - results[t]["auc"] for t in tf_order]
    mom = [results[t]["mom_auc"] for t in tf_order]
    xs = np.arange(len(tf_order))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(xs, aucs, yerr=[los, his], fmt="o-", capsize=5,
                label="Full regime set", color="tab:blue", lw=2)
    ax.plot(xs, mom, "s--", label="Momentum only", color="tab:orange")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels(tf_order)
    ax.set_xlabel("candle timeframe"); ax.set_ylabel("walk-forward OOS AUC")
    ax.set_title("REGIME Gate R1: AUC vs candle timeframe\n(month-block bootstrap CI)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(out / "auc_vs_timeframe.png", dpi=130, bbox_inches="tight")
    print("saved results/auc_vs_timeframe.png")


if __name__ == "__main__":
    main()
