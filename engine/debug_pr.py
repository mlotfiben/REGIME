"""Debug precision/recall counting in the economic sweep."""
import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.decision_variants import FEATS, CONF, load
from engine.precision_recall import walkforward_oos, simulate_trades

df = pd.read_parquet("data/xau_1h.parquet")
df.columns = [str(c).lower() for c in df.columns]
df, atr, X, y = load(df)
score = walkforward_oos(X, y)
common = score.index
y_c = y.loc[common]

print("base rate (+2ATR first) =", float(y_c.mean()), "| n positives =", int(y_c.sum()), "| n bars =", len(y_c))
print("score range:", round(score.min(),4), "..", round(score.max(),4), "| score>0.5 frac:", float((score>0.5).mean()))

for thr in [0.085, 0.15, 0.25, 0.5]:
    dec = pd.Series((score >= thr).astype(float), index=score.index)
    pos, trade_ret, hit_up = simulate_trades(df, atr, dec)
    n_entries = int((pos.diff() > 0).sum())
    tp = int(hit_up.sum())
    print(f"\nthr={thr}: n_entries={n_entries} tp(+2ATR first exits)={tp} "
          f"precision={tp/n_entries if n_entries else 0:.3f} recall={tp/y_c.sum() if y_c.sum() else 0:.3f}")
    print(f"   trade_ret nonzero bars = {(trade_ret!=0).sum()}, pos>0 bars = {(pos>0).sum()}")
    # sanity: among entered, fraction positive label
    # label at the bars we are IN a position (entry bar = position start)
    entry_bars = pos.index[(pos.diff()>0)]
    pos_label = y_c.loc[entry_bars] if len(entry_bars) else pd.Series(dtype=float)
    print(f"   label(+2ATR) at entry bars: mean={pos_label.mean() if len(pos_label) else float('nan'):.3f}, n={len(pos_label)}")
