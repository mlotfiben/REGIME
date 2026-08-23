"""Stage 1 — TILT decomposition: cost vs forfeited-passive-drift (XAUUSD 1h).

For the corrected meta-labeling variant (326 trades / 20y), decompose the Sharpe
shortfall vs the risk-matched constant baseline into:
  (a) cost paid on executed trades
  (b) passive drift forfeited while flat (opportunity cost)

Three counterfactuals (same 326 trades, same directions/sizes):
  scheme@0.02%     — as run (with cost)
  scheme@0%        — cost-free (isolates cost)
  always-invested  — same trades, but hold const exposure during non-traded hours
                     (removes the forfeited-drift; net of cost)

cost_contribution  = Sharpe(scheme@0%)       - Sharpe(scheme@0.02%)
oppcost_contribution = Sharpe(always-invested) - Sharpe(scheme@0.02%) - cost_contribution
total_shortfall    = Sharpe(const) - Sharpe(scheme@0.02%)

Decision rule (frozen): if oppcost / total_shortfall < 30% -> stop, retire tilt.
Caveat: the additive split assumes cost & drift effects are separable; report raw
Sharpe values and note the interaction caveat.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from decision_variants import CONF, FEATS, load, walkforward_entries, causal_reynolds

DATA = Path(__file__).resolve().parent.parent / "data"
COST = 0.0002


def sharpe(ret):
    ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) < 2 or ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(len(ret)))


def main():
    df = pd.read_parquet(DATA / "xau_1h.parquet")
    df.columns = [str(c).lower() for c in df.columns]
    df, atr, X, y = load(df)
    re_t = causal_reynolds(df)
    pos, trade_ret = walkforward_entries(X, y, df, atr, re_t, variant="M")

    idx = df.index
    strat_ret = trade_ret.reindex(idx).fillna(0.0)
    pos = pos.reindex(idx).fillna(0.0)
    exit_events = (strat_ret != 0).astype(float)
    n_trades = int(exit_events.sum())
    w_const = pos.mean()
    close_pct = df["close"].pct_change().fillna(0.0)
    const_ret = w_const * close_pct

    scheme_net = strat_ret - exit_events * COST      # as run, 0.02%
    scheme_gross = strat_ret                          # cost-free
    # always-invested: same trades, but hold const exposure when flat (net of cost)
    always_inv = strat_ret + (1.0 - pos) * close_pct * w_const - exit_events * COST

    sh_scheme = sharpe(scheme_net)
    sh_gross = sharpe(scheme_gross)
    sh_always = sharpe(always_inv)
    sh_const = sharpe(const_ret)

    total_shortfall = sh_const - sh_scheme
    cost_contrib = sh_gross - sh_scheme
    oppcost_contrib = sh_always - sh_scheme - cost_contrib

    print("=== TILT Stage 1 — decomposition (1h meta-labeling variant, corrected) ===")
    print(f"n_trades={n_trades} | avg exposure={w_const:.3f} | base rate={y.mean():.3f}")
    print(f"Sharpe(scheme @ 0.02%)        = {sh_scheme:.3f}")
    print(f"Sharpe(scheme @ 0% cost-free) = {sh_gross:.3f}")
    print(f"Sharpe(always-invested)       = {sh_always:.3f}")
    print(f"Sharpe(risk-matched const)    = {sh_const:.3f}")
    print()
    print(f"total shortfall (const - scheme)      = {total_shortfall:.3f}")
    print(f"cost contribution    = {cost_contrib:.3f}  ({100*cost_contrib/max(total_shortfall,1e-12):.1f}%)")
    print(f"opp-cost contribution = {oppcost_contrib:.3f}  ({100*oppcost_contrib/max(total_shortfall,1e-12):.1f}%)")
    decision = "PROCEED to Stage 2 (tilt)" if oppcost_contrib >= 0.30 * total_shortfall else \
               "STOP at Gate 0 (opportunity cost not material; retire tilt)"
    print(f"\n30% decision rule: oppcost {oppcost_contrib:.3f} vs 30% of {total_shortfall:.3f} = {0.30*total_shortfall:.3f}")
    print(f"VERDICT: {decision}")
    print("\nCaveat: additive split assumes cost & drift effects are separable; raw Sharpe "
          "values above are the honest numbers (interaction may exist).")


if __name__ == "__main__":
    main()
