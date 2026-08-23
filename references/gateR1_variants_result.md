# REGIME Parallel Variants result — Re_t gate vs meta-labeling (2026-08-23)

Both decision variants run in parallel on 1h barrier strategy, vs the FAIR baseline
(risk-matched constant exposure). Pre-registered in PREREG "PARALLEL VARIANTS".

Source: `engine/decision_variants.py`.

## Results (cost-sensitivity, scheme Sharpe vs const Sharpe 3.56)

| | base (no filter) | Variant R (Re_t gate) | Variant M (meta-labeling) |
|---|---|---|---|
| n_trades (20y) | 7,139 | 3,668 | **1,807** |
| avg exposure | 0.213 | 0.114 | 0.060 |
| Sharpe @ 0% | 10.22 | 6.52 | 6.65 |
| Sharpe @ 0.01% | 6.41 | 3.84 | 4.67 |
| **Sharpe @ 0.02%** | 2.57 | 1.14 | **2.66** |
| Sharpe @ 0.05% | −8.95 | −6.94 | −3.44 |
| **P(scheme worse) @ 0.02%** | 0.829 | 0.970 | **0.731** |
| diff CI @ 0.02% | — | [−0.065, +0.001] | [−0.044, +0.022] |

Const baseline Sharpe = 3.56 (same everywhere).

## Verdict
**Both FAIL the pre-committed pass bar** (scheme net Sharpe > const net Sharpe AND
bootstrap one-sided p < 0.05 that scheme is better). Neither survives cost vs doing
nothing.

## Interpretation (both legs reported together, per owner rule)
- **Variant R (Re_t/vol-regime gate) — FAIL, informative.** Halving turnover with the
  existing causal Re_t gauge made economics WORSE (Sharpe 1.14 vs 2.57; P(worse) 0.970).
  The Re_t gate removes good and bad trades alike — it does NOT discriminate profitable
  vs unprofitable barrier trades. **Independently re-confirms Gate C1** (instability/
  regime-gating on gold does not improve forward outcomes). The owner's existing gauge,
  used as a TRADE GATE, is not the source.
- **Variant M (meta-labeling) — best of the three, but still not a pass.** Cut turnover
  ~75% (1,807 trades) while HOLDING net Sharpe (2.66 vs base 2.57). The only variant that
  improves on base — consistent with Lopez de Prado's meta-labeling (Ch.3.6/50): a
  secondary model filters primary trades to raise precision. But at 0.02% cost it still
  loses to its own risk-matched constant (P(worse) 0.731, diff CI includes 0).

## Why this is the productive path (honest)
- Meta-labeling is the RIGHT direction: it attacks the exact failure (turnover), is the
  only variant that beats the base, and beats the naive single-feature Re_t gate decisively.
- But a 75% turnover cut was not enough — the per-trade 1h barrier edge is too thin. The
  edge must get meaningfully stronger (higher AUC / better features) for meta-labeling's
  precision gain to clear the cost hurdle. Not a dead end; a "need more edge" signal.

## Directionality of the source
- Re_t as a trade GATE = confirmed dead end (C1 replicate).
- Re_t as a FEATURE inside meta-labeling = where it belongs (M already uses it).

## Classification
Both variants: **predictive but operationally irrelevant** at 1h as tested. Meta-labeling
upgraded from "FAIL" to "promising mechanism, insufficient edge" — the mechanism is
validated, the edge is not yet.
