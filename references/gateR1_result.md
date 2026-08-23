# Gate R1 result — REGIME: regime → barrier-hit predictability (XAUUSD 1m)

Verdict: **real-but-tiny, decaying to operationally null.** The regime baseline has a
statistically real but economically meaningless predictive trace on the +2ATR/−1ATR
barrier target. Classified: **predictive but operationally irrelevant** (evidence
hierarchy) — NOT tradable alpha, NOT a clean null.

Source: `results/` + `engine/run_gate_r1.py`, `engine/check_gate_r1.py` (grep-able).

## Setup (frozen PREREG 2026-08-23)
- XAUUSD 1m, 6,822,714 bars (2004–2026). Raw CSV read-only, cached to parquet.
- Target A: price hits +2·ATR before −1·ATR within next H=60 bars. Base rate 0.372.
- Features (causal, frozen): mom(60/300/1440), trend_eff(120), breakout(240),
  kalman_slope(filterpy), vol_gate(1440).
- Logistic regression, walk-forward, purged+embargoed. **ACADEMIC gate** — tests
  predictive power (AUC); cost reported separately (known operational killer).

## Gate R1 — walk-forward overall AUC per fold (bar-level)

| fold | train bars | test bars | overall AUC | n_days |
|---|---|---|---|---|
| 1 (oldest) | 1,364,147 | 1,364,208 | **0.5331** | 1052 |
| 2 | 2,728,355 | 1,364,207 | 0.5219 | 1010 |
| 3 | 4,092,562 | 1,364,208 | 0.5156 | 1006 |
| 4 (most recent) | 5,456,770 | 1,364,208 | **0.5070** | 999 |

Decay across folds: 0.533 → 0.507. The most recent ~5 years is essentially null.

## Honesty check — month-block bootstrap CI (independent unit = month, not bar/day)

| fold | overall AUC | month-block 95% CI | excludes 0.5? |
|---|---|---|---|
| 1 (oldest) | 0.5331 | [0.5285, 0.5375] | YES (real) |
| 4 (most recent) | 0.5070 | [0.5023, 0.5109] | YES (but ~0.507) |

- The day-clustered t=44.5 / p≈0 in the naive run was **inflated by 1m label
  autocorrelation** (H=60 overlap within and across days → N=4,067 "days" are not
  independent). The month-block CI is the honest width.
- Even the most recent fold "excludes 0.5" only because n is enormous — at AUC 0.507
  that is operationally meaningless.

## Interpretation
- The regime features carry a **real but tiny** barrier-directional trace (AUC ~0.52,
  best in the oldest fold ~0.533, decaying to ~0.507 recently).
- Consistent with the program's standing finding: OHLC-only directional structure on
  gold is weak, sparse, and not a stable edge. Magnitude/vol transfers better than
  direction (the program's actual edge domain).
- **This is NOT a "PASS" in any operational sense.** It confirms the method "works" in
  the narrow statistical sense (features do move the forward barrier odds slightly)
  but is economically meaningless — the same conclusion the owner reached empirically:
  a directional model dies under costs.

## Cost note (secondary, per PREREG — academic gate)
At 1m with a barrier-target edge of AUC ~0.52 and high turnover, any realistic
round-trip cost (even 0.02%) swamps the edge. This is why the owner's earlier
directional model "died with the costs." The gate confirms the method is
statistically real but not economically viable at 1m — consistent with that prior.

## Scoping (overclaim rule)
Absence of a meaningful edge is scoped to: XAUUSD 1m, H=60 barrier target, these 7
frozen features, logistic walk-forward. A different H, different barrier multipliers,
or a magnitude (not directional) target could behave differently. What was tested
here shows a real-but-tiny, decaying, operationally-irrelevant trace.

## Classification
**predictive but operationally irrelevant** — statistically real, too weak for live
integration. FILED (method + result documented, reusable), not shipped, not discarded.
