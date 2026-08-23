# PREREG — REGIME pre-registration (draft)

Status: **DRAFT — freeze after owner review.** Asset: XAUUSD H1.

## Frozen spec
- **Asset / timeframe:** XAUUSD, H1 (125,793 bars, 2004–2026).
- **Target:** (choose one in review — barrier-hit +2ATR/−1ATR is recommended)
  - A: price hits +2·ATR before −1·ATR within next H bars (H frozen).
  - B: next H-window is high- vs low-efficiency.
  - C: regime label trend/chop/expansion/calm.
- **Features (causal, frozen):** mom_s/mom_m/mom_l, trend_eff_N, breakout, vol_gate,
  kalman_slope (filterpy causal).
- **Cost/slippage:** single round-trip number, correct units (e.g. 0.02%), applied
  on turnover only (NOT per-bar — per-bar cost crushes Sharpe, taxonomy bug 22).
- **Splits:** one train / one validation / one locked final test (~20%). Walk-forward
  with purge ≥ 1 bar + embargo.
- **Benchmarks:** persistence; risk-matched constant `w_const = mean(w_scheme)`;
  always-in.
- **Stats:** day-clustered (N = independent days), not per-bar. Deflation for search
  counting all prior trials (≥ 20).
- **Pass criterion (provisional, freeze in review):** OOS predictive power (AUC /
  IC) above the trivial benchmark, day-clustered CI excluding null, holds in both
  halves, winsorize-robust.

## A–D gate (mandatory before real data)
- **A. Spec:** every quantity precisely defined and frozen (above).
- **B. Leakage audit:** no return obs contributes simultaneously to feature +
  target; no-lookahead unit test (perturb future bar → past features bit-identical).
- **C. Effective-N:** independent units after overlap/cooldown/clustering; fix
  overlap structurally (cooldown).
- **D. Synthetic controls:** pipeline recovers a known injected efficiency/barrier
  edge (positive) AND stays silent when the shape exists but effect is absent
  (negative).

## Failure action
Document the null; retire or scope precisely; do not soften the gate (taxonomy
pitfall 15).
