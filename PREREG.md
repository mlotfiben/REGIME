# PREREG — REGIME pre-registration (draft)

Status: **DRAFT — freeze after owner review.** Asset: XAUUSD 1m.

## Frozen spec
- **Asset / timeframe:** XAUUSD, **1m** (6,822,715 bars, 2004–2026). Raw CSV
  `~/machL/MarketPressure/data/raw/XAU_1m_data.csv`, read-only, cached to parquet.
- **Target (A, frozen):** does price hit +2·ATR before −1·ATR within the next H bars?
  Binary. **H = 60 bars (1 hour of 1m) — FROZEN by owner 2026-08-23.**
- **Purpose (owner):** ACADEMIC — establish whether the method works with statistically
  valid conclusions. Cost is a known operational killer (a directional model already
  died under costs); therefore the GATE tests OOS PREDICTIVE POWER (AUC), and cost is
  reported as a SECONDARY decision-level number, not the gate criterion.
- **Features (causal, frozen):** mom_s/mom_m/mom_l, trend_eff_N, breakout, vol_gate,
  kalman_slope (filterpy causal). **1m windows FROZEN: mom=(60,300,1440), eff=120,
  breakout=240, atr=20, vol=1440** (recalibrated per-freq, owner rule).
- **Cost/slippage:** single round-trip, correct units, applied on turnover ONLY (NOT
  per-bar — per-bar cost crushes Sharpe, taxonomy bug 22). At 1m this is dominant;
  measure the actual number before freeze.
- **Splits:** one train / one validation / one locked final test (~20%). Walk-forward
  with purge ≥ 1 bar + embargo.
- **Benchmarks:** persistence; risk-matched constant `w_const = mean(w_scheme)`;
  always-in.
- **Stats:** day-clustered (N = independent days), NOT per-bar. **N does NOT balloon
  with 1m bars** — day-cluster inference. Deflation for search counting all prior
  trials (≥ 20).
- **Pass criterion (provisional, freeze in review):** OOS predictive power (AUC) above
  the trivial benchmark, day-clustered CI excluding null, holds in both halves,
  winsorize-robust.

## A–D gate (mandatory before real data)
- **A. Spec:** every quantity precisely defined and frozen (above).
- **B. Leakage audit:** no return obs contributes simultaneously to feature + target;
  no-lookahead unit test (perturb future bar → past features bit-identical).
- **C. Effective-N:** independent units after overlap/cooldown/clustering; fix overlap
  structurally (cooldown). At 1m with H=60 the overlap is heavy — cooldown matters.
- **D. Synthetic controls:** pipeline recovers a known injected efficiency/barrier
  edge (positive) AND stays silent when the shape exists but effect is absent
  (negative).

## Failure action
Document the null; retire or scope precisely; do not soften the gate (taxonomy
pitfall 15).
