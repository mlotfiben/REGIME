# REGIME — directional regime filter (XAUUSD H1)

Frozen project scaffold. Read `PLAN.md` FIRST (frozen, binding), then `PREREG.md`
(frozen pre-registration). Never edit these after results.

## Mission

Build a **transparent rule-based regime baseline** that labels each H1 bar as
`trend_up / trend_down / chop / expansion / stand_aside`, using only causal OHLC
features. The regime labels feed TWO goals, per owner decision 2026-08-23:

- (primary) the **efficiency / ATR-barrier target** — is the forward outcome
  predictable from the regime, evaluated OOS as predictive power, NOT as an
  abstain filter (Gate C1 in ~/INFLECTION already refuted "abstain in chop").
- (infrastructure) a **benchmark floor** that every future idea must beat.

## Non-negotiables

1. **Libraries, never re-invention.** Mandatory: `use-existing-libraries` skill +
   quant-project-methodology "use libraries" rule. Kalman = `filterpy`
   (`KalmanFilter`), FFT/indicators = numpy/scipy/pandas, vol = `arch`/statsmodels.
   Never hand-roll a Kalman/FFT/GARCH/indicator.
2. **Filter-only, never a smoother.** `filterpy.KalmanFilter` default is a causal
   filter. An RTS smoother uses future data → lookahead → forbidden. Force the
   causal `.predict()/.update()` path only.
3. **Features causal:** feature at bar t uses data ≤ t only. Label uses t+1..t+H.
   No-lookahead unit test: perturb a future bar → past features bit-identical.
4. **Fit at H1** (cache `data/xau_h1.parquet`, 125,793 bars). Never re-fit on raw 1m.
5. **Simulation-first:** synthetic positive control (inject a known
   efficiency/barrier edge → pipeline MUST recover it) before real data. Never tune
   feature params on real data.
6. **Pre-registration frozen:** `PLAN.md` + `gates/` immutable. Never edited after
   results. Fail action executed as written.
7. **Multi-way verification:** plots + tables + stats + source-grep. Never one number.
8. **Never delete raw data.** Read-only. Raw provenance in `data/raw_source.json`.
9. **Benchmark before complexity:** baseline = persistence / risk-matched constant /
   always-in. Signal only meaningful against baseline, not in isolation.
10. **Day-clustered statistics** (N = independent days), deflation for search.
    Count ALL 20+ prior trials as search width.

## Environment

- Venv: `~/SuperPowerfulAgentMarketAnalysis/.venv/bin/python` (pandas/numpy/scipy/
  pyarrow/statsmodels/filterpy confirmed).
- H1 cache: `~/INFLECTION/data/xau_h1.parquet` (125,793 rows, 2004–2026). REUSE —
  do not re-resample raw 1m.
- Raw (read-only): `~/machL/MarketPressure/data/raw/XAU_1m_data.csv` (6.82M 1m rows).

## Working style

Adversarial. Every decision: does this leak? overlap? chosen after seeing results?
feature actually new vs baseline? Verify every number against the file that produced
it (grep). A self-report is not evidence.
