# PLAN — REGIME directional regime filter (XAUUSD H1)

Status: **DRAFT — to be frozen after owner review.** Nothing runs on real data
until this + `PREREG.md` + `gates/` are frozen and committed.

## 0. Question

Does a **transparent rule-based regime baseline** — built from causal OHLC features
(multi-horizon momentum, trend efficiency, channel breakout, volatility gate) —
predict the **efficiency / ATR-barrier target** of the forward window, OOS, beyond a
trivial benchmark? Evaluated as **predictive power**, not as an abstain filter.

### Scope control (owner rule, 2026-08-23)
- **Do NOT re-test "abstain in chop"** — Gate C1 (~/INFLECTION) already refuted it:
  stand-aside forfeits gold's positive drift, Sharpe −1.125 vs risk-matched const.
- **Do NOT re-litigate direction** — closed null across price + all news reps.
- The regime labels are the INPUT; the forward **efficiency/barrier** outcome is the
  testable object.

## 1. Data
- XAUUSD H1, `data/xau_h1.parquet` (125,793 bars, 2004–2026), reused from
  ~/INFLECTION. Never re-fit on raw 1m.
- Cost/slippage: frozen in `PREREG.md` (round-trip, single number, correct units).

## 2. Features (all causal, data ≤ t)
1. `mom_s / mom_m / mom_l` — normalized return over short / medium / long windows.
2. `trend_eff_N` — trend efficiency = |close[t] − close[t−N]| / Σ|close[i]−close[i−1]|.
3. `breakout` — close relative to prior N-bar high/low.
4. `vol_gate` — forecast vol (from `filterpy.KalmanFilter` on log-price or realized
   vol) relative to its own history; `kalman_slope` sign.

### Library-first (mandatory)
- Kalman: `filterpy.KalmanFilter`, causal `.predict()/.update()` ONLY. Never RTS
  smoother (lookahead).
- Indicators/FFT/roll: numpy / pandas / scipy.

## 3. Target (the testable object)
One of (frozen in PREREG):
- **Barrier hit:** does price hit +2·ATR before −1·ATR within next H bars?
- **Efficiency label:** is the next H-window high- or low-efficiency?
- **Regime classification:** trend / chop / expansion / calm.

## 4. Benchmarks (baseline before complexity)
- Persistence / momentum trivial model.
- Risk-matched constant exposure `w_const = mean(w_scheme)`.
- Always-in (gold's drift).

## 5. Protocol
Frozen in `PREREG.md`: one train / one validation / one locked final test; walk-forward
with purge + embargo; day-clustered statistics (N = independent days); deflation
counting all prior trials; no-lookahead + effective-N + synthetic positive/negative
control (A–D gate) before any real-data test.

## 6. Not re-litigating
Direction (closed), abstain-in-chop (C1 closed). Strictly: regime → forward
efficiency/barrier predictability.
