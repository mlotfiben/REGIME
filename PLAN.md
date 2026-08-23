# PLAN — REGIME directional regime filter (XAUUSD 1m)

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
- XAUUSD **1m**, raw `~/machL/MarketPressure/data/raw/XAU_1m_data.csv` (6,822,715 rows,
  2004–2026), read-only. **Cached to `data/xau_1m.parquet`** after first load —
  never recompute from CSV per run.
- Owner decision 2026-08-23: **1m bars** chosen for statistical power (far more
  barrier-hit events, intrabar ATR-touch resolution that H1 cannot see).
- Cost/slippage: frozen in `PREREG.md` (single round-trip, correct units, applied on
  turnover only). At 1m cost is the dominant concern — must be honest and measured.

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

## 3. Target (frozen: A — owner decision 2026-08-23)
- **A (frozen):** does price hit +2·ATR before −1·ATR within the next H bars?
  (H frozen in PREREG.) Binary. At 1m, intrabar touches are visible.

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
