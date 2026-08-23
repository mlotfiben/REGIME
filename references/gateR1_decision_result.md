# REGIME Decision-Level result — does the barrier edge survive costs? (2026-08-23)

Verdict: **FAIL at both timeframes.** The regime barrier signal is economically
meaningless after realistic cost, measured against the FAIR baseline (its own
risk-matched constant exposure). The accuracy gate (AUC 0.59–0.64) is real; the
decision-level gate fails — the same conclusion the owner reached empirically.

Source: `engine/decision_backtest.py`, `results/`. Pre-registered in `PREREG.md`
"DECISION-LEVEL STAGE" (2026-08-23).

## Strategy (long-only, frozen)
Walk-forward logistic p = P(hit +2ATR before −1ATR in next H bars); enter long when
p ≥ 80th-pctile of TRAINING p (causal per fold). Exit at +2ATR / −1ATR / time(close[t+H]).
One round-trip cost per trade. Returns fractional, comparable to baseline.

## Fair baseline (the crux)
RISK-MATCHED CONSTANT exposure `w_const = mean(position)` held constantly (captures
gold's drift, zero turnover). Always-in reported for context only (different avg risk).

## Results — cost-sensitivity curve

### 30m (n_trades 9,502, exposure 0.204)
| cost% | scheme Sharpe | const Sharpe | always-in | schemeNet |
|---|---|---|---|---|
| 0.000 | 11.36 | 3.55 | 3.55 | +2.01 |
| 0.010 | 6.01 | 3.55 | 3.55 | +1.06 |
| 0.020 | 0.61 | 3.55 | 3.55 | +0.11 |
| 0.050 | −15.45 | 3.55 | 3.55 | −2.74 |
| 0.100 | −39.25 | 3.55 | 3.55 | −7.50 |

day-cluster bootstrap P(scheme worse) at 0.02% = **0.996** (CI of diff [−0.071, −0.009])

### 1h (n_trades 7,139, exposure 0.213)
| cost% | scheme Sharpe | const Sharpe | always-in | schemeNet |
|---|---|---|---|---|
| 0.000 | 10.22 | 3.56 | 3.56 | +1.90 |
| 0.010 | 6.41 | 3.56 | 3.56 | +1.19 |
| 0.020 | 2.57 | 3.56 | 3.56 | +0.48 |
| 0.050 | −8.95 | 3.56 | 3.56 | −1.67 |
| 0.100 | −26.90 | 3.56 | 3.56 | −5.24 |

day-cluster bootstrap P(scheme worse) at 0.02% = **0.829** (CI of diff [−0.046, +0.017])

## Pass-bar check (pre-committed)
PASS requires scheme net Sharpe > risk-matched const net Sharpe at measured cost AND
day-clustered bootstrap one-sided p < 0.05 that the Sharpe difference (scheme − const)
is positive, Bonferroni across 1h & 30m.

- 30m: scheme 0.61 vs const 3.55 at 0.02% → WORSE; P(worse)=0.996 → **FAIL**.
- 1h: scheme 2.57 vs const 3.56 at 0.02% → WORSE; P(worse)=0.829 → **FAIL** (CI crosses 0).
- Both: **FAIL.**

## Why (mechanism, consistent with the program's known turnover trap)
- The signal is real but LOW-VOLUME/high-TURNOVER: 7k–9.5k trades over ~20 years at
  ~0.20 average exposure. Each trade pays a full round-trip cost.
- The barrier edge (AUC 0.52–0.64) is too thin to pay 9,500 round-trips.
- Gross Sharpe is huge (10–11) but collapses the moment any realistic cost applies —
  the cost-sensitivity curve flips the scheme below the constant baseline almost
  immediately (already worse at 0.01% for 30m; at 0.02% for 1h the CI already crosses 0).
- This is EXACTLY the owner's empirical finding: a directional model that "dies with
  the costs." The academic gate confirms the method is statistically real but the
  economic gate kills it — now measured, not asserted.

## Live-test answer
**NO live test.** The decision-level gate FAILED net-of-cost against the fair baseline.
A live test on an unvalidated, cost-negative signal would be premature (the program's
repeated mistake). Even the accuracy gate (AUC 0.64) does not survive costs, so there
is nothing to take live.

## Classification
**predictive but operationally irrelevant** (accuracy real, economic null). FILED.
Consistent with the program's established finding: gold directional structure is weak,
sparse, and not economically exploitable on OHLC alone at any tested timeframe.

## Scoping (overclaim rule)
Scoped to: XAUUSD OHLC, Target A (+2ATR/−1ATR barrier), these 7 features, logistic
walk-forward, long-only at θ=80th-pctile, cost curve 0–0.10%. A different target
(magnitude/vol, not directional), different entry rule, or a genuinely lower-cost
execution could behave differently. What was tested shows the directional barrier edge
does not survive realistic cost.
