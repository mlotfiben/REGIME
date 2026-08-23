# REGIME Precision-Recall analysis (1h enhanced features) — CORRECTED 2026-08-23

Question (owner): controlling precision vs recall, if I choose ~20% recall how much
precision can I get, and where does precision/cost competition make things worse?

## ⚠ BUG FOUND & FIXED — supersedes earlier "breakthrough" impression

The first PR sweep printed a false positive: at low thresholds it showed
precision=0.52 and diff=+0.608 (scheme beating const) — that was an **overlapping-trade
artifact**. `simulate_trades` let the loop re-enter at the exact exit bar when
consecutive bars passed the threshold, creating adjacent/overlapping positions and
inflating both trade count and returns (17,002 nonzero return bars vs ~4,500 real
trades). FIX: after any exit, the next bar considered is j+1 (no immediate re-entry);
one open position at a time, one return per trade. Same bug existed in the earlier
`decision_backtest.py` / `decision_variants.py` simulations — see correction note below.

## Economic precision/recall (trade-level, unambiguous)
- win = completed trade with realized return > 0
- precision = fraction of trades that win
- recall = wins / total +2ATR-first label positives
- cost 0.02% round-trip per trade; vs risk-matched constant (const Sharpe 3.56)

## Full frontier (selected rows)

| thr | precision | recall | n_trades | schemeShp | diff |
|---|---|---|---|---|---|
| 0.085 | 0.480 | 0.662 | 12,965 | 2.08 | −1.479 |
| 0.096 | 0.478 | 0.586 | 11,526 | 2.54 | **−1.020 (best)** |
| 0.120 | 0.468 | 0.448 | 9,003 | 1.69 | −1.874 |
| 0.150 | ~0.45 | 0.29 | 6,091 | −0.19 | −3.75 |
| 0.167 | 0.449 | 0.212 | 4,441 | −0.48 | −4.04 |
| 0.200 | 0.450 | 0.106 | 2,208 | −0.60 | −4.16 |
| 0.256 | 0.403 | 0.019 | 454 | −1.11 | −4.67 |

## Key findings (answer to the owner's question)
1. **At ~20% recall: precision ≈ 0.45** (4,071 trades), but scheme Sharpe **−0.67** vs
   const 3.56 → **diff −4.23**. At 20% recall the scheme is far WORSE than holding gold flat.
2. **The diff is negative at EVERY recall** — the scheme never beats the risk-matched
   constant anywhere on the frontier. Best point is at high recall (~0.59, precision
   0.48, 11.5k trades): diff −1.02, still a loss.
3. **Precision is essentially flat (~0.45–0.48) across the whole recall range** — the
   model does NOT trade off precision against recall in a useful way. This is the 
   signature of a weak classifier: it can't lift precision by being more selective
   beyond the ~0.48 ceiling. The base rate is 0.10 (+2ATR-first); even the best
   precision (0.48) is a modest 4.8× lift that the per-trade edge cannot pay for.

## Conclusion (honest)
**There is no recall level where the precision/cost tradeoff makes the strategy
economically viable.** Precision maxes ~0.48 and cost kills it at every point. The
competition the owner hypothesized is real, but the model's precision ceiling is too
low to win it. This confirms, with the PR lens, the same verdict as every prior test:
**real-but-not-economically-exploitable.**

## ⚠ Correction to earlier results (integrity) — RE-RUN WITH FIX
The overlapping-trade bug also inflated the earlier `decision_backtest.py` (base test)
and `decision_variants.py` (Re_t gate + meta-labeling) results. Re-run with the fixed
simulator (no immediate re-entry), same protocol:

**Base decision test (corrected):**
| timeframe | n_trades | Sharpe @ 0.02% | P(scheme worse) |
|---|---|---|---|
| 30m | 7,874 | 0.029 | 0.9985 |
| 1h | 5,659 | 1.22 | 0.970 |
| | | | |
| (earlier buggy: 1h 2.57 / 0.829) | | | |

**Parallel variants (corrected, enhanced features):**
| variant | n_trades | Sharpe @ 0.02% | P(scheme worse) |
|---|---|---|---|
| R (Re_t gate) | 2,253 | −0.85 | 0.9993 |
| M (meta-labeling) | 326 | 1.62 | 0.9273 |
| (earlier buggy: R 1.14/0.970, M 2.66/0.731) | | | |

**Conclusion UNCHANGED:** every variant still FAILS to beat the risk-matched constant
under cost. The magnitudes were inflated; the verdict (no economically viable edge) is
robust. All numbers above are the honest, corrected values.

## Scoping (overclaim rule)
Scoped to: XAUUSD 1h, enhanced features, Target A barrier, logistic, cost 0.02%, PR
frontier thr ∈ [0.085, 0.256]. A different model, barrier, or cost could differ.
