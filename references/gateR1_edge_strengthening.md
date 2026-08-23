# REGIME Edge-Strengthening result — fracdiff + session features (2026-08-23)

Question: does literature-backed feature engineering (fractional differentiation, AFML
Ch.5; session features) raise OOS AUC enough to make the barrier edge economically
viable under cost?

## 1. AUC comparison (1h, walk-forward logistic)

| feature set | n | folds | mean AUC | month-block 95% CI |
|---|---|---|---|---|
| BASE (7) | 125,718 | 0.642/0.632/0.643 | **0.6388** | [0.6271, 0.6444] |
| ENHANCED (10: +fracdiff_price, fracdiff_ret, session) | 125,718 | 0.681/0.667/0.655 | **0.6678** | [0.6554, 0.6737] |

**AUC gain = +0.029**, CIs non-overlapping → real, literature-backed edge strengthening.

## 2. But the decision-level test did NOT improve — it got worse

Same parallel variants (pre-registered) run on the ENHANCED feature set, vs risk-matched
constant (const Sharpe 3.56).

| | base features | ENHANCED features |
|---|---|---|
| **Variant M (meta-labeling)** | | |
| trades (20y) | 1,807 | 350 |
| Sharpe @ 0.02% | 2.66 | 1.75 |
| P(scheme worse) @ 0.02% | 0.731 | **0.903** |
| **Variant R (Re_t gate)** | | |
| P(scheme worse) @ 0.02% | 0.970 | **1.000** |

Both variants STILL FAIL the pass bar. Meta-labeling with richer features got WORSE
(Sharpe 2.66→1.75; P 0.731→0.903).

## 3. Interpretation — the central finding

**Raising classification AUC (+0.029) did not raise net-of-cost economics; it lowered it.**
Mechanism: the richer features made the meta-label OVER-CONFIDENT → it selected only
350 ultra-high-confidence trades (exposure 0.011) whose per-trade edge still doesn't
beat holding gold flat. More features → narrower, more confident, but not more profitable
selection.

**AUC is NOT the binding constraint.** The economic gate (cost × turnover × per-trade
edge) is. A "+3-5% AUC" is achievable and economically irrelevant here. This is why the
decision-level test is mandatory — a naive "just get a better AUC" reading of the
owner's question would have been a trap.

## 4. Scoping / overclaim rule
Scoped to: XAUUSD 1h, Target A barrier, these 10 features, logistic walk-forward,
long-only, meta-label θ=80th-pctile, cost 0–0.10%. A different model family (trees),
different meta-label threshold, or a genuinely different cost regime could behave
differently. What was tested: edge-strengthening via fracdiff+session does NOT rescue
the economics at 1h.

## Classification
AUC: real gain, filed as reusable feature engineering. Economics: still
**predictive but operationally irrelevant**. The ray is real but its source is not in
feature count — it's in the per-trade edge vs cost ratio, which neither AUC-raising
nor turnover-reduction alone has cleared.

## Library note (use-existing-libraries rule)
`fracdiff` has NO maintained PyPI library (`fracdiff` not on PyPI; `mfe` 0.0.4 is a
placeholder). Implemented the canonical AFML Ch.5 fixed-window fracdiff with numpy;
documented + unit-tested (6 tests: d=0 identity, d=1, causality, stationarity,
session). Added to the use-existing-libraries map as a from-scratch-with-documentation
case.
