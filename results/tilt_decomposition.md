# TILT Stage 1 — Decomposition: cost vs forfeited passive drift (XAUUSD 1h)

**Status:** Gate 0 result — **STOP, retire tilt.** Run 2026-08-23, pre-registered decision
rule applied mechanically.

## Question
For the corrected meta-labeling variant (326 trades / 20y, avg exposure 0.011), decompose the
Sharpe shortfall vs the risk-matched constant baseline into (a) cost paid on trades vs (b)
forfeited passive drift while flat.

## Decomposition (same 326 trades, same directions/sizes)

| scheme | Sharpe |
|---|---|
| scheme @ 0.02% (as run) | 1.621 |
| scheme @ 0% (cost-free) | 2.917 |
| always-invested (trades + const exposure while flat, net of cost) | 2.173 |
| risk-matched constant (fair baseline) | 3.560 |

total shortfall (const − scheme @ 0.02%) = 1.939
- **cost contribution = +1.296 (66.8% of shortfall)**
- **opp-cost contribution = −0.743 (−38.3%)**

## Interpretation — the premise is falsified

**Opportunity cost is NEGATIVE, not material.** Adding the risk-matched constant exposure
during the scheme's ~99% flat hours produced an always-invested Sharpe of **2.17 — LOWER than
the cost-free trades alone (2.92) AND lower than the constant baseline (3.56).**

This means the trades themselves **drag down** the constant baseline — they are net-negative
when overlaid on the drift, not additive-positive. The problem is NOT that the scheme forfeits
drift while flat (that would show as a large positive opp-cost component). The problem is that
the executed trades are, after cost, worse than just holding the drift — and cost is the
dominant, near-exclusive mechanism (66.8%).

## Decision rule (frozen) — applied mechanically
30% rule: opp-cost contribution must be ≥ 0.30 × total shortfall to proceed.
- opp-cost = **−0.743**; 30% × 1.939 = **+0.582**. −0.743 < +0.582.
- **VERDICT: STOP at Gate 0. Opportunity cost is not material (it is negative). The tilt
  overlay premise is falsified. Retire the tilt construction.**

## Why this is the right outcome (honest)
The TILT proposal's central premise — that a continuous exposure overlay would rescue the
strategy by not forfeiting gold's drift while flat — is contradicted by the decomposition. The
meta-labeling trades, run against the constant baseline, REDUCE its Sharpe. A `base·(1+tilt)`
construction would inherit the same problem: it leans on a signal whose trades are
cost-negative relative to the drift they ride. Building it would have been unmotivated
search — exactly what Rule 6 forbids.

The Stage-1-first design worked as intended: it killed the hypothesis cheaply, before any
construction effort, with a pre-registered mechanical gate.

## Caveat (overclaim rule)
The additive split assumes cost and drift effects are separable. The raw Sharpe values are the
honest numbers; an interaction between cost and drift may exist, but it would have to reverse a
−38% to +30% sign — implausible. Scoped to: XAUUSD 1h, meta-labeling variant, 0.02% cost, this
construction. A tilt overlay built on a DIFFERENT signal (e.g. the magnitude/vol axis, not
direction) is a different object and is NOT tested or killed here.

## Deliverable
As required by the TILT proposal, this decomposition is recorded regardless of outcome.
Retirement-log entry: TILT overlay — retired at Gate 0, Stage 1, opportunity cost negative.
