# SESSION RECAP — REGIME, MAGVOL, OFI (2026-08-23)

**Purpose:** complete handoff to a fresh session. Everything below is committed & pushed
(REGIME `mlotfiben/REGIME`, MAGVOL `mlotfiben/MAGVOL`, AUGUR VPS). Verify any number against
its cited source (grep), never cross-document.

---

## The big-picture finding of this program

After ~21+ pre-registered projects, the systematic conclusion is:
**No tested signal survives realistic cost on this universe.**
- **Direction** (REGIME): real accuracy edge (AUC 0.64 @ 1h), economically null under cost.
- **Magnitude/vol** (MAGVOL): real standalone, but redundant vs the HAR benchmark (not independent).
- **Flow** (OFI): the one thread that could break the pattern — order flow, not price. Currently
  PENDING re-test (see below).

The program's ONE validated, replicated positive remains **Re_t→vol rank-transfer** (p<0.001,
gold/BTC/ETH) — but MAGVOL showed it does not beat HAR as an *incremental* feature.

---

## Project 1 — REGIME: directional regime filter (XAUUSD)

**Repo:** `mlotfiben/REGIME` (all 8 commits pushed). **Status: closed / classified
"predictive but operationally irrelevant".**

| # | Experiment | Result |
|---|---|---|
| Gate D | synthetic controls | PASS (pos AUC 0.86, neg 0.497) |
| Gate R1 @1m | walk-forward AUC | real-but-tiny, decaying (0.519→0.507) |
| AUC vs timeframe | 1m/15m/30m/1h/4h | **peak 0.636 @ 1h**; toy sine 1.0, trend/chop 0.69 |
| Decision test 1h/30m | cost curve | **FAIL** — Sharpe 1.22@0.02% vs const 3.56 |
| Re_t gate / meta-labeling | parallel | both FAIL; meta-labeling best (326 trades, Sharpe 1.62) |
| Edge-strengthen (fracdiff+session) | AUC | 0.639→0.668 real, but economics got WORSE |
| Precision-Recall | PR frontier | precision flat ~0.45-0.48, never beats const |

**Central lesson: AUC is NOT the binding constraint; turnover×cost is.** A "+3-5% AUC" is
achievable and economically irrelevant.

**⚠ Critical bug found & fixed (owner caught the class in REGIME):** the trade simulators let a
trade re-enter at its exit bar → overlapping positions inflated returns. Fixed in all simulators;
every earlier verdict re-ran and STOOD (magnitudes corrected). **Lesson: audit trade/rebalance
accounting; run synthetic controls BEFORE real data.**

---

## Project 2 — MAGVOL: magnitude/vol diagnostic (18-asset watchlist)

**Repo:** `mlotfiben/MAGVOL`. **Status: closed (Q1 null → stop rule fired).**

- **Gate D synthetic controls:** PASS (pos ΔQLIKE +0.052 all seeds, neg ~0) — after fixing a
  log-vol QLIKE explosion and a weekend-zero-RV bug.
- **Q1 real test:** Re_t adds no incremental vol-forecast over HAR-RV on 18 assets (10/18 pos,
  coin-flip; all ΔQLIKE ~0.001; **GLD itself negative**). Stop rule (≥12/18) → retire.
- **Distinction:** program's Re_t→vol positive was Re_t *standalone*; this tested it *incremental
  over HAR* — it's a persistence proxy, not independent.

---

## Project 3 — OFI: order-flow lead-lag (PENDING, the live-trade decision)

**Status: PENDING re-test ~2026-09-13.** Original was INCONCLUSIVE (data corruption). This is the
one thread that could justify a live BOT (order flow ≠ price).

- **Collector restarted 2026-08-23 14:02 UTC** with the fixed `@bookTicker` code on the VPS
  (pid 3829692, fresh clean `ofi_log.jsonl`). Old contaminated log backed up
  (`ofi_log_contaminated...log`). Per owner: **drop >0.5%-gap records after 3 weeks, keep the rest.**
- **Frozen spec:** `~/AUGUR/PREREG_OFI_RETEST.md` (committed & pushed). Sequence:
  continuity audit → positive control → lead-lag h={1,3,6,12} → **cost gate (p<0.01)** →
  paper pilot → live. **"Lead-lag real" alone is NOT enough — must clear the cost gate** (the
  trap that killed every prior signal).
- **Decision (owner):** positive = trade live with BOT; negative = drop the live BOT program.

---

## TILT proposal (external feedback) — tested & killed at Gate 0

Claude's feedback proposed a continuous exposure-tilt overlay (base·(1+tilt), never flat). **Stage
1 decomposition showed the premise is falsified:** opportunity cost while flat is NEGATIVE
(−38%, not material); the trades themselves drag down gold's drift. Cost dominates (66.8%). Retired
at Gate 0 per the pre-registered 30% rule — a cheap, honest kill. Recorded in
`~/REGIME/results/tilt_decomposition.md`.

---

## Standing methodology (non-negotiable)

- Load `quant-project-methodology` + `vrpf-validation-harness` + `use-existing-libraries` for ANY
  trading/quant work.
- Pre-register before testing; simulation-first (synthetic pos/neg controls) before real data;
  multi-way verification; day-clustered stats; fair baseline = risk-matched constant exposure.
- **No live capital without a paper pilot.** Cost gate mandatory before any "tradable" claim.
- Verify every number against its source file (grep).
- **Library-first:** never hand-roll what a maintained library does (filterpy, arch, sklearn,
  pandas/scipy). Re_t must be the causal trailing-ATR version (centered→lookahead bug).

---

## NEXT SESSION — new goal: buy-and-hold monthly, long-run optimization

**New direction agreed (owner, 2026-08-23):** investable research on **monthly buy-and-hold** —
find the best long-run options using AUGUR/AETHER + simulation. This is portfolio/allocation
engineering (a different class than signal discovery).

**Open questions to resolve FIRST (owner to confirm):**
1. "Best options" = best *assets* (which watchlist names), best *allocation* (weights), or best
   *vehicle* (ETFs/index vs individual)?
2. Monthly cadence = DCA (fixed monthly), lump-sum timing, or rebalancing?
3. Horizon = 5/10/20+ years?

**Suggested defaults to start:** monthly DCA into a diversified allocation across the 22-asset
2030 watchlist, optimized on long-run risk-adjusted return, AUGUR/AETHER magnitude signals as a
tiebreaker. Validate with the SAME rigor: pre-register, risk-matched constant baseline, cost-sensitivity,
synthetic controls, day-clustered stats.

**Relevant existing assets:** 19-asset 30m watchlist (2018–2026), 22-asset 1h, raw 1m archives,
gold/BTC. AUGUR=AETHER core/collector (import AETHER, don't copy). Deploy via git pull, not scp.
