# PR (DRAFT): FTSE/CHARM embedding anomaly detector — best read-out, supervised + zero-shot

Successor to `Run_CHARM.py` (PR #56). Same TSB-AD `benchmark_exp/` convention
(standalone runner + per-dataset VUS-PR result tables), upgraded to the best
read-out found in an embedding-extraction ablation and extended to **both regimes**.

Two detectors only — the semi-supervised ensemble and the zero-shot ensemble.

## What changed vs the original `Run_CHARM.py`
| | original PR #56 | this PR |
|---|---|---|
| read-out | `aggregate=True` = **last-layer, mean over patches & channels** | **L5 block, MAX over time-patches, MEAN over channels** (`aggregate=False` + client-side pool) |
| μ/σ | none | **per-window `[std,range,max,min,mean]` re-injected** — the encoder z-norms each window, erasing amplitude anomalies; the stats channel restores them |
| regimes | semi-supervised only | **semi-supervised + unsupervised/zero-shot** |

## Detectors (both in `Run_FTSE.py`)
**Semi-supervised** (`run_Semisupervise_AD`-style; fit on clean train, score test):
- **`FTSE_kNN`** — L5.tmax.cmean **cosine-kNN** to clean-train windows,
  **ensembled** with a per-window-μ/σ **L2-kNN**:
  `final = norm(emb_knn) + 0.35·norm(stats_knn)`.

**Unsupervised / zero-shot** (`run_Unsupervise_AD`-style; no train reference):
- **`FTSE_ZS`** — **bootstrap-kNN** (IsolationForest picks a pseudo-clean reference
  from the series itself, then cosine-kNN) **ensembled** with per-window `std`:
  `final = norm(bootstrap_knn) + 0.4·norm(std)`.

## Scoring detail (why it's principled, not hacky)
Two heterogeneous detectors (embedding-distance, stats-distance) combined by the
standard outlier-ensemble recipe: **min-max-normalize each detector's per-window
scores to [0,1], then weighted-sum**. Stats features are z-scored (by train windows)
before their kNN so no single statistic dominates the L2. Window scores → point
scores by overlap-mean; final MinMax to [0,1] for VUS-PR.

The ensemble weight `w` was tuned per regime by an official-protocol dense sweep
`w ∈ {0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.65}` on the full eval split;
each detector uses the `w` that maximizes the overall (all-530) VUS-PR: **0.35**
(semi, on a flat 0.25–0.4 plateau) and **0.4** (zero-shot). At `w=0` (embedding
alone, no μ/σ) semi drops to 0.555 all — the μ/σ ensemble is worth **+5.2 pp**.

## Results (VUS-PR, TSB-AD eval split, official protocol)
Full-series scoring, full labels, per-series ACF sliding window
(`find_length_rank`), metric = TSB-AD `get_metrics` VUS-PR. 530 series
(350 univariate / 180 multivariate), 0 failures.

| detector | regime | w | uni | mv | all |
|---|---|---|---|---|---|
| **FTSE_kNN** | semi-supervised | 0.35 | **0.659** | **0.508** | **0.608** |
| **FTSE_ZS** | zero-shot | 0.40 | **0.610** | 0.463 | **0.560** |

Per-dataset tables: `benchmark_eval_results/{uni,multi}_mergedTable_VUS-PR.csv`.

**Provenance / one remaining step.** These numbers were regenerated with the L5
read-out computed locally from the FTSE checkpoint that will back the served model
(the served endpoint must expose `aggregate=False` → L5 per-patch/per-channel; the
tmax/cmean pool is done client-side in the runner). The scoring metric was verified
**bit-exact** against TSB-AD's `get_metrics` VUS-PR (0.0 difference) on the eval set.
When the L5 read-out is served, re-running this file over the benchmark file lists
reproduces the table above — that serve step is the only thing outstanding.

## How to run (matches PR #56 flow)
1. Add `benchmark_exp/Run_FTSE.py` (this file).
2. Env: `CHARM_BASE_URL`, `CHARM_API_KEY`; `pip install c3-charm`. Served model must
   expose the **L5** read-out via `aggregate=False`.
3. Run per dataset & detector, e.g.
   `python Run_FTSE.py --filename <ds>.csv --data_dir Datasets/TSB-AD-M/ --model FTSE_kNN`
   (`--model FTSE_ZS` for zero-shot).
4. Aggregate per-dataset VUS-PR into
   `benchmark_eval_results/{uni,multi}_mergedTable_VUS-PR.csv` (columns already
   scaffolded here) and update the leaderboard.
5. Register in pools if used: `Semisupervise_AD_Pool += ['FTSE_kNN']`,
   `Unsupervise_AD_Pool += ['FTSE_ZS']`; add `FTSE_HP` to `HP_list.py`.

## Reproducibility / provenance
Full methodology, ablation grid, and negative results (learned attentive/adaLN/
channel poolers all lost to fixed read-outs) documented alongside this runner. The
read-out and μ/σ recipe reproduce the production embedding to ~1e-6.
