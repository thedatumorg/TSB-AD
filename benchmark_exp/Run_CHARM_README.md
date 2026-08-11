# PR (DRAFT): CHARM embedding anomaly detector — best read-out + μ/σ, supervised + zero-shot

Successor to the original `Run_CHARM.py` (PR #56). Same TSB-AD `benchmark_exp/` convention
(standalone runner + per-dataset VUS-PR tables), upgraded with the best read-out and the
μ/σ ensemble from an embedding-extraction study, plus a multivariate no-channel-pooling variant.

## Detectors (`Run_CHARM.py`)
**Semi-supervised** (clean train split):
- **`CHARM_kNN`** *(best overall)* — L5 max-over-time / **mean-over-channel** embedding
  cosine-kNN to clean-train windows, ensembled with a per-window `[std,range,max,min,mean]`
  L2-kNN: `norm(emb) + 0.35·norm(stats)` (min-max).
- **`CHARM_kNN_nopool`** *(best on multivariate; use when channel count is high)* — same, but
  the embedding read-out does **not pool channels** (adaptive: per-channel cosine max-fused for
  C≤20, else concat + per-dimension standardize), ensembled with μ/σ by **z-score-sum**
  (parameter-free).

**Unsupervised / zero-shot** (no train reference):
- **`CHARM_ZS`** — bootstrap-kNN (IsolationForest picks a pseudo-clean reference from the series
  itself) ensembled with per-window `std`: `norm(bknn) + 0.4·norm(std)`.

## Why μ/σ + read-out
The original `Run_CHARM.py` used `aggregate=True` (last-layer mean over patches **and** channels).
This uses `aggregate=False` and pools client-side as **max over time-patches** on **L5**. The
encoder instance-normalizes each window, erasing amplitude / level-shift anomalies from the
embedding; the per-window statistics (no model call) recover them (**~+7 pp VUS-PR**). Combine
the two detectors by normalizing each and adding.

## Results — VUS-PR, TSB-AD eval, **stride-1**, official protocol (350 uni / 180 mv / 530 all)
| detector | regime | uni | mv | all |
|---|---|---|---|---|
| **CHARM_kNN** | semi | **0.659** | 0.506 | **0.607** |
| **CHARM_kNN_nopool** | semi (high-C) | 0.645 | **0.515** | 0.601 |
| **CHARM_ZS** | zero-shot | 0.615 | 0.463 | 0.560 |
| *(orig Run_CHARM, last-layer mean, no μ/σ)* | semi | — | — | ~0.499 |

Per-dataset tables: `benchmark_eval_results/CHARM_{uni,multi}_mergedTable_VUS-PR.csv`.

**Notes on the variants.** `CHARM_kNN` (min-max ensemble) is the best on the overall leaderboard.
Not pooling channels helps **only on multivariate** (and grows with channel count — negligible at
C≤3, sizeable at C≥20, large at C>60); since univariate is 66% of the eval, `CHARM_kNN_nopool` is
slightly below on "all" but ahead on mv, so it is offered as the recommended detector **when the
channel count is high**. A parameter-free **z-score-sum** combiner matches min-max on "all" (0.602)
and wins on mv — it is what `CHARM_kNN_nopool` uses.

## Provenance / one remaining step
Numbers were regenerated with the L5 read-out from the CHARM checkpoint that will back the served
model (the served endpoint must expose `aggregate=False` → L5 per-patch/per-channel; tmax/channel
pooling is client-side in the runner). The scoring metric was verified **bit-exact** against
TSB-AD's `get_metrics` VUS-PR. Serving the L5 read-out is the only outstanding step.

## How to run
```
python Run_CHARM.py --filename <ds>.csv --data_dir Datasets/TSB-AD-M/ --model CHARM_kNN
# --model CHARM_kNN_nopool  (multivariate, high channel count)
# --model CHARM_ZS          (zero-shot)
```
Env: `CHARM_BASE_URL`, `CHARM_API_KEY`; `pip install c3-charm`.
