# CHARM anomaly detector

`Run_CHARM.py` adds two detectors built on CHARM time-series embeddings, served through the CHARM API.

| detector | setting | idea |
|---|---|---|
| `CHARM_kNN` | semi-supervised | kNN distance to the clean train windows, at several window sizes |
| `CHARM_ZS` | zero-shot | kNN distance to a pseudo-clean reference chosen from the series itself, plus per-window std |

`CHARM_kNN` combines two sub-detectors. Each is run at several window sizes, every scale's score is z-scored, and the scales are merged by element-wise max ("anomalous at any scale"):

- embedding: channel-mean-pooled embeddings, cosine kNN, windows 64 / 128 / 256
- statistics: per-window `[std, range, max, min, mean]`, L2 kNN, windows 16 to 512 (window 128 only for multivariate series with at most 3 channels)

The final score is the sum of the two. Z-scoring each scale before the max matters: skipping it costs about 6pp.

## Results

VUS-PR on the TSB-AD eval set (350 uni, 180 mv, 530 total, no failures):

| detector | uni | mv | all |
|---|---|---|---|
| `CHARM_kNN` | 0.678 | 0.539 | 0.631 |
| `CHARM_ZS` | 0.606 | 0.459 | 0.556 |

Per-series scores are in `benchmark_eval_results/CHARM_{uni,multi}_mergedTable_VUS-PR.csv`.

## Run

```
pip install c3-charm
export CHARM_BASE_URL=... CHARM_API_KEY=...
python Run_CHARM.py --filename <series>.csv --data_dir Datasets/TSB-AD-U/ --model CHARM_kNN
python Run_CHARM.py --filename <series>.csv --data_dir Datasets/TSB-AD-U/ --model CHARM_ZS
```

The protocol is the same as `Run_Detector_U.py`: fit on `data[:train_len]`, score the full series, evaluate on the full labels. `CHARM_kNN` makes requests at 3 window sizes per series. Requests respect the server's 500,000 time-point limit and are retried with a smaller batch on transient server errors.

## Checking the numbers through the API

VUS-PR from `Run_CHARM.py` through the API versus the table:

| series | `CHARM_kNN` API / table | `CHARM_ZS` API / table |
|---|---|---|
| NAB_014 | 0.904 / 0.904 | 0.504 / 0.560 |
| MSL_143 | 0.997 / 0.997 | 0.817 / 0.797 |
| IOPS_267 | 0.310 / 0.310 | 0.354 / 0.362 |
| YAHOO_741 | 0.616 / 0.615 | 0.574 / 0.574 |
| YAHOO_579 | 1.000 / 1.000 | |
| YAHOO_649, YAHOO_689 | 1.000 / 1.000, 0.006 / 0.006 | |
| Genesis (mv) | 0.930 / 0.930 | |
| Daphnet (mv) | 0.387 / 0.388 | |

`CHARM_kNN` reproduces the table. `CHARM_ZS` picks its reference windows with an IsolationForest (averaged over 20 forests to reduce variance), and tiny numeric differences in the embeddings can still change which windows are selected, so per-series scores can differ by a few points. The 530-series average is stable.
