# Anlu (s38000)

Anlu is a pretrained reference-conditioned anomaly detector. The normal training
prefix supplies its reference input; model weights remain frozen during evaluation.
This integration supports univariate time series and returns one raw anomaly logit
per query timestamp. Higher scores indicate more anomalous observations.

## Install

From the TSB-AD checkout:

```bash
pip install -e .
pip install "torch>=2.1" "safetensors>=0.4" "huggingface_hub>=0.23"
```

The inference architecture is included in TSB-AD. It does not require a separate
Time-RCD checkout, training code, an external inference service, or a second
foundation-model download.

## Python interface

```python
import numpy as np
from TSB_AD.model_wrapper import run_Semisupervise_AD
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

# normal_prefix: [T_reference, 1]; query: [T_query, 1]
# In the official benchmark, query is the complete series, including its prefix.
scores = run_Semisupervise_AD(
    "Anlu", normal_prefix, query,
    **Optimal_Uni_algo_HP_dict["Anlu"],
    device="cuda:0",  # use "cpu" for CPU inference
)
if not isinstance(scores, np.ndarray):
    raise RuntimeError(scores)  # the existing TSB-AD dispatcher returns error strings
assert scores.shape == (len(query),)
```

Without a local checkpoint, the detector downloads `model.safetensors` from
[`thu-sail-lab/Anlu`](https://huggingface.co/thu-sail-lab/Anlu). The checkpoint
contains both encoders, the reference resampler, adapters, and output heads.
Its SHA256 is checked before loading:

```text
aad7c29293665fc9c6dbed7db2798cf9d1986ba4fc53694d1bd2c7e9f90388d8
```

For offline use, provide `checkpoint="/path/to/model.safetensors"` in the Python
call, or set the `ANLU_CHECKPOINT` environment variable. A containing directory
is also accepted. An explicit local path bypasses Hugging Face entirely.

## Official Eva350 evaluation

Download TSB-AD-U following the repository's dataset instructions. Use the
official 350-row evaluation list, not a different univariate file list.

```bash
export ANLU_CHECKPOINT=/absolute/path/to/model.safetensors  # optional local weights
mkdir -p results/anlu/score results/anlu/metrics
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python benchmark_exp/Run_Detector_U.py \
  --AD_Name Anlu \
  --dataset_dir /absolute/path/to/TSB-AD-U/ \
  --file_lsit Datasets/File_List/TSB-AD-U-Eva.csv \
  --score_dir results/anlu/score/ \
  --save_dir results/anlu/metrics/ \
  --save True
```

`--file_lsit` is the existing benchmark argument spelling. This command uses
the benchmark's full metric suite. Use a fresh output directory for a new run:
the upstream script skips score files that already exist.

## Fixed inference protocol

| Setting | Value |
|---|---|
| Checkpoint | s38000, frozen |
| Query window | 10,000 points |
| Stride | 1,500 points |
| Reference | first min(10,000, prefix length) points |
| Window batch size | 8; reduced on CUDA out-of-memory |
| Normalization | full-query mean and population standard deviation shared by query and reference; std floor 1e-4 |
| Output | anomaly-class logit minus normal-class logit |
| Window aggregation | equal mean over covering windows; backward tail window contributes only previously uncovered suffix |

The detector never receives anomaly labels or a dataset/file identifier. It
performs no gradient updates, threshold tuning, or per-dataset parameter search.
Evaluation labels are used only by TSB-AD's metric functions. The normal-prefix
requirement is why Anlu is registered in `Semisupervise_AD_Pool`.

## Verification

```bash
python -m unittest discover -s tests -p test_anlu.py
```

Release verification additionally compares all 350 full-series score vectors
against the original s38000 implementation. The included
`benchmark_exp/anlu_results/Uni_Anlu_VUSPR.csv` reports VUS-PR only; it does not
claim results for the other benchmark metrics. VUS-PR uses the original
250-threshold metric and the current benchmark's unmodified
`find_length_rank(data, rank=1)` per-series window rule. Historical scores that
use a different metric-window fallback rule should not be compared as though
they used this same protocol.
