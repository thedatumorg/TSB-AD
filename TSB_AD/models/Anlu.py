"""Anlu s38000: frozen, reference-conditioned univariate anomaly detection.

The normal training prefix is an input reference, not a fine-tuning dataset.
Inference returns one raw anomaly logit per query timestamp (higher is worse).
"""
import hashlib
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

MODEL_REPO = "thu-sail-lab/Anlu"
WEIGHTS_SHA256 = "aad7c29293665fc9c6dbed7db2798cf9d1986ba4fc53694d1bd2c7e9f90388d8"


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _series(data, name):
    value = np.asarray(data, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or value.shape[1] != 1 or value.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty univariate array, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN, infinity, or values outside float32 range")
    return value


@lru_cache(maxsize=2)
def _download(repo_id, revision, local_files_only):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Anlu downloads require huggingface_hub; or set ANLU_CHECKPOINT to a local weights file") from exc
    return hf_hub_download(repo_id, "model.safetensors", revision=revision, local_files_only=local_files_only)


@lru_cache(maxsize=1)
def _load_model(path, device):
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError("Install safetensors to load the Anlu inference checkpoint") from exc
    from .anlu.model import TotoFlamingoSharedResamplerModel

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Anlu checkpoint SHA256 mismatch: expected the published s38000 weights")
    config = SimpleNamespace(
        encoder=SimpleNamespace(d_model=512, d_proj=256, patch_size=16, num_layers=8,
                                num_heads=8, d_ff_dropout=0.1, use_rope=True,
                                activation="gelu", num_features=1, num_reference_tokens=1),
        dropout=0.1, mgm_memory_tokens=32, mgm_resampler_layers=1, flamingo_layers="2,4,6,7",
    )
    model = TotoFlamingoSharedResamplerModel(config)
    model.load_state_dict(load_file(path, device="cpu"), strict=True)
    return model.to(torch.device(device)).eval().requires_grad_(False)


def _windows(length, window_size, stride):
    """Overlap averaging with a backward tail window covering only unseen points."""
    width = min(length, window_size)
    starts = range(0, length - width + 1, stride)
    result = [(start, start + width, start, start + width) for start in starts]
    covered = result[-1][1]
    if covered < length:
        result.append((length - width, length, covered, length))
    return result


class Anlu:
    """Frozen Anlu detector; ``fit`` stores an unlabeled reference prefix.

    Parameters follow the s38000 Full-series evaluation protocol. A local
    ``model.safetensors`` file (or its containing directory) can be supplied via
    ``checkpoint`` or ``ANLU_CHECKPOINT``. Otherwise weights are downloaded from
    the model repository and verified against their fixed SHA256.

    This release supports univariate series. No labels, filenames, per-dataset
    settings, threshold search, or gradient updates enter the detector.
    """

    def __init__(self, window_size=10000, stride=1500, batch_size=8,
                 reference_length=10000, device="auto", checkpoint=None,
                 model_repo=MODEL_REPO, revision="main", local_files_only=False):
        self.window_size = _positive_int("window_size", window_size)
        self.stride = _positive_int("stride", stride)
        self.batch_size = _positive_int("batch_size", batch_size)
        self.reference_length = _positive_int("reference_length", reference_length)
        if self.stride > self.window_size:
            raise ValueError("stride must not exceed window_size (every timestamp must be covered)")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.checkpoint = checkpoint
        self.model_repo = model_repo
        self.revision = revision
        self.local_files_only = bool(local_files_only)
        self.reference_ = None

    def fit(self, X, y=None):
        """Store the first ``reference_length`` normal points; do not train."""
        self.reference_ = _series(X, "reference")[:self.reference_length].copy()
        return self

    def _model(self):
        checkpoint = self.checkpoint or os.environ.get("ANLU_CHECKPOINT")
        if checkpoint is None:
            checkpoint = _download(self.model_repo, self.revision, self.local_files_only)
        path = Path(checkpoint).expanduser()
        if path.is_dir():
            path = path / "model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"Anlu checkpoint not found: {path}")
        return _load_model(str(path.resolve()), str(self.device))

    @staticmethod
    def _forward_batch(model, series, reference, windows):
        size = len(windows)
        width = max(end - start for start, end, _, _ in windows)
        query = series.new_zeros((size, width, 1))
        mask = torch.zeros((size, width), dtype=torch.bool, device=series.device)
        for i, (start, end, _, _) in enumerate(windows):
            query[i, :end-start] = series[start:end]
            mask[i, :end-start] = True
        ref = reference[None].expand(size, -1, -1)
        ref_mask = torch.ones((size, len(reference)), dtype=torch.bool, device=series.device)
        return model(query, mask, ref, ref_mask, return_dict=True)["anomaly_logits"]

    @torch.inference_mode()
    def decision_function(self, X):
        if self.reference_ is None:
            raise RuntimeError("Call fit(normal_training_prefix) before decision_function(query)")
        query = _series(X, "query")
        model = self._model()
        series = torch.as_tensor(query, dtype=torch.float32, device=self.device)
        reference = torch.as_tensor(self.reference_, dtype=torch.float32, device=self.device)
        # Exactly the historical query_series normalization: one set of full-query
        # statistics is shared by the query and the reference before windowing.
        mean = series.mean(dim=0, keepdim=True)
        std = series.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
        series = (series - mean) / std
        reference = (reference - mean) / std
        windows = _windows(len(series), self.window_size, self.stride)
        scores = torch.zeros(len(series), dtype=torch.float32, device=self.device)
        counts = torch.zeros_like(scores)
        batch_size = self.batch_size
        cursor = 0
        while cursor < len(windows):
            chosen = windows[cursor:cursor + batch_size]
            try:
                logits = self._forward_batch(model, series, reference, chosen)
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                continue
            for i, (start, _, collect_start, collect_end) in enumerate(chosen):
                scores[collect_start:collect_end] += logits[i, collect_start-start:collect_end-start]
                counts[collect_start:collect_end] += 1
            cursor += len(chosen)
            del logits
        if not bool((counts > 0).all()):
            raise RuntimeError("Anlu inference left timestamps uncovered")
        output = (scores / counts).cpu().numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("Anlu produced nonfinite anomaly scores")
        self.decision_scores_ = output
        return output
