from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import time
from pathlib import Path

for _env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env_name, "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.slidingWindows import find_length_rank


EPS = 1e-9


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _as_2d_float(data: np.ndarray) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"expected a 1D/2D time series, got shape {x.shape}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _flatten_signal(data: np.ndarray) -> np.ndarray:
    x = _as_2d_float(data)
    if x.shape[1] == 1:
        return x[:, 0].astype(np.float64, copy=False)
    return np.mean(x, axis=1, dtype=np.float64)


def _zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    scale = float(np.nanstd(x) + EPS)
    return ((x - float(np.nanmean(x))) / scale).astype(np.float32)


def _effective_patch_size(train_len: int, requested: int) -> int:
    requested = max(4, int(requested))
    if train_len >= requested:
        return requested
    if train_len >= 8:
        return max(4, int(train_len))
    return int(train_len)


def _create_patch_tensor(data: np.ndarray, patch_size: int, stride: int) -> torch.Tensor:
    array = _as_2d_float(data)
    if len(array) < patch_size:
        raise ValueError(f"data length {len(array)} is shorter than patch_size {patch_size}")
    stride = max(1, int(stride))
    if array.shape[1] == 1:
        values = np.ascontiguousarray(array[:, 0])
        windows = sliding_window_view(values, int(patch_size))[::stride]
        return torch.from_numpy(np.ascontiguousarray(windows)).float().unsqueeze(1).contiguous()
    base = torch.as_tensor(np.ascontiguousarray(array.T), dtype=torch.float32)
    return base.unfold(1, int(patch_size), stride).permute(1, 0, 2).contiguous()


def _patch_scores_to_points(patch_scores: np.ndarray, patch_size: int, num_points: int) -> np.ndarray:
    s = np.nan_to_num(np.asarray(patch_scores, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    kernel = np.ones(int(patch_size), dtype=np.float32)
    sums = np.convolve(s, kernel, mode="full")[: int(num_points)]
    counts = np.convolve(np.ones_like(s), kernel, mode="full")[: int(num_points)]
    out = np.divide(sums, counts, out=np.zeros(int(num_points), dtype=np.float32), where=counts != 0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class _RevIN1d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5, min_sigma: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.min_sigma = float(min_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        sigma = (var + self.eps).sqrt().clamp_min(self.min_sigma)
        return (x - mu) / sigma


class _PatchEncoder(nn.Module):
    """PaAno clean CNN encoder used by the Wk1/Wk4 PaAno+PAI recipe."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 64,
        projection_dim: int = 256,
        use_revin: bool = True,
        include_public_heads: bool = False,
    ) -> None:
        super().__init__()
        layers = [128, 256, 128, int(embed_dim)]
        kernels = [7, 5, 3, 3]
        self.revin = _RevIN1d(int(in_channels)) if use_revin else None
        blocks = []
        prev = int(in_channels)
        for out_channels, kernel in zip(layers, kernels):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(prev, int(out_channels), kernel_size=int(kernel), padding=int(kernel) // 2, bias=False),
                    nn.BatchNorm1d(int(out_channels)),
                    nn.ReLU(inplace=True),
                )
            )
            prev = int(out_channels)
        self.convblocks = nn.ModuleList(blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.projection_head = nn.Sequential(
            nn.Linear(int(embed_dim), int(projection_dim)),
            nn.ReLU(),
            nn.Linear(int(projection_dim), int(projection_dim)),
        )
        self.mask_prediction_head = None
        self.classification_head = None
        if include_public_heads:
            self.mask_prediction_head = nn.Sequential(
                nn.Linear(int(embed_dim), 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, int(embed_dim)),
            )
            self.classification_head = nn.Linear(int(embed_dim) * 2, 1)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x)
        for block in self.convblocks:
            x = block(x)
        return self.gap(x).flatten(1)

    def projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.projection_head(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class _EMATeacher:
    def __init__(self, student: nn.Module, tau_start: float, tau_end: float, total_steps: int):
        self.teacher = copy.deepcopy(student)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.total_steps = max(1, int(total_steps))
        self.step = 0

    def _tau(self) -> float:
        t = min(self.step, self.total_steps) / float(self.total_steps)
        return self.tau_end - (self.tau_end - self.tau_start) * (0.5 * (1.0 + math.cos(math.pi * t)))

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        self.step += 1
        tau = self._tau()
        for teacher_param, student_param in zip(self.teacher.parameters(), student.parameters()):
            teacher_param.data.mul_(tau).add_(student_param.data, alpha=1.0 - tau)
        for teacher_buffer, student_buffer in zip(self.teacher.buffers(), student.buffers()):
            teacher_buffer.copy_(student_buffer)

    @torch.no_grad()
    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher.embedding(x)

    @torch.no_grad()
    def projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.teacher.projection(h)


def _triplet_grad_scale(x: torch.Tensor, ratio: float = 0.01) -> torch.Tensor:
    return x.detach() + (x - x.detach()) * float(ratio)


def _sample_temporal_positive(indices: torch.Tensor, total: int, radius: int, device: torch.device) -> torch.Tensor:
    offsets = torch.tensor([*range(-radius, 0), *range(1, radius + 1)], device=device, dtype=torch.long)
    candidates = indices.unsqueeze(1) + offsets.unsqueeze(0)
    valid = (candidates >= 0) & (candidates < int(total))
    noise = torch.rand(candidates.shape, device=device)
    choice_score = torch.where(valid, noise, torch.full_like(noise, -1.0))
    choice = choice_score.argmax(dim=1)
    out = candidates.gather(1, choice.unsqueeze(1)).squeeze(1)
    no_valid = valid.sum(dim=1) == 0
    if no_valid.any():
        out[no_valid] = indices[no_valid]
    return out


class _PatchDataset(Dataset):
    def __init__(self, patches: torch.Tensor):
        self.patches = patches.float().contiguous()
        self.indices = torch.arange(int(patches.shape[0]), dtype=torch.long).view(-1, 1)

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.patches[index], self.indices[index]


def _train_public_paano_encoder(
    train_patches: torch.Tensor,
    in_channels: int,
    hp: dict,
    device: torch.device,
) -> tuple[_PatchEncoder, dict]:
    model = _PatchEncoder(
        in_channels=in_channels,
        embed_dim=int(hp.get("embed_dim", 64)),
        projection_dim=int(hp.get("projection_dim", 256)),
        use_revin=bool(hp.get("use_revin", False)),
        include_public_heads=True,
    ).to(device)
    model.train()

    total = int(train_patches.shape[0])
    batch_size = max(2, min(int(hp.get("batch_size", 512)), total))
    num_iters = max(1, int(hp.get("num_iters", 200)))
    lr = float(hp.get("lr", 1e-4))
    positive_radius = int(hp.get("positive_radius", 2))
    pretext_step = int(hp.get("pretext_step", 64))
    num_rand_patches = int(hp.get("num_rand_patches", 5))
    temperature = float(hp.get("temperature", 1.0))
    triplet_margin = float(hp.get("triplet_margin", 0.5))

    loader = DataLoader(
        _PatchDataset(train_patches),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )
    train_patches_device = train_patches.to(device=device, non_blocking=True)

    mask_param_ids = {id(param) for param in model.mask_prediction_head.parameters()}
    base_params = [param for param in model.parameters() if id(param) not in mask_param_ids]
    optimizer = torch.optim.AdamW(
        [{"params": base_params, "lr": lr, "base_lr": lr, "final_lr": lr / 10.0}],
        weight_decay=1e-4,
    )
    criterion_pretext = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device), reduction="none")

    def cosine_lr(iteration: int) -> float:
        t = min(iteration, num_iters)
        return (lr / 10.0) + (lr - (lr / 10.0)) * 0.5 * (1.0 + math.cos(math.pi * t / num_iters))

    iteration = 0
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    last_loss = float("inf")

    while iteration < num_iters:
        for batch_data, batch_indices in loader:
            if iteration >= num_iters:
                break
            iteration += 1
            for param_group in optimizer.param_groups:
                param_group["lr"] = cosine_lr(iteration)

            anchors = batch_data.to(device=device, non_blocking=True)
            batch_indices = batch_indices.squeeze(-1).to(device=device, non_blocking=True).long()
            cur_batch = int(anchors.shape[0])
            mu = 1.0 if int(anchors.shape[1]) != 1 else 10.0

            pos_indices = _sample_temporal_positive(batch_indices, total, positive_radius, device)
            positives = train_patches_device.index_select(0, pos_indices)

            pretext_active = iteration < (num_iters / 10.0)
            if pretext_active:
                target = batch_indices - pretext_step
                pre_mask = (target >= 0) & (target < total)
                target_clamped = target.clamp(0, total - 1)
                pretext_patches = train_patches_device.index_select(0, target_clamped).clone()
                if (~pre_mask).any():
                    pretext_patches[~pre_mask] = 0
                all_patches = torch.cat([anchors, positives, pretext_patches], dim=0)
            else:
                pre_mask = None
                all_patches = torch.cat([anchors, positives], dim=0)

            all_embeddings = model.embedding(all_patches)
            h_anchor = all_embeddings[:cur_batch]
            h_positive = all_embeddings[cur_batch : cur_batch * 2]
            h_pretext = all_embeddings[cur_batch * 2 : cur_batch * 3] if pretext_active else None

            z_anchor = F.normalize(model.projection(h_anchor), dim=1)
            z_positive = F.normalize(model.projection(h_positive), dim=1)
            sim_ap = (z_anchor @ z_positive.T) / temperature
            pos_sims = sim_ap.diag()
            sim_ap_fill = sim_ap.clone()
            sim_ap_fill.diagonal().fill_(float("inf"))
            neg_dists = 1.0 - sim_ap_fill
            hard_neg_dists = torch.max(neg_dists, dim=1).values
            pos_dists = 1.0 - pos_sims
            triplet_loss = F.relu(pos_dists - hard_neg_dists + triplet_margin).mean() / mu
            triplet_loss = _triplet_grad_scale(triplet_loss)

            if pretext_active:
                h_pre = h_pretext[pre_mask]
                h_anchor_pre = h_anchor[pre_mask]
                h_concat_pre = torch.cat([h_anchor_pre, h_pre], dim=1)
                all_indices = torch.arange(cur_batch, device=device)
                anchor_indices = all_indices.repeat_interleave(num_rand_patches)
                rand_offsets = torch.randint(1, cur_batch, (cur_batch * num_rand_patches,), device=device)
                unadj_indices = (anchor_indices + rand_offsets) % cur_batch
                h_unadj = h_anchor.index_select(0, unadj_indices)
                h_anchor_unadj = h_anchor.repeat_interleave(num_rand_patches, dim=0)
                h_concat_unadj = torch.cat([h_anchor_unadj, h_unadj], dim=1)
                pretext_features = torch.cat([h_concat_pre, h_concat_unadj], dim=0)
                pretext_labels = torch.cat(
                    [
                        torch.ones(h_concat_pre.shape[0], device=device),
                        torch.zeros(h_concat_unadj.shape[0], device=device),
                    ]
                )
                pretext_outputs = model.classification_head(pretext_features).squeeze(1)
                pretext_loss_all = criterion_pretext(pretext_outputs, pretext_labels)
                pretext_loss = pretext_loss_all[: h_concat_pre.shape[0]].mean() + pretext_loss_all[
                    h_concat_pre.shape[0] :
                ].mean()
                lambda_pretext = 1.0 * (1.0 - (iteration / (num_iters / 10.0)))
            else:
                pretext_loss = torch.tensor(0.0, device=device)
                lambda_pretext = 0.0

            loss = triplet_loss + float(lambda_pretext) * pretext_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

            if float(loss.item()) < best_loss:
                best_loss = float(loss.item())
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model, {"iterations": int(iteration), "best_loss": best_loss, "last_loss": last_loss}


def _train_t1_encoder(
    train_patches: torch.Tensor,
    in_channels: int,
    hp: dict,
    device: torch.device,
) -> tuple[_PatchEncoder, _PatchEncoder, dict]:
    model = _PatchEncoder(
        in_channels=in_channels,
        embed_dim=int(hp.get("embed_dim", 64)),
        projection_dim=int(hp.get("projection_dim", 256)),
        use_revin=bool(hp.get("use_revin", True)),
    ).to(device)
    model.train()

    train_patches_device = train_patches.to(device=device, non_blocking=True)
    total = int(train_patches_device.shape[0])
    batch_size = max(2, min(int(hp.get("batch_size", 512)), total))
    num_iters = max(1, int(hp.get("num_iters", 200)))
    lr = float(hp.get("lr", 1e-4))
    alpha = float(hp.get("alpha", 10.0))
    triplet_margin = float(hp.get("triplet_margin", 0.5))
    ema_weight = float(hp.get("ema_weight", 0.5))
    positive_radius = int(hp.get("positive_radius", 2))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    teacher = _EMATeacher(model, tau_start=0.996, tau_end=0.999, total_steps=num_iters)

    def cosine_lr(iteration: int) -> float:
        t = min(iteration, num_iters)
        end = lr / 10.0
        return end + (lr - end) * 0.5 * (1.0 + math.cos(math.pi * t / num_iters))

    iteration = 0
    last_loss = float("nan")
    while iteration < num_iters:
        permutation = torch.randperm(total, device=device)
        for start in range(0, total, batch_size):
            if iteration >= num_iters:
                break
            iteration += 1
            for param_group in optimizer.param_groups:
                param_group["lr"] = cosine_lr(iteration)

            batch_indices = permutation[start : start + batch_size]
            anchors = train_patches_device.index_select(0, batch_indices)
            pos_indices = _sample_temporal_positive(batch_indices, total, positive_radius, device)
            positives = train_patches_device.index_select(0, pos_indices)

            features = model.embedding(torch.cat([anchors, positives], dim=0))
            batch_n = int(anchors.shape[0])
            h_anchor = features[:batch_n]
            h_positive = features[batch_n:]

            z_anchor_raw = model.projection(h_anchor)
            z_positive_raw = model.projection(h_positive)
            z_anchor = F.normalize(z_anchor_raw, dim=1)
            z_positive = F.normalize(z_positive_raw, dim=1)

            sim = z_anchor @ z_positive.T
            pos_sim = sim.diag()
            neg = sim.clone()
            neg.diagonal().fill_(float("-inf"))
            hard_neg_sim = neg.max(dim=1).values
            triplet = F.relu((1.0 - pos_sim) - (1.0 - hard_neg_sim) + triplet_margin).mean()
            triplet = _triplet_grad_scale(triplet)

            norm_anchor = h_anchor.norm(dim=1).clamp_min(1e-8)
            norm_positive = h_positive.norm(dim=1).clamp_min(1e-8)
            t1_loss = ((torch.log(norm_anchor) - torch.log(norm_positive)) ** 2).mean()

            with torch.no_grad():
                teacher_proj = teacher.projection(teacher.embedding(anchors))
                teacher_proj = F.normalize(teacher_proj, dim=1)
            ema_loss = (1.0 - (F.normalize(z_anchor_raw, dim=1) * teacher_proj).sum(dim=1)).mean()

            loss = triplet + alpha * t1_loss + ema_weight * ema_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            teacher.update(model)
            last_loss = float(loss.detach().cpu())

    model.eval()
    teacher.teacher.eval()
    return model, teacher.teacher, {"iterations": int(iteration), "last_loss": last_loss}


@torch.inference_mode()
def _embed_patches(model: _PatchEncoder, patches: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    model.eval()
    outputs = []
    step = max(1, int(batch_size))
    for start in range(0, int(patches.shape[0]), step):
        end = min(start + step, int(patches.shape[0]))
        batch = patches[start:end].to(device=device, non_blocking=True)
        outputs.append(model.embedding(batch).detach().cpu().float())
    return torch.cat(outputs, dim=0)


def _build_wk1_bank(train_embeddings: torch.Tensor, bank_k: int, seed: int = 42) -> torch.Tensor:
    train_h = train_embeddings.detach().cpu().float()
    n = int(train_h.shape[0])
    if n <= 1:
        return train_h
    k_use = min(int(bank_k), max(1, n - 1))
    if k_use >= n:
        return train_h
    normalized = F.normalize(train_h, dim=1, eps=1e-12)
    mbk = MiniBatchKMeans(
        n_clusters=k_use,
        init="k-means++",
        random_state=int(seed),
        batch_size=max(8192, k_use),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    )
    mbk.fit(normalized.numpy())
    centers = torch.as_tensor(mbk.cluster_centers_, dtype=train_h.dtype)
    nearest = torch.cdist(normalized, centers).argmin(dim=0)
    return train_h.index_select(0, nearest)


def _public_paano_bank_size(num_samples: int) -> int:
    if num_samples <= 1:
        return int(num_samples)
    k = int(round(0.1 * num_samples))
    min_cores_eff = min(500, max(1, num_samples - 1))
    return max(min_cores_eff, min(k, num_samples - 1))


def _build_public_paano_bank(train_embeddings: torch.Tensor, seed: int = 42) -> torch.Tensor:
    train_h = train_embeddings.detach().cpu().float()
    n = int(train_h.shape[0])
    if n <= 1:
        return train_h
    k_use = _public_paano_bank_size(n)
    if k_use >= n:
        return train_h
    normalized = F.normalize(train_h, dim=1, eps=1e-12)
    mbk = MiniBatchKMeans(
        n_clusters=k_use,
        init="k-means++",
        random_state=int(seed),
        batch_size=max(8192, k_use),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    )
    mbk.fit(normalized.numpy())
    centers = torch.as_tensor(mbk.cluster_centers_, dtype=train_h.dtype)
    nearest = torch.cdist(normalized, centers).argmin(dim=0)
    return train_h.index_select(0, nearest)


def _split_train_bank_embeddings(train_embeddings: torch.Tensor, calib_frac: float, seed: int) -> torch.Tensor:
    """A7/A28 bank split: hold out 20% train patches, build the bank from the rest."""
    train_h = train_embeddings.detach().cpu().float()
    n = int(train_h.shape[0])
    if n <= 2:
        return train_h
    rng = np.random.default_rng(int(seed))
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    n_calib = max(1, int(round(n * float(calib_frac))))
    n_calib = min(n - 1, n_calib)
    bank_idx = np.sort(all_idx[n_calib:])
    if len(bank_idx) == 0:
        return train_h
    return train_h.index_select(0, torch.as_tensor(bank_idx, dtype=torch.long))


def _cosine_topk(query: torch.Tensor, bank: torch.Tensor, top_k: int, batch_size: int, device: torch.device) -> np.ndarray:
    q = query.detach().cpu().float()
    b_cpu = bank.detach().cpu().float()
    if int(b_cpu.shape[0]) == 0:
        return np.zeros(int(q.shape[0]), dtype=np.float32)
    k = max(1, min(int(top_k), int(b_cpu.shape[0])))
    b = F.normalize(b_cpu, dim=1, eps=1e-12).to(device=device, dtype=torch.float32)
    out = []
    step = max(1, int(batch_size))
    for start in range(0, int(q.shape[0]), step):
        end = min(start + step, int(q.shape[0]))
        qb = F.normalize(q[start:end], dim=1, eps=1e-12).to(device=device, dtype=torch.float32, non_blocking=True)
        sims = qb @ b.T
        top = sims.topk(k=k, dim=1, largest=True).values
        out.append((1.0 - top).mean(dim=1).detach().cpu())
    return torch.cat(out).numpy().astype(np.float32, copy=False)


def _euclidean_topk(query: torch.Tensor, bank: torch.Tensor, top_k: int, batch_size: int, device: torch.device) -> np.ndarray:
    q = query.detach().cpu().float()
    b_cpu = bank.detach().cpu().float()
    if int(b_cpu.shape[0]) == 0:
        return np.zeros(int(q.shape[0]), dtype=np.float32)
    k = max(1, min(int(top_k), int(b_cpu.shape[0])))
    b = b_cpu.to(device=device, dtype=torch.float32)
    out = []
    step = max(1, int(batch_size))
    for start in range(0, int(q.shape[0]), step):
        end = min(start + step, int(q.shape[0]))
        qb = q[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        a2 = (qb ** 2).sum(dim=1, keepdim=True)
        b2 = (b ** 2).sum(dim=1, keepdim=True).T
        distances = (a2 + b2 - 2.0 * (qb @ b.T)).clamp_min(0.0).sqrt()
        top = distances.topk(k=k, dim=1, largest=False).values.mean(dim=1)
        out.append(top.detach().cpu())
    return torch.cat(out).numpy().astype(np.float32, copy=False)


def _magG(train_raw: np.ndarray, query_raw: np.ndarray) -> np.ndarray:
    train = _flatten_signal(train_raw)
    query = _flatten_signal(query_raw)
    med = float(np.median(train))
    mad = float(np.median(np.abs(train - med)) + EPS)
    return (np.abs(query - med) / mad).astype(np.float32)


def _T2_meanshift(train_raw: np.ndarray, query_raw: np.ndarray, window: int) -> np.ndarray:
    train = _flatten_signal(train_raw)
    query = _flatten_signal(query_raw)
    med = float(np.median(train))
    mad = float(np.median(np.abs(train - med)) + EPS)
    n = int(len(query))
    cumsum = np.concatenate([[0.0], np.cumsum(query, dtype=np.float64)])
    out = np.zeros(n, dtype=np.float32)
    radius = max(1, int(window))
    for t in range(n):
        left = max(0, t - radius)
        right = min(n, t + radius + 1)
        win_mean = (cumsum[right] - cumsum[left]) / max(1, right - left)
        out[t] = abs(float(win_mean) - med) / mad
    return out


def _fallback_magnitude_score(train_raw: np.ndarray, query_raw: np.ndarray, window: int) -> np.ndarray:
    mag = _magG(train_raw, query_raw)
    t2 = _T2_meanshift(train_raw, query_raw, window)
    return (0.667 * _zscore(mag) + 0.2 * _zscore(t2)).astype(np.float32)


class PaAno_PAI(BaseDetector):
    """PaAno T1 encoder with A28f PAI fusion, computed from scratch."""

    def __init__(self, HP: dict | None = None, normalize: bool = True, contamination: float = 0.1):
        super().__init__(contamination=contamination)
        self.HP = dict(HP or {})
        self.normalize = bool(normalize)
        self._is_fitted = False

    def fit(self, X: np.ndarray, y=None):
        hp = dict(self.HP)
        _set_seed(int(hp.get("seed", 2021)))
        train = _as_2d_float(X)
        n_samples, n_features = train.shape
        self._train_raw = train
        self._train_len = int(n_samples)
        self._patch_size = _effective_patch_size(n_samples, int(hp.get("patch_size", 64)))
        self._stride = max(1, int(hp.get("stride", 1)))
        self._score_recipe = str(hp.get("score_recipe", "a28"))

        if bool(hp.get("train_zscore", False)):
            self._train_mean = train.mean(axis=0, keepdims=True).astype(np.float32)
            self._train_std = train.std(axis=0, keepdims=True).astype(np.float32)
            self._train_std = np.where(self._train_std == 0.0, 1e-8, self._train_std).astype(np.float32)
            train_for_model = ((train - self._train_mean) / self._train_std).astype(np.float32)
        else:
            self._train_mean = np.zeros((1, n_features), dtype=np.float32)
            self._train_std = np.ones((1, n_features), dtype=np.float32)
            train_for_model = train

        if self._patch_size < 4 or n_samples < self._patch_size:
            self._fallback_only = True
            self.decision_scores_ = _fallback_magnitude_score(train, train, int(hp.get("t2_window", 32)))
            self._is_fitted = True
            return self

        self._fallback_only = False
        train_patches = _create_patch_tensor(train_for_model, patch_size=self._patch_size, stride=self._stride)
        device = _resolve_device(hp.get("device", "auto"))
        self._device = device
        embed_batch_size = int(hp.get("embed_batch_size", max(128, int(hp.get("batch_size", 512)))))

        if self._score_recipe == "public_pai":
            model, train_info = _train_public_paano_encoder(
                train_patches=train_patches,
                in_channels=n_features,
                hp=hp,
                device=device,
            )
            self._model = model
            self._teacher = model
            self._train_info = train_info
            train_embeddings = _embed_patches(self._model, train_patches, device, batch_size=embed_batch_size)
            self._bank = _build_public_paano_bank(train_embeddings, seed=int(hp.get("bank_seed", 42)))
        else:
            model, teacher, train_info = _train_t1_encoder(
                train_patches=train_patches,
                in_channels=n_features,
                hp=hp,
                device=device,
            )
            self._model = model
            self._teacher = teacher if bool(hp.get("use_teacher", True)) else model
            self._train_info = train_info
            train_embeddings = _embed_patches(self._teacher, train_patches, device, batch_size=embed_batch_size)
            bank_source = _split_train_bank_embeddings(
                train_embeddings,
                calib_frac=float(hp.get("calib_frac", 0.20)),
                seed=int(hp.get("calib_seed", 42)),
            )
            self._bank = _build_wk1_bank(
                bank_source,
                bank_k=int(hp.get("bank_k", 1024)),
                seed=int(hp.get("bank_seed", 42)),
            )
        self.decision_scores_ = self._score(train)
        self._is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("PaAno_PAI must be fitted before calling decision_function")
        return self._score(X)

    def _score(self, X: np.ndarray) -> np.ndarray:
        query = _as_2d_float(X)
        hp = self.HP
        if getattr(self, "_fallback_only", False) or len(query) < int(getattr(self, "_patch_size", 4)):
            return _fallback_magnitude_score(self._train_raw, query, int(hp.get("t2_window", 32)))

        query_for_model = ((query - self._train_mean) / self._train_std).astype(np.float32)
        patches = _create_patch_tensor(query_for_model, patch_size=self._patch_size, stride=self._stride)
        embed_batch_size = int(hp.get("embed_batch_size", max(128, int(hp.get("batch_size", 512)))))
        embeddings = _embed_patches(self._teacher, patches, self._device, batch_size=embed_batch_size)
        if getattr(self, "_score_recipe", "a28") == "public_pai":
            cos_patch = _cosine_topk(
                embeddings,
                self._bank,
                top_k=int(hp.get("top_k", 3)),
                batch_size=int(hp.get("distance_batch_size", 512)),
                device=self._device,
            )
            return _patch_scores_to_points(cos_patch, self._patch_size, len(query)).astype(np.float32)

        eu_patch = _euclidean_topk(
            embeddings,
            self._bank,
            top_k=int(hp.get("top_k", 3)),
            batch_size=int(hp.get("distance_batch_size", 512)),
            device=self._device,
        )
        eu = _patch_scores_to_points(eu_patch, self._patch_size, len(query))
        mag = _magG(self._train_raw, query)
        t2 = _T2_meanshift(self._train_raw, query, int(hp.get("t2_window", 32)))

        score = (
            _zscore(eu)
            + float(hp.get("mag_weight_unit_eu", 0.667)) * _zscore(mag)
            + float(hp.get("t2_weight_unit_eu", 0.2)) * _zscore(t2)
        )
        return np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def run_PaAno_PAI_Unsupervised(data: np.ndarray, HP: dict) -> np.ndarray:
    clf = PaAno_PAI(HP=HP)
    clf.fit(data)
    return np.asarray(clf.decision_scores_, dtype=np.float32).ravel()


def run_PaAno_PAI_Semisupervised(data_train: np.ndarray, data_test: np.ndarray, HP: dict) -> np.ndarray:
    clf = PaAno_PAI(HP=HP)
    clf.fit(data_train)
    return np.asarray(clf.decision_function(data_test), dtype=np.float32).ravel()


PaAno_PAI_HP = {
    "score_recipe": "a28",
    "train_zscore": False,
    "patch_size": 64,
    "stride": 1,
    "num_iters": 200,
    "batch_size": 512,
    "embed_batch_size": 512,
    "distance_batch_size": 512,
    "lr": 1e-4,
    "seed": 2021,
    "calib_seed": 42,
    "calib_frac": 0.20,
    "bank_seed": 42,
    "pretext_step": 64,
    "num_rand_patches": 5,
    "temperature": 1.0,
    "alpha": 10.0,
    "ema_weight": 0.5,
    "triplet_margin": 0.5,
    "positive_radius": 2,
    "embed_dim": 64,
    "projection_dim": 256,
    "use_revin": True,
    "use_teacher": True,
    "bank_k": 1024,
    "top_k": 3,
    "mag_weight_unit_eu": 0.667,
    "t2_weight_unit_eu": 0.2,
    "t2_window": 32,
    "device": "auto",
}


def _train_index_from_filename(filename: str) -> int:
    parts = Path(filename).stem.split("_")
    if "tr" in parts:
        idx = parts.index("tr")
        return int(parts[idx + 1])
    return int(parts[-3])


def _load_tsb_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path).dropna()
    data = df.iloc[:, 0:-1].values.astype(np.float32)
    label = df["Label"].astype(int).to_numpy()
    return data, label


def _evaluate_one_file(filename: str, dataset_dir: Path, hp: dict, save_metrics: bool = False) -> tuple[np.ndarray, dict | None, float]:
    data, label = _load_tsb_file(dataset_dir / filename)
    train_index = _train_index_from_filename(filename)
    data_train = data[:train_index, :]
    start = time.time()
    score = run_PaAno_PAI_Semisupervised(data_train, data, hp)
    runtime = time.time() - start
    if not save_metrics:
        return score, None, runtime
    sliding_window = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
    try:
        metrics = get_metrics(score, label, metric="all", slidingWindow=sliding_window)
    except TypeError:
        metrics = get_metrics(score, label, slidingWindow=sliding_window)
    return score, metrics, runtime


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Running PaAno_PAI for TSB-AD")
    parser.add_argument("--filename", type=str, default="001_NAB_id_1_Facility_tr_1007_1st_2014.csv")
    parser.add_argument("--data_direc", type=str, default="../Datasets/TSB-AD-U/")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--file_lsit", type=str, default=None)
    parser.add_argument("--file_list", type=str, default=None)
    parser.add_argument("--score_dir", type=str, default="eval/score/uni/")
    parser.add_argument("--save_dir", type=str, default="eval/metrics/uni/")
    parser.add_argument("--save", nargs="?", const=True, default=False, type=_str2bool)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--AD_Name", type=str, default="PaAno_PAI")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_iters", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ema_weight", type=float, default=None)
    parser.add_argument("--use_teacher", type=_str2bool, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    hp = dict(PaAno_PAI_HP)
    if args.device is not None:
        hp["device"] = args.device
    if args.num_iters is not None:
        hp["num_iters"] = int(args.num_iters)
    if args.batch_size is not None:
        hp["batch_size"] = int(args.batch_size)
    if args.seed is not None:
        hp["seed"] = int(args.seed)
    if args.ema_weight is not None:
        hp["ema_weight"] = float(args.ema_weight)
    if args.use_teacher is not None:
        hp["use_teacher"] = bool(args.use_teacher)

    dataset_dir = Path(args.dataset_dir or args.data_direc)
    file_list_arg = args.file_list or args.file_lsit

    if file_list_arg:
        target_dir = Path(args.score_dir) / args.AD_Name
        target_dir.mkdir(parents=True, exist_ok=True)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(target_dir / f"000_run_{args.AD_Name}.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        file_names = pd.read_csv(file_list_arg)["file_name"].astype(str).values
        rows = []
        for filename in file_names:
            output_path = target_dir / f"{Path(filename).stem}.npy"
            if args.skip_existing and output_path.exists():
                continue
            print(f"Processing:{filename} by {args.AD_Name}", flush=True)
            try:
                score, metrics, runtime = _evaluate_one_file(filename, dataset_dir, hp, save_metrics=args.save)
                np.save(output_path, score)
                logging.info("Success at %s using %s | Time cost: %.3fs", filename, args.AD_Name, runtime)
                if args.save and metrics is not None:
                    row = {"file": filename, "Time": runtime}
                    row.update(metrics)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(Path(args.save_dir) / f"{args.AD_Name}.csv", index=False)
            except Exception as exc:
                logging.exception("Failed at %s: %s", filename, exc)
        raise SystemExit(0)

    start_total = time.time()
    score, metrics, runtime = _evaluate_one_file(args.filename, dataset_dir, hp, save_metrics=True)
    print("score: ", score.shape)
    print("runtime: ", runtime)
    print("Evaluation Result: ", metrics)
    print(f"Total Runtime: {time.time() - start_total:.2f} seconds")
