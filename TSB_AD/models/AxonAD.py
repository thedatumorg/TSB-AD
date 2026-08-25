from __future__ import division, print_function

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import tqdm

from .base import BaseDetector
from ..utils.dataset import ReconstructDataset
from ..utils.torch_utility import get_gpu, EarlyStoppingTorch


# ==========================================
# 1. HELPER MODULES
# ==========================================

class CausalConv1d(nn.Module):
    """Causal 1D conv: output[t] depends only on input[:t]."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)

    def forward(self, x):  # [B,C,T]
        y = self.conv(x)
        if self.pad > 0:
            y = y[:, :, :-self.pad]
        return y


class TCNQueryPredictor(nn.Module):
    """
    Minimal predictor: hidden_dim tied to d_model, dilations fixed.
    """
    def __init__(self, d_model, dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        hid = d_model
        layers = []
        in_ch = d_model
        for d in dilations:
            layers += [
                CausalConv1d(in_ch, hid, kernel_size=3, dilation=d),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = hid
        layers += [nn.Conv1d(hid, d_model, kernel_size=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # [B,T,C]
        x = x.permute(0, 2, 1)   # [B,C,T]
        y = self.net(x)          # [B,C,T]
        return y.permute(0, 2, 1)


class PredictiveAttention(nn.Module):
    """
    Attention block that:
      - uses REAL attention for reconstruction (teacher)
      - predicts Q via causal TCN from history-only (student)

    JEPA-style: we supervise the student by matching an EMA teacher's Q on
    masked timesteps (handled in the wrapper model, not here).
    """
    def __init__(self, d_model, num_heads=4, forecast_steps=1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.forecast_steps = int(forecast_steps)

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model)

        self.q_pred_net = TCNQueryPredictor(d_model=d_model)

    def _split_heads(self, x):  # [B,T,C] -> [B,H,T,D]
        B, T, C = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, h_teacher, h_hist):
        """
        h_teacher: [B,T,C] (grad) used for REAL attention reconstruction
        h_hist:    [B,T,C] (grad) used for student Q prediction (NO detach!)
        """
        B, T, C = h_teacher.shape

        # Teacher Q,K,V (for reconstruction)
        q_real = self._split_heads(self.wq(h_teacher))     # [B,H,T,D]
        k      = self._split_heads(self.wk(h_teacher))     # [B,H,T,D]
        v      = self._split_heads(self.wv(h_teacher))     # [B,H,T,D]

        # Student Q prediction from history-only (shifted by forecast_steps)
        s = max(0, min(self.forecast_steps, T))
        if s == 0:
            x_shift = h_hist
        else:
            zeros = h_hist.new_zeros(B, s, C)
            x_shift = torch.cat([zeros, h_hist[:, :-s, :]], dim=1)  # [B,T,C]

        q_pred_flat = self.q_pred_net(x_shift)             # [B,T,C]
        q_pred = self._split_heads(q_pred_flat)            # [B,H,T,D]

        # Reconstruction context uses TEACHER attention only
        if hasattr(F, "scaled_dot_product_attention"):
            ctx = F.scaled_dot_product_attention(q_real, k, v, dropout_p=0.0, is_causal=False)  # [B,H,T,D]
        else:
            scale = math.sqrt(self.head_dim)
            attn = F.softmax((q_real @ k.transpose(-2, -1)) / scale, dim=-1)
            ctx = attn @ v

        out = ctx.transpose(1, 2).contiguous().view(B, T, C)
        out = self.wo(out)

        return out, q_real, q_pred, k


# ==========================================
# 2. CORE MODEL
# ==========================================

class TCNTransformerModel(nn.Module):
    """
    Minimal backbone:
      embed + pos + (attn + ffn) + head

    JEPA-style addition:
      - EMA target encoder for producing q_target (stop-grad)
      - masked-timestep supervision in Q-space (representation prediction)

    Learnable scalars:
      - log_tau: kept for KL-tail scoring (optional)
      - s_rec, s_kl: auto-balance losses (no alpha hyperparam)
        (s_kl now balances JEPA loss)
    """
    def __init__(self, feats, seq_len, d_model=64, num_heads=4, forecast_steps=1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.forecast_steps = int(forecast_steps)

        # --- Online / student encoder ---
        self.embed = nn.Linear(feats, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attn = PredictiveAttention(d_model=d_model, num_heads=num_heads, forecast_steps=forecast_steps)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.head = nn.Linear(d_model, feats)

        # --- EMA target encoder (JEPA-style) ---
        # We only need an encoder up to q-space: embed+pos+ln1 and a Q projection.
        self.t_embed = nn.Linear(feats, d_model)
        self.t_pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.t_ln1 = nn.LayerNorm(d_model)
        self.t_wq = nn.Linear(d_model, d_model, bias=False)

        # Init target as copy of online
        self._init_target()

        # Fixed EMA momentum (no new hyperparam in signature)
        self.ema_m = 0.9

        # --- Learnable scalars (minimal + useful) ---
        self.log_tau = nn.Parameter(torch.zeros(()))  # tau = exp(log_tau) (student sharpness)
        self.s_rec = nn.Parameter(torch.zeros(()))    # uncertainty-style weights
        self.s_kl  = nn.Parameter(torch.zeros(()))    # balances JEPA loss

        # Masking defaults (no signature change)
        self.mask_ratio = 0.5  # mask ~50% of tail timesteps for JEPA loss
        self.mask_block_frac = 0.5  # each block length ~ block_frac * tail_len

    def _init_target(self):
        # Copy weights
        with torch.no_grad():
            self.t_embed.weight.copy_(self.embed.weight)
            self.t_embed.bias.copy_(self.embed.bias)
            self.t_pos.copy_(self.pos)
            self.t_ln1.weight.copy_(self.ln1.weight)
            self.t_ln1.bias.copy_(self.ln1.bias)

            self.t_wq.weight.copy_(self.attn.wq.weight)

        # Freeze target grads
        for p in self.t_embed.parameters():
            p.requires_grad_(False)
        self.t_pos.requires_grad_(False)
        for p in self.t_ln1.parameters():
            p.requires_grad_(False)
        for p in self.t_wq.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def ema_update_target(self):
        """EMA update target encoder from online encoder (JEPA-style)."""
        m = float(self.ema_m)

        def ema_param(t_param, s_param):
            t_param.mul_(m).add_(s_param, alpha=(1.0 - m))

        ema_param(self.t_embed.weight, self.embed.weight)
        ema_param(self.t_embed.bias, self.embed.bias)
        ema_param(self.t_pos, self.pos)
        ema_param(self.t_ln1.weight, self.ln1.weight)
        ema_param(self.t_ln1.bias, self.ln1.bias)
        ema_param(self.t_wq.weight, self.attn.wq.weight)

    def tau(self):
        return torch.exp(self.log_tau).clamp(0.25, 4.0)

    def _split_heads(self, x, num_heads):  # [B,T,C] -> [B,H,T,D]
        B, T, C = x.shape
        head_dim = C // num_heads
        return x.view(B, T, num_heads, head_dim).transpose(1, 2)

    def _make_tail_mask(self, B, T, s, tail_len, device):
        """
        JEPA mask over timesteps (time-patch mask) focused on the tail window.
        Returns mask: [B, T] bool.
        """
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        if tail_len <= 0:
            return mask

        # Tail range [t_start, T)
        t_start = max(s, T - tail_len)
        if t_start >= T:
            return mask

        span_len = max(1, int(round(tail_len * float(self.mask_block_frac))))
        span_len = min(span_len, T - t_start)

        # Mask about mask_ratio fraction of tail timesteps via blocks
        target_masked = max(1, int(round((T - t_start) * float(self.mask_ratio))))
        # Approx number of blocks
        n_blocks = max(1, int(math.ceil(target_masked / span_len)))

        # Sample blocks per batch element
        for b in range(B):
            for _ in range(n_blocks):
                # start in [t_start, T - span_len]
                lo = t_start
                hi = max(lo, T - span_len)
                if hi == lo:
                    st = lo
                else:
                    st = int(torch.randint(low=lo, high=hi + 1, size=(1,), device=device).item())
                mask[b, st:st + span_len] = True

        # Ensure we don't supervise invalid timesteps (< s)
        if s > 0:
            mask[:, :s] = False
        return mask

    def forward(self, x):
        """
        Returns:
          rec:      [B,T,feats]
          q_real:   [B,H,T,D]  (online teacher for reconstruction)
          q_pred:   [B,H,T,D]  (student predictor output)
          k:        [B,H,T,D]
          q_tgt:    [B,H,T,D]  (EMA target Q, stop-grad)
          tmask:    [B,T] bool (masked timesteps for JEPA loss; only meaningful in train)
        """
        B, T, _ = x.shape

        # --- Online path ---
        h = self.embed(x) + self.pos
        h_norm = self.ln1(h)

        h_attn, q_real, q_pred, k = self.attn(h_teacher=h_norm, h_hist=h_norm)
        h2 = h + h_attn
        h2 = h2 + self.ffn(self.ln2(h2))
        rec = self.head(h2)

        # --- Target path (EMA encoder -> q_tgt) ---
        with torch.no_grad():
            ht = self.t_embed(x) + self.t_pos
            ht_norm = self.t_ln1(ht)
            # Use same head config as online attention
            q_tgt_flat = self.t_wq(ht_norm)  # [B,T,C]
            q_tgt = self._split_heads(q_tgt_flat, self.attn.num_heads)  # [B,H,T,D]

        # --- JEPA timestep mask (train only) ---
        if self.training:
            s = max(0, min(self.forecast_steps, T))
            # Focus mask on tail; use same tail_len default as scoring (k_tail is in wrapper),
            # here we approximate with max(1, T//3) if wrapper doesn't override.
            # The wrapper will still compute tail length properly for scoring.
            tail_len = max(1, (T - s) // 2)
            tmask = self._make_tail_mask(B=B, T=T, s=s, tail_len=tail_len, device=x.device)
        else:
            tmask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        return rec, q_real, q_pred, k, q_tgt, tmask


# ==========================================
# 3. DETECTOR WRAPPER
# ==========================================

class TCNTransformer(BaseDetector):
    """
    End-to-end (single-stage) training with minimal moving parts.

    Training loss (JEPA-style):
      L = exp(-s_rec)*L_rec + s_rec  +  exp(-s_kl)*L_jepa + s_kl

    Scoring (recommended):
      score = robust_z(MSE) + robust_z(JEPA_tail_dist) (+ optional robust_z(KL_tail))

    Notes:
      - KL-tail remains available (and robust / MPS-safe) for scoring/ablation.
      - JEPA target is EMA encoder producing q_tgt (stop-grad).
    """
    def __init__(self,
                 win_size=100,
                 feats=1,
                 d_model=64,
                 num_heads=4,
                 batch_size=128,
                 epochs=50,
                 patience=3,
                 lr=1e-3,
                 validation_size=0.2,
                 kl_tail_k=5,
                 forecast_steps=1):
        super().__init__()
        self.cuda = True
        self.device = get_gpu(self.cuda)

        self.win_size = win_size
        self.feats = feats
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.validation_size = validation_size

        self.kl_tail_k = int(kl_tail_k)
        self.forecast_steps = int(forecast_steps)

        self.model = TCNTransformerModel(
            feats=feats,
            seq_len=win_size,
            d_model=d_model,
            num_heads=num_heads,
            forecast_steps=forecast_steps,
        ).to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.mse = nn.MSELoss()
        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

        # robust stats for scoring
        self.mse_med = self.mse_iqr = None
        self.jepa_med = self.jepa_iqr = None
        self.kl_med = self.kl_iqr = None  # optional term

        self.__anomaly_score = None

    # ---------- KL-tail (kept for scoring/ablation; log-space, tail-only, teacher stop-grad) ----------

    def _kl_tail(self, q_real, q_pred, k):
        """
        KL(softmax(real) || softmax(pred)) computed on last k query rows, log-space stable.

        Teacher stop-grad: q_real and k are detached for KL => KL trains predictor (+ tau) if you use it.
        """
        B, H, T, D = q_real.shape
        scale = math.sqrt(D)

        s = max(0, min(self.forecast_steps, T))
        valid_T = T - s
        if valid_T <= 0:
            return q_real.new_zeros(B)

        k_tail = min(self.kl_tail_k, valid_T) if self.kl_tail_k > 0 else valid_T
        t0 = max(s, T - k_tail)

        qR = q_real[:, :, t0:, :].detach()          # [B,H,k,D]
        qP = q_pred[:, :, t0:, :]                   # [B,H,k,D]
        Kd = k.detach().transpose(-2, -1)           # [B,H,D,T]

        sr = (qR @ Kd) / scale                      # [B,H,k,T]
        sp = (qP @ Kd) / scale                      # [B,H,k,T]
        sp = sp * self.model.tau()

        # Fairness mask: for query timestep t, student only has info up to (t - s)
        t_idx = torch.arange(t0, T, device=sr.device)               # [k]
        max_key = (t_idx - s).clamp(min=0)                          # [k]
        j_idx = torch.arange(T, device=sr.device).view(1, 1, 1, T)  # [1,1,1,T]
        max_key = max_key.view(1, 1, -1, 1)                         # [1,1,k,1]
        mask_future = j_idx > max_key                               # [1,1,k,T]

        # MPS-safe finite negative
        neg = torch.finfo(sr.dtype).min
        sr = sr.masked_fill(mask_future, neg)
        sp = sp.masked_fill(mask_future, neg)

        logp = F.log_softmax(sr, dim=-1)
        logq = F.log_softmax(sp, dim=-1)
        p = logp.exp()

        kl = (p * (logp - logq)).sum(dim=-1)   # [B,H,k]
        return kl.mean(dim=(1, 2))             # [B]

    # ---------- JEPA distance (representation mismatch) ----------

    def _jepa_tail_dist(self, q_tgt, q_pred):
        """
        Cosine distance on tail timesteps:
          dist = mean_{tail} (1 - cos(norm(q_pred), norm(q_tgt)))

        Uses valid timesteps only (t >= s) and last kl_tail_k positions (for alignment with scoring).
        """
        B, H, T, D = q_pred.shape
        s = max(0, min(self.forecast_steps, T))
        valid_T = T - s
        if valid_T <= 0:
            return q_pred.new_zeros(B)

        k_tail = min(self.kl_tail_k, valid_T) if self.kl_tail_k > 0 else valid_T
        t0 = max(s, T - k_tail)

        qp = F.normalize(q_pred[:, :, t0:, :], dim=-1)
        qt = F.normalize(q_tgt[:, :, t0:, :], dim=-1)  # q_tgt already stop-grad in model forward
        cos = (qp * qt).sum(dim=-1)  # [B,H,k]
        dist = 1.0 - cos
        return dist.mean(dim=(1, 2))  # [B]

    def _jepa_masked_loss(self, q_tgt, q_pred, tmask):
        """
        JEPA loss: supervised only on masked timesteps (time-patch mask),
        matching q_pred to q_tgt in normalized space via cosine distance.
        """
        # q_tgt: [B,H,T,D], q_pred: [B,H,T,D], tmask: [B,T] bool
        B, H, T, D = q_pred.shape
        if tmask is None or tmask.numel() == 0:
            return q_pred.new_zeros(())

        # Exclude invalid (<s) timesteps already handled by tmask construction, but keep safe
        s = max(0, min(self.forecast_steps, T))
        if s > 0:
            tmask = tmask.clone()
            tmask[:, :s] = False

        m = tmask.view(B, 1, T).to(q_pred.dtype)  # [B,1,T]
        denom = m.sum() * H + 1e-9

        qp = F.normalize(q_pred, dim=-1)
        qt = F.normalize(q_tgt, dim=-1)
        cos = (qp * qt).sum(dim=-1)  # [B,H,T]
        dist = (1.0 - cos) * m       # [B,H,T] masked
        return dist.sum() / denom

    # ---------- Robust calibration ----------

    @staticmethod
    def _robust_fit(arr, eps=1e-9):
        med = float(np.median(arr))
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = float((q75 - q25) + eps)
        return med, iqr

    def _calibrate(self, tsTrain):
        loader = DataLoader(ReconstructDataset(tsTrain, window_size=self.win_size),
                            batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        mse_vals, jepa_vals, kl_vals = [], [], []

        with torch.no_grad():
            for d, _ in loader:
                d = d.to(self.device)
                rec, q_real, q_pred, k, q_tgt, _ = self.model(d)

                mse_rec = (rec - d).pow(2).sum(dim=-1).mean(dim=1)
                jepa = self._jepa_tail_dist(q_tgt, q_pred)

                # Optional KL term (kept for ablations / optional scoring)
                kl = self._kl_tail(q_real, q_pred, k)

                mse_vals.append(mse_rec.cpu())
                jepa_vals.append(jepa.cpu())
                kl_vals.append(kl.cpu())

        if len(mse_vals) == 0:
            self.mse_med, self.mse_iqr = 0.0, 1.0
            self.jepa_med, self.jepa_iqr = 0.0, 1.0
            self.kl_med, self.kl_iqr = 0.0, 1.0
            return

        mse_all = torch.cat(mse_vals).numpy()
        jepa_all = torch.cat(jepa_vals).numpy()
        kl_all = torch.cat(kl_vals).numpy()

        self.mse_med, self.mse_iqr = self._robust_fit(mse_all)
        self.jepa_med, self.jepa_iqr = self._robust_fit(jepa_all)
        self.kl_med, self.kl_iqr = self._robust_fit(kl_all)

    # ---------- Training ----------

    def fit(self, data):
        split = int((1 - self.validation_size) * len(data))
        if split <= self.win_size:
            tsTrain = data
            tsValid = data[:0]
        else:
            tsTrain = data[:split]
            tsValid = data[split:]

        train_loader = DataLoader(ReconstructDataset(tsTrain, window_size=self.win_size),
                                  batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(ReconstructDataset(tsValid, window_size=self.win_size),
                                  batch_size=self.batch_size, shuffle=False)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            loop = tqdm.tqdm(train_loader, leave=True)

            for d, _ in loop:
                d = d.to(self.device)
                rec, q_real, q_pred, k, q_tgt, tmask = self.model(d)

                L_rec = self.mse(rec, d)
                L_jepa = self._jepa_masked_loss(q_tgt, q_pred, tmask)

                # Learnable balancing (no alpha / lambda tuning)
                s_rec = self.model.s_rec
                s_kl  = self.model.s_kl  # now balances JEPA
                loss = torch.exp(-s_rec) * L_rec + s_rec + torch.exp(-s_kl) * L_jepa + s_kl

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                # EMA update (JEPA-style target)
                self.model.ema_update_target()

                # Optional: monitor tail distances (no extra backward)
                with torch.no_grad():
                    tail_jepa = self._jepa_tail_dist(q_tgt, q_pred).mean().item()
                    tail_kl = self._kl_tail(q_real, q_pred, k).mean().item()

                loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(
                    rec=float(L_rec.item()),
                    jepa=float(L_jepa.item()),
                    tail_jepa=float(tail_jepa),
                    tail_kl=float(tail_kl),
                    tau=float(self.model.tau().item()),
                    s_rec=float(s_rec.item()),
                    s_kl=float(s_kl.item()),
                    loss=float(loss.item()),
                )

            # Validation
            if len(valid_loader) > 0:
                self.model.eval()
                val_rec_sum, val_jepa_sum, val_loss_sum, n_batches = 0.0, 0.0, 0.0, 0

                with torch.no_grad():
                    for d, _ in valid_loader:
                        d = d.to(self.device)
                        rec, q_real, q_pred, k, q_tgt, _ = self.model(d)

                        v_rec = self.mse(rec, d)
                        # Use tail distance for validation stability (deterministic) instead of masked loss
                        v_jepa = self._jepa_tail_dist(q_tgt, q_pred).mean()

                        s_rec = self.model.s_rec
                        s_kl  = self.model.s_kl
                        v_loss = torch.exp(-s_rec) * v_rec + s_rec + torch.exp(-s_kl) * v_jepa + s_kl

                        val_rec_sum  += float(v_rec.item())
                        val_jepa_sum += float(v_jepa.item())
                        val_loss_sum += float(v_loss.item())
                        n_batches += 1

                val_rec_avg  = val_rec_sum / n_batches
                val_jepa_avg = val_jepa_sum / n_batches
                val_loss_avg = val_loss_sum / n_batches

                # Early stop on reconstruction (most stable)
                self.early_stopping(val_rec_avg, self.model)

                print(f"[VAL] rec={val_rec_avg:.6f}  jepa_tail={val_jepa_avg:.6f}  loss={val_loss_avg:.6f}")

                if self.early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

        self._calibrate(tsTrain)

    # ---------- Inference ----------

    def decision_function(self, data):
        loader = DataLoader(ReconstructDataset(data, window_size=self.win_size),
                            batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        scores = []
        eps = 1e-9

        with torch.inference_mode():
            for d, _ in tqdm.tqdm(loader, leave=True):
                d = d.to(self.device)
                rec, q_real, q_pred, k, q_tgt, _ = self.model(d)

                mse_rec = (rec - d).pow(2).sum(dim=-1).mean(dim=1)
                jepa = self._jepa_tail_dist(q_tgt, q_pred)

                # Optional KL term (kept for ablation / can be added back if you want)
                kl = self._kl_tail(q_real, q_pred, k)

                z_mse = (mse_rec - self.mse_med) / (self.mse_iqr + eps)
                z_jepa = (jepa - self.jepa_med) / (self.jepa_iqr + eps)

                # Recommended score: MSE + JEPA mismatch
                score = z_mse + z_jepa

                # If you want to include KL too, uncomment:
                # z_kl = (kl - self.kl_med) / (self.kl_iqr + eps)
                # score = z_mse + z_jepa + z_kl

                scores.append(score.cpu())

        scores = torch.cat(scores).numpy()
        self.__anomaly_score = scores

        # keep your original center padding behavior
        if self.__anomaly_score.shape[0] < len(data):
            pad_l = math.ceil((self.win_size - 1) / 2)
            pad_r = (self.win_size - 1) // 2
            self.__anomaly_score = np.array(
                [self.__anomaly_score[0]] * pad_l +
                list(self.__anomaly_score) +
                [self.__anomaly_score[-1]] * pad_r
            )

        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
