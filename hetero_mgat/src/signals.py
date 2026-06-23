"""
signals.py
==========
The continuous "health" signals consumed by the GrowthController (and,
for repr-collapse / oversmoothing, also reported for the EWC-only and
Cumulative-No-EWC baselines so all three are directly comparable).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import Config
from .model import HeteroMGAT, ItemFeatures


def signal_repr_and_smooth(cfg: Config, model: HeteroMGAT, ei_gpu: Dict,
                           global_u_gpu: torch.Tensor, hist_gpu: torch.Tensor,
                           feats: ItemFeatures,
                           n_sample: int = None) -> Tuple[float, float]:
    """
    Fused S4 + S5 in a single encode() call.

    S4 (repr_cos)   — mean pairwise cosine between DIFFERENT items at layer2.
                      Detects embedding collapse (all items → same point).
    S5 (smooth_cos) — cosine between SAME item at layer0 vs layer2.
                      Detects over-smoothing (GNN not changing representations).
    """
    n_sample = n_sample or cfg.SIGNAL_SAMPLE
    model.eval()
    with torch.no_grad():
        _, _, layer0, layer2 = model.encode(ei_gpu, global_u_gpu, hist_gpu, feats)
        idx = torch.randperm(feats.num_items, device=cfg.device)[:n_sample]

        emb = F.normalize(layer2[idx], dim=-1)
        sim = emb @ emb.T
        tri_mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        repr_cos = sim[tri_mask].mean().item()

        l0 = F.normalize(layer0[idx], dim=-1)
        l2 = F.normalize(layer2[idx], dim=-1)
        smooth_cos = (l0 * l2).sum(dim=-1).mean().item()

        del layer0, layer2, emb, sim, l0, l2, tri_mask
    if cfg.device == 'cuda':
        torch.cuda.empty_cache()
    return float(repr_cos), float(smooth_cos)


def signal_bpr_plateau(cfg: Config, bpr_history: List[float]) -> Tuple[bool, str]:
    """
    S3: relative BPR improvement over last PLATEAU_WINDOW epochs.
    Only fires when improvement is positive but insufficient.
    Worsening BPR does NOT fire this signal.
    """
    if len(bpr_history) < cfg.PLATEAU_WINDOW:
        return False, 'n/a'
    recent = bpr_history[-cfg.PLATEAU_WINDOW:]
    ref = recent[0]
    rel = (ref - recent[-1]) / (ref + 1e-8)
    fired = (0.0 <= rel < cfg.PLATEAU_REL_DROP)
    return fired, f'rel={rel:.4f}'


def repr_health_snapshot(cfg: Config, model: HeteroMGAT, ei_gpu: Dict,
                         global_u_gpu: torch.Tensor, hist_gpu: torch.Tensor,
                         feats: ItemFeatures) -> Dict[str, float]:
    """
    Computes both representation-health signals (repr-collapse mean cosine,
    over-smoothing layer0-vs-layer2 cosine) for any model, regardless of
    whether it has a GrowthController. Safe to call on EWC-only /
    Cumulative-No-EWC as well as the dynamic model.
    """
    repr_cos, smooth_cos = signal_repr_and_smooth(
        cfg, model, ei_gpu, global_u_gpu, hist_gpu, feats)
    return {
        'repr_collapse_cos': round(repr_cos, 4),
        'oversmooth_cos': round(smooth_cos, 4),
    }
