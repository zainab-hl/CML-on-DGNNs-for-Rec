from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import LightGCN


def signal_repr_collapse(
    model: LightGCN, adj, global_u_gpu: torch.Tensor, n_sample: int
) -> float:
    """
    S4 — Representation collapse.
    Returns mean pairwise cosine similarity across a random sample of item embeddings.
    High values (> repr_collapse_thr) indicate embeddings are collapsing toward a single point.
    """
    model.eval()
    with torch.no_grad():
        _, x_item, _, _, _ = model.encode(adj, global_u_gpu)
        idx = torch.randperm(model.num_items, device=model.cfg.device)[:n_sample]
        emb = F.normalize(x_item[idx], dim=-1)
        sim = emb @ emb.T
        mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
        mean_cos = sim[mask].mean().item()
        del x_item, emb, sim
    if model.cfg.device == "cuda":
        torch.cuda.empty_cache()
    return float(mean_cos)


def signal_oversmooth(
    model: LightGCN, adj, global_u_gpu: torch.Tensor, n_sample: int
) -> float:
    """
    S5 — Over-smoothing.
    Returns mean cosine similarity between layer-0 and layer-K item embeddings.
    High values (> oversmooth_thr) indicate the graph convolution is washing out
    initial features.
    """
    model.eval()
    with torch.no_grad():
        _, _, _, layer0, layerK = model.encode(adj, global_u_gpu)
        idx = torch.randperm(model.num_items, device=model.cfg.device)[:n_sample]
        l0 = F.normalize(layer0[idx], dim=-1)
        lK = F.normalize(layerK[idx], dim=-1)
        cos = (l0 * lK).sum(dim=-1).mean().item()
        del layer0, layerK, l0, lK
    if model.cfg.device == "cuda":
        torch.cuda.empty_cache()
    return float(cos)


def signal_bpr_plateau(
    bpr_history: list[float], window: int, rel_drop: float
) -> tuple[bool, str]:
    """
    S3 — BPR plateau.
    Fires when the relative BPR improvement over the last `window` epochs
    is below `rel_drop`, indicating training has stalled.
    """
    if len(bpr_history) < window:
        return False, "n/a"
    recent = bpr_history[-window:]
    ref = recent[0]
    rel_improvement = (ref - recent[-1]) / (ref + 1e-8)
    fired = 0.0 <= rel_improvement < rel_drop
    return fired, f"rel={rel_improvement:.4f}"


def repr_health_snapshot(
    model: LightGCN, adj, global_u_gpu: torch.Tensor
) -> dict:
    """Compute repr-collapse and over-smoothing for any model (no growth state needed)."""
    cfg = model.cfg
    repr_cos = signal_repr_collapse(model, adj, global_u_gpu, cfg.repr_collapse_sample)
    smooth_cos = signal_oversmooth(model, adj, global_u_gpu, cfg.oversmooth_sample)
    return {
        "repr_collapse_cos": round(repr_cos, 4),
        "oversmooth_cos": round(smooth_cos, 4),
    }
