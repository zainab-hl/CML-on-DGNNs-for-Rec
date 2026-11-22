
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .complexity import ComplexityTracker
from .config import Config
from .data import DataModule
from .ewc import EWC, accumulate_fisher
from .evaluator import Evaluator
from .model import HeteroMGAT, ItemFeatures


def train_one_epoch(cfg: Config, model: HeteroMGAT, optimizer, ei_gpu: Dict,
                    global_gpu: torch.Tensor, hist_gpu: torch.Tensor,
                    feats: ItemFeatures, pos_u_local: torch.Tensor,
                    pos_i: torch.Tensor, N_: int, ewc: Optional[EWC] = None,
                    warmup: float = 1.0, skip_phase2: bool = False
                    ) -> Tuple[float, float, torch.Tensor]:
    """
    Two-phase epoch:
    Phase 1 — predictor batches on detached embeddings (many cheap batches)
    Phase 2 — one full GNN backward + EWC penalty
               Skipped when skip_phase2=True (used during growth warmup)
    Returns (avg_bpr, ewc_val, gate_weights [N_items, 4])
    """
    model.train()

    # ── Phase 1: detached encode ──────────────────────────────────────
    with torch.no_grad():
        x_det, gates, _, _ = model.encode(ei_gpu, global_gpu, hist_gpu, feats)
    if cfg.device == 'cuda':
        torch.cuda.empty_cache()

    graph_conv_names = {n for n, _ in model.graph_conv_parameters()}
    for name, p in model.named_parameters():
        p.requires_grad_(name not in graph_conv_names)

    perm = torch.randperm(N_)
    tot, steps = 0.0, 0
    for s in range(0, N_, cfg.BATCH):
        idx = perm[s:s + cfg.BATCH]
        pu_local = pos_u_local[idx]
        pi = pos_i[idx]
        ni = torch.randint(0, feats.num_items, (len(idx),))
        optimizer.zero_grad()
        ps = model.decode(x_det, pu_local, pi, cfg.device)
        ns = model.decode(x_det, pu_local, ni, cfg.device)
        bpr = -F.logsigmoid(ps - ns).mean()
        bpr.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += bpr.item(); steps += 1
        del ps, ns, bpr, pu_local, pi, ni

    del x_det
    if cfg.device == 'cuda':
        torch.cuda.empty_cache()
    for p in model.parameters():
        p.requires_grad_(True)

    # ── Phase 2: full GNN backward ──────────────────────────────────────
    if skip_phase2:
        return tot / max(steps, 1), 0.0, gates

    idx2 = perm[:min(cfg.PHASE2_BATCH, N_)]
    pu2 = pos_u_local[idx2]
    pi2 = pos_i[idx2]
    ni2 = torch.randint(0, feats.num_items, (len(idx2),))

    optimizer.zero_grad()
    x2, _, _, _ = model.encode(ei_gpu, global_gpu, hist_gpu, feats)
    ps2 = model.decode(x2, pu2, pi2, cfg.device)
    ns2 = model.decode(x2, pu2, ni2, cfg.device)
    bpr2 = -F.logsigmoid(ps2 - ns2).mean()
    ewc_val = torch.tensor(0., device=cfg.device)
    if ewc is not None:
        ewc_val = ewc.penalty(model, warmup_factor=warmup)
    (bpr2 + ewc_val).backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    del x2, ps2, ns2, bpr2
    if cfg.device == 'cuda':
        torch.cuda.empty_cache()

    return tot / max(steps, 1), ewc_val.item(), gates


def simple_train_epoch(cfg: Config, model: HeteroMGAT, optimizer, ei_gpu: Dict,
                       global_gpu: torch.Tensor, hist_gpu: torch.Tensor,
                       feats: ItemFeatures, pos_u_local: torch.Tensor,
                       pos_i: torch.Tensor, N_: int) -> float:
    """Baseline training (no EWC, no growth) — used by static, ewc_only,
    AND the cumulative-no-EWC baseline."""
    model.train()
    with torch.no_grad():
        x_det, _, _, _ = model.encode(ei_gpu, global_gpu, hist_gpu, feats)
    if cfg.device == 'cuda':
        torch.cuda.empty_cache()

    graph_conv_names = {n for n, _ in model.graph_conv_parameters()}
    for name, p in model.named_parameters():
        p.requires_grad_(name not in graph_conv_names)

    perm = torch.randperm(N_)
    tot, steps = 0.0, 0
    for s in range(0, N_, cfg.BATCH):
        idx = perm[s:s + cfg.BATCH]
        pu_local = pos_u_local[idx]
        pi = pos_i[idx]
        ni = torch.randint(0, feats.num_items, (len(idx),))
        optimizer.zero_grad()
        ps = model.decode(x_det, pu_local, pi, cfg.device)
        ns = model.decode(x_det, pu_local, ni, cfg.device)
        bpr = -F.logsigmoid(ps - ns).mean()
        bpr.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot += bpr.item(); steps += 1
        del ps, ns, bpr, pu_local, pi, ni

    del x_det
    for p in model.parameters():
        p.requires_grad_(True)
    if cfg.device == 'cuda':
        torch.cuda.empty_cache()
    return tot / max(steps, 1)


def run_period_training(cfg: Config, dm: DataModule, evaluator: Evaluator,
                        model: HeteroMGAT, optimizer, train_df: pd.DataFrame,
                        cumtrain_ei, epochs: int, ewc: Optional[EWC],
                        past_valid_records: List[dict], seed: int, tag: str = '',
                        complexity_tracker: Optional[ComplexityTracker] = None
                        ) -> Tuple[dict, EWC, float, List[torch.Tensor], List[float]]:
    """
    Returns (valid_log, new_ewc, final_bpr, gate_history, bpr_history).
    """
    feats = dm.item_features
    ei, global_u, n_u_local, local_u_map, _ = cumtrain_ei
    mask = train_df['u_idx'].apply(lambda u: int(u) in local_u_map)
    train_f = train_df[mask].reset_index(drop=True)
    pos_u_local = torch.tensor(
        [local_u_map[int(v)] for v in train_f['u_idx'].values], dtype=torch.long
    )
    pos_i = torch.tensor(train_f['i_idx'].values, dtype=torch.long)
    N_ = len(pos_u_local)
    hist_ten = dm.build_hist_tensor(n_u_local, local_u_map)

    print(f'  [{tag}]  edges={N_:,}  users={n_u_local:,}  '
          f'heads={model.num_heads}  H={model.hidden}')

    ei_gpu, global_gpu, hist_gpu = dm.to_gpu(ei, global_u, hist_ten)

    gate_history: List[torch.Tensor] = []
    bpr_history: List[float] = []
    avg_bpr = 0.0

    for epoch in range(1, epochs + 1):
        warmup = min(1.0, epoch / cfg.EWC_WARMUP_EPOCHS) if ewc else 1.0
        if complexity_tracker: complexity_tracker.start_epoch()
        avg_bpr, ewc_val, gates = train_one_epoch(
            cfg, model, optimizer, ei_gpu, global_gpu, hist_gpu, feats,
            pos_u_local, pos_i, N_, ewc=ewc, warmup=warmup,
        )
        if complexity_tracker: complexity_tracker.end_epoch()
        gate_history.append(gates)
        bpr_history.append(avg_bpr)
        ep_str = f'  ewc={ewc_val:.4f}(×{warmup:.2f})' if ewc else ''
        print(f'    epoch {epoch:02d}/{epochs}  bpr={avg_bpr:.4f}{ep_str}')

    print('  → Capturing Fisher...')
    snap, F_ = accumulate_fisher(
        cfg, model, ei_gpu, global_gpu, hist_gpu, feats, train_f, local_u_map
    )
    new_ewc = EWC(cfg, snap, F_)

    print(f'  → Validating ({len(past_valid_records)} period(s))...')
    valid_log = {'tag': tag}
    forgetting_detected = False

    for rec in past_valid_records:
        pidx = rec['period_idx']
        vm = evaluator.run_eval(model, rec['valid_df'], rec['cum_ei'],
                                rec['ref_train_df'], seed=seed)
        valid_log[f'P{pidx}_r20'] = vm['recall@20']
        valid_log[f'P{pidx}_auc'] = vm['auc']
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()

        if pidx != past_valid_records[-1]['period_idx']:
            base = rec.get('best_r20') or 0.0
            if base > 0:
                drop = (base - vm['recall@20']) / base
                if drop > cfg.FORGET_THR:
                    forgetting_detected = True
                    print(f'    ⚠ Forgetting P{pidx}: '
                          f'{base:.4f}→{vm["recall@20"]:.4f} ({drop:.1%})')
        if vm['recall@20'] > (rec.get('best_r20') or 0.0):
            rec['best_r20'] = vm['recall@20']

    parts = []
    for r in past_valid_records:
        k = f'P{r["period_idx"]}_r20'
        parts.append(f'P{r["period_idx"]} R20={valid_log.get(k, 0):.4f}')
    print(f'  Valid: {"  ".join(parts)}')

    if forgetting_detected and ewc is not None:
        ewc.boost()

    del ei_gpu, global_gpu, hist_gpu
    if cfg.device == 'cuda':
        torch.cuda.empty_cache()

    return valid_log, new_ewc, avg_bpr, gate_history, bpr_history
