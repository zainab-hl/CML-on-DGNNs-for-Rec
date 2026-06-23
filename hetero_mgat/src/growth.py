"""
growth.py
=========
Severity-based growth controller.

Signals
──────
S1 — Forgetting     (avg BWT below -FORGET_THR)
S2 — Modal dominance (4-modality gate spread > 1/(n_mod+1); independent,
                      resets gate weights, no contribution to sev_*)
S3 — BPR plateau    (relative improvement < PLATEAU_REL_DROP)
S4 — Repr collapse  (mean pairwise cosine > REPR_COLLAPSE_THR)
S5 — Over-smoothing (layer0 vs layer2 cosine > OVERSMOOTH_THR)

Severity scores
──────────────
sev_width = 0.7·S3 + 0.5·S5 + 0.2·S1 + 0.1·S4   → grow hidden
  (capacity saturated: plateau, oversmooth)
sev_depth = 0.7·S4 + 0.5·S1 + 0.2·S5 + 0.1·S3   → grow attention heads
  (representations collapse / forgetting is high)

NOTE: MGAT's GAT depth (2 layers) is architecture-fixed. "Depth" growth
here means growing attention heads, NOT adding GATConv layers.

Growth decision (threshold = cfg.SEV_THRESHOLD)
─────────────────────────────────────────────────
sev_width ≥ THR  AND  sev_depth < THR  → grow hidden only
sev_depth ≥ THR  AND  sev_width < THR  → grow heads only
both ≥ THR                             → grow both
both < THR                             → no growth

S2 is independent: it resets the fusion gate weights each time it fires
but does NOT contribute to sev_width / sev_depth.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import Config
from .model import HeteroMGAT, ItemFeatures, build_dynamic, reset_gate_weights
from .signals import signal_bpr_plateau, signal_repr_and_smooth


class GrowthController:
    def __init__(self, cfg: Config, num_brands: int, num_main: int):
        self.cfg = cfg
        self.num_brands = num_brands
        self.num_main = num_main
        self.log: List[dict] = []
        self._modal_low_streak = 0
        self._hidden = cfg.HIDDEN_INIT
        self._heads = cfg.HEADS_INIT

    # ── S2: gate-spread reset ────────────────────────────────────────────
    @staticmethod
    def _reset_gate_bias(model: HeteroMGAT) -> None:
        reset_gate_weights(model)
        print('    [S2] Gate weights reset to uniform.')

    # ── Main update ──────────────────────────────────────────────────────
    def update(self, model: HeteroMGAT, ewc, ei_gpu: Dict,
               global_u_gpu: torch.Tensor, hist_gpu: torch.Tensor,
               feats: ItemFeatures, avg_bwt: float, bpr_history: List[float],
               gate_history: List[torch.Tensor]) -> Tuple[HeteroMGAT, bool, List[str], bool]:
        """Returns: (model, grew, reasons_str_list, grew_hidden_flag)"""
        cfg = self.cfg

        # S1 — forgetting
        s1 = 0.0
        if not np.isnan(avg_bwt) and avg_bwt < -cfg.FORGET_THR:
            s1 = min(abs(avg_bwt) / cfg.FORGET_THR, 2.0)

        # S2 — modal dominance (4-modality gate spread)
        gate_vals = [0.25, 0.25, 0.25, 0.25]
        spread = 0.0
        s2_fired = False
        if gate_history:
            mg = torch.stack(gate_history).mean(0).mean(0)  # [4]
            gate_vals = mg.tolist()
            spread = max(gate_vals) - min(gate_vals)
            if spread > cfg.S2_SPREAD_THR:
                self._modal_low_streak += 1
            else:
                self._modal_low_streak = 0
            if self._modal_low_streak >= cfg.MODAL_DOM_CONSEC:
                s2_fired = True

        # S3 — BPR plateau
        s3_fired, plateau_str = signal_bpr_plateau(cfg, bpr_history)
        s3 = 1.0 if s3_fired else 0.0

        # S4 + S5: fused single encode pass
        repr_cos, smooth_cos = signal_repr_and_smooth(
            cfg, model, ei_gpu, global_u_gpu, hist_gpu, feats)
        s4 = 1.0 if repr_cos > cfg.REPR_COLLAPSE_THR else 0.0
        s5 = 1.0 if smooth_cos > cfg.OVERSMOOTH_THR else 0.0

        # ── Severity scores ──────────────────────────────────────────────
        sev_width = 0.7 * s3 + 0.5 * s5 + 0.2 * s1 + 0.1 * s4
        sev_depth = 0.7 * s4 + 0.5 * s1 + 0.2 * s5 + 0.1 * s3

        gate_str = '/'.join(f'{g:.3f}' for g in gate_vals)
        print(f'  Signals:'
              f'  bwt={avg_bwt:+.4f}(s1={s1:.2f})'
              f'  gates=[{gate_str}]'
              f'  spread={spread:.3f}>thr={cfg.S2_SPREAD_THR:.3f}(streak={self._modal_low_streak})'
              f'  s3={s3_fired}({plateau_str})'
              f'  repr_cos={repr_cos:.3f}(s4={s4:.0f})'
              f'  smooth_cos={smooth_cos:.3f}(s5={s5:.0f})')
        print(f'  Severity: sev_width={sev_width:.3f}  sev_depth={sev_depth:.3f}'
              f'  (thr={cfg.SEV_THRESHOLD})')

        # ── S2 gate reset (always, independent) ─────────────────────────
        if s2_fired:
            self._reset_gate_bias(model)
            self._modal_low_streak = 0
            print(f'  [S2 fired] spread={spread:.3f}>{cfg.S2_SPREAD_THR:.3f}')

        # ── Growth decision matrix ───────────────────────────────────────
        at_heads_max = self._heads >= cfg.HEADS_MAX
        at_hidden_max = self._hidden >= cfg.HIDDEN_MAX

        grow_width = (sev_width >= cfg.SEV_THRESHOLD) and not at_hidden_max
        grow_depth = (sev_depth >= cfg.SEV_THRESHOLD) and not at_heads_max

        reasons = []
        if s1 > 0: reasons.append(f'S1_forgetting(bwt={avg_bwt:.4f},s1={s1:.2f})')
        if s2_fired: reasons.append(f'S2_modal_dom(spread={spread:.3f},streak={self._modal_low_streak})')
        if s3_fired: reasons.append(f'S3_bpr_plateau({plateau_str})')
        if s4: reasons.append(f'S4_repr_collapse(cos={repr_cos:.3f})')
        if s5: reasons.append(f'S5_oversmooth(cos={smooth_cos:.3f})')
        reasons.append(f'sev_w={sev_width:.3f}  sev_d={sev_depth:.3f}')

        if not grow_width and not grow_depth:
            print(f'  [NO GROWTH]  Fired: {[r for r in reasons[:-1]] or ["none"]}')
            return model, False, [], False

        action = ('width+depth(heads)' if grow_width and grow_depth
                  else 'width' if grow_width else 'depth(heads)')
        print(f'  [GROWTH: {action}]  Fired: {[r for r in reasons[:-1]]}')

        grew_hidden_flag = grow_width
        model = self._grow(model, grow_width=grow_width, grow_depth=grow_depth)
        self.log.append({
            'old_heads': self._heads - (1 if grow_depth else 0),
            'new_heads': self._heads,
            'old_hidden': self._hidden // (2 if grow_width else 1),
            'new_hidden': self._hidden,
            'action': action,
            'sev_width': round(sev_width, 4),
            'sev_depth': round(sev_depth, 4),
            'reasons': '; '.join(reasons),
        })
        return model, True, reasons, grew_hidden_flag

    # ── _grow: explicit width / depth(heads) axes ───────────────────────
    def _grow(self, model: HeteroMGAT, grow_width: bool, grow_depth: bool) -> HeteroMGAT:
        cfg = self.cfg
        old_heads = self._heads
        old_hidden = self._hidden

        new_heads = (self._heads + 1) if grow_depth else self._heads
        new_hidden = min(self._hidden * 2, cfg.HIDDEN_MAX) if grow_width else self._hidden

        new_heads = min(new_heads, cfg.HEADS_MAX)
        new_hidden = min(new_hidden, cfg.HIDDEN_MAX)

        new_model = build_dynamic(cfg, self.num_brands, self.num_main,
                                  hidden=new_hidden, heads=new_heads)
        old_sd = model.state_dict()
        new_sd = new_model.state_dict()
        copied = 0
        for k in new_sd:
            if k in old_sd and old_sd[k].shape == new_sd[k].shape:
                new_sd[k] = old_sd[k].clone()
                copied += 1
        new_model.load_state_dict(new_sd)

        # GATE-3: reset gate to uniform after copying potentially
        # imbalanced weights into the new model.
        reset_gate_weights(new_model)
        print(f'  [GATE-3] Gate weights reset to uniform after growth.')

        print(f'  [GROWTH] heads {old_heads}→{new_heads}  '
              f'hidden {old_hidden}→{new_hidden}'
              f'  (width={grow_width} depth={grow_depth})'
              f'  ({copied} tensors copied)')

        self._heads = new_heads
        self._hidden = new_hidden
        del model
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()
        return new_model.to(cfg.device)
