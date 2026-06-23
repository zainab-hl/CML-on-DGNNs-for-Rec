from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .ewc import EWC
from .model import LightGCN
from .signals import (
    signal_bpr_plateau,
    signal_oversmooth,
    signal_repr_collapse,
)


class GrowthController:
    """
    Severity-based adaptive growth controller.

    Signals
    -------
    S1 — Forgetting      (avg BWT < -forget_thr)
    S2 — Modal dominance (gate spread > 1/(n_mod+1); resets gate bias only)
    S3 — BPR plateau     (relative improvement < plateau_rel_drop)
    S4 — Repr collapse   (mean pairwise cosine > repr_collapse_thr)
    S5 — Over-smoothing  (layer-0 vs layer-K cosine > oversmooth_thr)

    Severity scores
    ---------------
    sev_width = 0.7·S3 + 0.5·S5 + 0.2·S1 + 0.1·S4
    sev_depth = 0.7·S4 + 0.5·S1 + 0.2·S5 + 0.1·S3

    S2 acts independently: fires a gate-bias reset but never contributes to
    sev_width / sev_depth.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log: list[dict] = []
        self._modal_low_streak = 0
        self._hidden = cfg.hidden_init
        self._K = cfg.k_init

    @staticmethod
    def _reset_gate_bias(model: LightGCN):
        gate_linear = model.fusion.gate[-2]
        if isinstance(gate_linear, nn.Linear):
            nn.init.constant_(gate_linear.weight, 0.0)
            if gate_linear.bias is not None:
                nn.init.zeros_(gate_linear.bias)
        print("    [S2] Gate bias reset to uniform.")

    def update(
        self,
        model: LightGCN,
        ewc: Optional[EWC],
        adj,
        global_u_gpu: torch.Tensor,
        avg_bwt: float,
        bpr_history: list,
        gate_history: list,
    ):
        cfg = self.cfg

        # S1 — forgetting
        s1 = 0.0
        if not np.isnan(avg_bwt) and avg_bwt < -cfg.forget_thr:
            s1 = min(abs(avg_bwt) / cfg.forget_thr, 2.0)

        # S2 — modal dominance
        text_w = img_w = 0.5
        s2_fired = False
        spread = 0.0
        if gate_history:
            mg = torch.stack(gate_history).mean(0).mean(0)
            text_w, img_w = mg[0].item(), mg[1].item()
            spread = max(text_w, img_w) - min(text_w, img_w)
            if spread > cfg.s2_spread_thr:
                self._modal_low_streak += 1
            else:
                self._modal_low_streak = 0
            if self._modal_low_streak >= cfg.modal_dom_consec:
                s2_fired = True

        # S3 — BPR plateau
        s3_fired, plateau_str = signal_bpr_plateau(bpr_history, cfg.plateau_window, cfg.plateau_rel_drop)
        s3 = 1.0 if s3_fired else 0.0

        # S4 — repr collapse
        repr_cos = signal_repr_collapse(model, adj, global_u_gpu, cfg.repr_collapse_sample)
        s4 = 1.0 if repr_cos > cfg.repr_collapse_thr else 0.0

        # S5 — over-smoothing
        smooth_cos = signal_oversmooth(model, adj, global_u_gpu, cfg.oversmooth_sample)
        s5 = 1.0 if smooth_cos > cfg.oversmooth_thr else 0.0

        sev_width = 0.7 * s3 + 0.5 * s5 + 0.2 * s1 + 0.1 * s4
        sev_depth = 0.7 * s4 + 0.5 * s1 + 0.2 * s5 + 0.1 * s3

        print(
            f"  Signals:"
            f"  bwt={avg_bwt:+.4f}(s1={s1:.2f})"
            f"  text={text_w:.3f}  img={img_w:.3f}"
            f"  spread={spread:.3f}>thr={cfg.s2_spread_thr:.3f}(streak={self._modal_low_streak})"
            f"  s3={s3_fired}({plateau_str})"
            f"  repr_cos={repr_cos:.3f}(s4={s4:.0f})"
            f"  smooth_cos={smooth_cos:.3f}(s5={s5:.0f})"
        )
        print(
            f"  Severity: sev_width={sev_width:.3f}  sev_depth={sev_depth:.3f}"
            f"  (thr={cfg.sev_threshold})"
        )

        if s2_fired:
            self._reset_gate_bias(model)
            self._modal_low_streak = 0
            print(f"  [S2 fired] spread={spread:.3f}>{cfg.s2_spread_thr:.3f}")

        grow_width = (sev_width >= cfg.sev_threshold) and (self._hidden < cfg.hidden_max)
        grow_depth = (sev_depth >= cfg.sev_threshold) and (self._K < cfg.k_max)

        reasons = []
        if s1 > 0:     reasons.append(f"S1_forgetting(bwt={avg_bwt:.4f},s1={s1:.2f})")
        if s2_fired:   reasons.append(f"S2_modal_dom(spread={spread:.3f},streak={self._modal_low_streak})")
        if s3_fired:   reasons.append(f"S3_bpr_plateau({plateau_str})")
        if s4:         reasons.append(f"S4_repr_collapse(cos={repr_cos:.3f})")
        if s5:         reasons.append(f"S5_oversmooth(cos={smooth_cos:.3f})")
        reasons.append(f"sev_w={sev_width:.3f}  sev_d={sev_depth:.3f}")

        if not grow_width and not grow_depth:
            print(f"  [NO GROWTH]  Fired: {reasons[:-1] or ['none']}")
            return model, False, [], False

        action = "width+depth" if (grow_width and grow_depth) else ("width" if grow_width else "depth")
        print(f"  [GROWTH: {action}]  Fired: {reasons[:-1]}")

        model = self._grow(model, grow_width=grow_width, grow_depth=grow_depth)
        self.log.append({
            "old_K": self._K - (1 if grow_depth else 0),
            "new_K": self._K,
            "old_hidden": self._hidden // (2 if grow_width else 1),
            "new_hidden": self._hidden,
            "action": action,
            "sev_width": round(sev_width, 4),
            "sev_depth": round(sev_depth, 4),
            "reasons": "; ".join(reasons),
        })
        return model, True, reasons, grow_width

    def _grow(self, model: LightGCN, grow_width: bool, grow_depth: bool) -> LightGCN:
        cfg = self.cfg
        old_K, old_hidden = self._K, self._hidden
        new_K = min(self._K + 1, cfg.k_max) if grow_depth else self._K
        new_hidden = min(self._hidden * 2, cfg.hidden_max) if grow_width else self._hidden

        new_model = LightGCN(
            model.num_users, model.num_items,
            model.T_text, model.T_image, cfg,
            hidden=new_hidden, K=new_K,
        ).to(cfg.device)

        old_sd, new_sd = model.state_dict(), new_model.state_dict()
        copied = 0
        for k in new_sd:
            if k in old_sd and old_sd[k].shape == new_sd[k].shape:
                new_sd[k] = old_sd[k].clone()
                copied += 1
        new_model.load_state_dict(new_sd)

        print(
            f"  [GROWTH] K {old_K}→{new_K}  hidden {old_hidden}→{new_hidden}"
            f"  (width={grow_width} depth={grow_depth})  ({copied} tensors copied)"
        )
        self._K = new_K
        self._hidden = new_hidden
        del model
        if cfg.device == "cuda":
            torch.cuda.empty_cache()
        return new_model
