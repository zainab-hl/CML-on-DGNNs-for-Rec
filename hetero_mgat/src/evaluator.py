"""
evaluator.py
============
Recall@K / NDCG / MRR / AUC evaluation, with cold-user mean-item-
embedding fallback and warm/cold breakdown.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import Config
from .data import DataModule
from .model import HeteroMGAT


def _empty() -> Dict[str, float]:
    return {k: 0. for k in ['recall@10', 'recall@20', 'ndcg@10', 'mrr', 'auc',
                             'recall@20_warm', 'recall@20_cold',
                             'n_warm', 'n_cold']}


def fmt(m: Dict[str, float]) -> str:
    return (f'R@10={m["recall@10"]:.4f}  R@20={m["recall@20"]:.4f}  '
            f'NDCG@10={m["ndcg@10"]:.4f}  MRR={m["mrr"]:.4f}  '
            f'AUC={m["auc"]:.4f}  '
            f'[warm={m["recall@20_warm"]:.4f} cold={m["recall@20_cold"]:.4f} '
            f'nw={int(m["n_warm"])} nc={int(m["n_cold"])}]')


class Evaluator:
    """Bound to a DataModule so it has access to NUM_ITEMS and the item
    feature cache without needing them passed at every call site."""

    def __init__(self, cfg: Config, dm: DataModule):
        self.cfg = cfg
        self.dm = dm

    @torch.no_grad()
    def run_eval(self, model: HeteroMGAT, eval_df: pd.DataFrame,
                cum_ei_tuple, ref_train_df: pd.DataFrame,
                label: str = '', seed: int = 0) -> Dict[str, float]:
        """
        Cold users get mean-item-embedding as user representation.
        cum_ei_tuple MUST be the cumulative edge_index for this period.
        """
        cfg, dm = self.cfg, self.dm
        if eval_df is None or len(eval_df) == 0:
            return _empty()
        model.eval()

        ei, global_u, n_u_local, local_u_map, _ = cum_ei_tuple
        hist_ten = dm.build_hist_tensor(n_u_local, local_u_map)
        ei_gpu, global_gpu, hist_gpu = dm.to_gpu(ei, global_u, hist_ten)

        x_dict, _, _, _ = model.encode_eval(ei_gpu, global_gpu, hist_gpu, dm.item_features)
        item_cpu = x_dict['item']
        user_cpu = x_dict['user']

        mean_item_emb = item_cpu.mean(dim=0, keepdim=True)  # (1, hidden)

        del ei_gpu, global_gpu, hist_gpu
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()

        gt_all = eval_df.groupby('u_idx')['i_idx'].apply(set).to_dict()
        train_pos = ref_train_df.groupby('u_idx')['i_idx'].apply(set).to_dict()

        all_users = list(gt_all.keys())
        warm_users = [u for u in all_users if int(u) in local_u_map]
        cold_users = [u for u in all_users if int(u) not in local_u_map]

        if label:
            print(f'    [{label}] warm={len(warm_users)}  cold={len(cold_users)}  '
                  f'cold_frac={len(cold_users) / max(len(all_users), 1):.0%}')

        rng = np.random.default_rng(seed)
        if len(warm_users) > cfg.EVAL_USERS:
            warm_eval = rng.choice(warm_users, cfg.EVAL_USERS, replace=False).tolist()
        else:
            warm_eval = warm_users

        n_cold_eval = min(len(cold_users), cfg.EVAL_USERS)
        if n_cold_eval > 0:
            cold_eval = rng.choice(cold_users, n_cold_eval, replace=False).tolist()
        else:
            cold_eval = []

        def score_all_items(u_rep):
            scores_list = []
            for s in range(0, dm.NUM_ITEMS, 4096):
                e = min(s + 4096, dm.NUM_ITEMS)
                u_g = u_rep.expand(e - s, -1).to(cfg.device)
                i_g = item_cpu[s:e].to(cfg.device)
                sc = model.predictor(
                    torch.cat([u_g, i_g], dim=1)
                ).squeeze(-1).detach().cpu().numpy()
                scores_list.append(sc)
                del u_g, i_g
            return np.concatenate(scores_list)

        def eval_user_list(ulist, is_cold=False):
            r10_l, r20_l, ndcg10_l, mrr_l = [], [], [], []
            auc_lab, auc_sc = [], []
            for u in ulist:
                u = int(u)
                if is_cold:
                    u_rep = mean_item_emb
                else:
                    local_idx = local_u_map[u]
                    u_rep = user_cpu[local_idx].unsqueeze(0)

                scores = score_all_items(u_rep)
                pos = gt_all[u]
                top20 = scores.argsort()[::-1][:20].tolist()

                r10_l.append(len(set(top20[:10]) & pos) / max(len(pos), 1))
                r20_l.append(len(set(top20[:20]) & pos) / max(len(pos), 1))

                hits = [1 if top20[r] in pos else 0 for r in range(min(10, len(top20)))]
                dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
                idcg = sum(1 / np.log2(r + 2) for r in range(min(len(pos), 10)))
                ndcg10_l.append(dcg / max(idcg, 1e-8))

                rr = 0.
                for rank, item in enumerate(top20):
                    if item in pos:
                        rr = 1. / (rank + 1)
                        break
                mrr_l.append(rr)

                excl = train_pos.get(u, set())
                for p_item in list(pos):
                    pool = [i for i in
                            rng.choice(dm.NUM_ITEMS, cfg.NEG_SAMPLES * 4, replace=False)
                            if i not in excl][:cfg.NEG_SAMPLES]
                    if len(pool) < 5:
                        continue
                    auc_lab.append(np.array([1] + [0] * len(pool)))
                    auc_sc.append(np.array([scores[p_item]] + [scores[n] for n in pool]))

            return r10_l, r20_l, ndcg10_l, mrr_l, auc_lab, auc_sc

        r10_w, r20_w, ndcg_w, mrr_w, alab_w, asc_w = eval_user_list(warm_eval, is_cold=False)
        r10_c, r20_c, ndcg_c, mrr_c, alab_c, asc_c = eval_user_list(cold_eval, is_cold=True)

        all_labs = alab_w + alab_c
        all_scs = asc_w + asc_c
        try:
            auc = roc_auc_score(np.concatenate(all_labs), np.concatenate(all_scs))
        except Exception:
            auc = 0.5

        n_w = len(r20_w)
        n_c = len(r20_c)
        n_total = max(n_w + n_c, 1)

        def wmean(lw, lc):
            if not lw and not lc:
                return 0.
            arr = lw + lc
            return float(np.mean(arr))

        r20_warm_val = float(np.mean(r20_w)) if r20_w else 0.
        r20_cold_val = float(np.mean(r20_c)) if r20_c else 0.
        r20_combined = (n_w * r20_warm_val + n_c * r20_cold_val) / n_total

        if label:
            print(f'    [{label}] R20_warm={r20_warm_val:.4f}  '
                  f'R20_cold={r20_cold_val:.4f}  R20_combined={r20_combined:.4f}')

        del item_cpu, user_cpu, x_dict, mean_item_emb
        if cfg.device == 'cuda':
            torch.cuda.empty_cache()

        return {
            'recall@10': round(wmean(r10_w, r10_c), 4),
            'recall@20': round(r20_combined, 4),
            'ndcg@10': round(wmean(ndcg_w, ndcg_c), 4),
            'mrr': round(wmean(mrr_w, mrr_c), 4),
            'auc': round(float(auc), 4),
            'recall@20_warm': round(r20_warm_val, 4),
            'recall@20_cold': round(r20_cold_val, 4),
            'n_warm': n_w,
            'n_cold': n_c,
        }
