from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .config import Config
from .data import DataModule
from .model import LightGCN


class Evaluator:
    """
    Ranking evaluation: Recall@10/20, NDCG@10, MRR, AUC.
    Supports warm users (seen during training) and cold users
    (mean-item-embedding fallback).
    """

    def __init__(self, cfg: Config, data: DataModule):
        self.cfg = cfg
        self.data = data

    @staticmethod
    def _empty() -> dict:
        return {k: 0.0 for k in [
            "recall@10", "recall@20", "ndcg@10", "mrr", "auc",
            "recall@20_warm", "recall@20_cold", "n_warm", "n_cold",
        ]}

    @staticmethod
    def fmt(m: dict) -> str:
        return (
            f"R@10={m['recall@10']:.4f}  R@20={m['recall@20']:.4f}  "
            f"NDCG@10={m['ndcg@10']:.4f}  MRR={m['mrr']:.4f}  AUC={m['auc']:.4f}  "
            f"[warm={m['recall@20_warm']:.4f} cold={m['recall@20_cold']:.4f} "
            f"nw={int(m['n_warm'])} nc={int(m['n_cold'])}]"
        )

    @torch.no_grad()
    def run_eval(
        self,
        model: LightGCN,
        eval_df: Optional[object],
        cum_adj,
        ref_train_df,
        seed: int = 0,
        label: str = "",
    ) -> dict:
        cfg = self.cfg
        data = self.data
        if eval_df is None or len(eval_df) == 0:
            return self._empty()
        model.eval()

        unique_u = np.unique(ref_train_df["u_idx"].values)
        local_u_map = {int(g): l for l, g in enumerate(unique_u.tolist())}
        global_u = torch.from_numpy(unique_u).long().to(cfg.device)

        x_user, x_item, _, _, _ = model.encode_eval(cum_adj, global_u)
        x_user, x_item = x_user.cpu(), x_item.cpu()
        mean_item_emb = x_item.mean(dim=0, keepdim=True)
        del global_u
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

        gt_all = eval_df.groupby("u_idx")["i_idx"].apply(set).to_dict()
        train_pos = ref_train_df.groupby("u_idx")["i_idx"].apply(set).to_dict()

        all_users = list(gt_all.keys())
        warm_users = [u for u in all_users if int(u) in local_u_map]
        cold_users = [u for u in all_users if int(u) not in local_u_map]

        rng = np.random.default_rng(seed)
        warm_eval = (
            rng.choice(warm_users, cfg.eval_users, replace=False).tolist()
            if len(warm_users) > cfg.eval_users else warm_users
        )
        n_cold_eval = min(len(cold_users), cfg.eval_users)
        cold_eval = (
            rng.choice(cold_users, n_cold_eval, replace=False).tolist()
            if n_cold_eval > 0 else []
        )

        def eval_user_list(ulist, is_cold=False):
            r10_l, r20_l, ndcg10_l, mrr_l = [], [], [], []
            auc_lab, auc_sc = [], []
            for u in ulist:
                u = int(u)
                u_rep = mean_item_emb if is_cold else x_user[local_u_map[u]].unsqueeze(0)
                scores = (u_rep * x_item).sum(dim=-1).numpy()
                pos = gt_all[u]
                top20 = scores.argsort()[::-1][:20].tolist()

                r10_l.append(len(set(top20[:10]) & pos) / max(len(pos), 1))
                r20_l.append(len(set(top20[:20]) & pos) / max(len(pos), 1))

                hits = [1 if top20[r] in pos else 0 for r in range(min(10, len(top20)))]
                dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
                idcg = sum(1 / np.log2(r + 2) for r in range(min(len(pos), 10)))
                ndcg10_l.append(dcg / max(idcg, 1e-8))

                rr = 0.0
                for rank, item in enumerate(top20):
                    if item in pos:
                        rr = 1.0 / (rank + 1)
                        break
                mrr_l.append(rr)

                excl = train_pos.get(u, set())
                for p_item in list(pos):
                    pool = [i for i in rng.choice(data.num_items, cfg.neg_samples * 4, replace=False)
                            if i not in excl][: cfg.neg_samples]
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

        n_w, n_c = len(r20_w), len(r20_c)
        n_total = max(n_w + n_c, 1)

        def wmean(lw, lc):
            return float(np.mean(lw + lc)) if (lw or lc) else 0.0

        r20_warm_val = float(np.mean(r20_w)) if r20_w else 0.0
        r20_cold_val = float(np.mean(r20_c)) if r20_c else 0.0
        r20_combined = (n_w * r20_warm_val + n_c * r20_cold_val) / n_total

        if label:
            print(f"    [{label}] warm={n_w}  cold={n_c}  "
                  f"R20_warm={r20_warm_val:.4f}  R20_cold={r20_cold_val:.4f}  "
                  f"R20_combined={r20_combined:.4f}")

        del x_user, x_item, mean_item_emb
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

        return {
            "recall@10": round(wmean(r10_w, r10_c), 4),
            "recall@20": round(r20_combined, 4),
            "ndcg@10": round(wmean(ndcg_w, ndcg_c), 4),
            "mrr": round(wmean(mrr_w, mrr_c), 4),
            "auc": round(float(auc), 4),
            "recall@20_warm": round(r20_warm_val, 4),
            "recall@20_cold": round(r20_cold_val, 4),
            "n_warm": n_w,
            "n_cold": n_c,
        }
