from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch_sparse import SparseTensor

from .config import Config


class DataModule:
    """Handles loading, embedding projection, graph construction, and temporal splits."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.inter: pd.DataFrame
        self.pretrain_df: pd.DataFrame
        self.period_dfs: list[pd.DataFrame]
        self.T_text: torch.Tensor
        self.T_image: torch.Tensor
        self.num_users: int
        self.num_items: int
        self.user2idx: dict
        self.item2idx: dict

    def load(self):
        self._load_interactions()
        self._load_embeddings()
        self._build_temporal_splits()
        print(f"Interactions : {len(self.inter):,}")
        print(f"Users : {self.num_users:,}  Items : {self.num_items:,}")

    # ── Interactions ───────────────────────────────────────────────────────

    def _load_interactions(self):
        cfg = self.cfg
        inter = pd.read_csv(cfg.data_dir / f"{cfg.dataset}.inter", sep="\t")
        inter.columns = inter.columns.str.strip()
        inter = inter.rename(columns={c: c.split(":")[0].strip() for c in inter.columns})
        inter = (
            inter[["userID", "itemID", "rating", "timestamp"]]
            .dropna()
            .reset_index(drop=True)
        )
        inter["rating"] = inter["rating"].astype(float)
        inter["timestamp"] = inter["timestamp"].astype(np.int64)

        unique_users = sorted(inter["userID"].unique())
        unique_items = sorted(inter["itemID"].unique())
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.item2idx = {it: i for i, it in enumerate(unique_items)}
        inter["u_idx"] = inter["userID"].map(self.user2idx)
        inter["i_idx"] = inter["itemID"].map(self.item2idx)

        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

        inter["datetime"] = pd.to_datetime(inter["timestamp"], unit="s", utc=True)
        self.inter = inter.sort_values("timestamp").reset_index(drop=True)

    # ── Embeddings ─────────────────────────────────────────────────────────

    def _load_embeddings(self):
        cfg = self.cfg
        text_raw = np.load(cfg.data_dir / "text_feat.npy").astype(np.float32)
        image_raw = np.load(cfg.data_dir / "image_feat.npy").astype(np.float32)
        emb_n = text_raw.shape[0]

        image_raw /= np.linalg.norm(image_raw, axis=1, keepdims=True) + 1e-8

        pca_t = PCA(n_components=cfg.text_dim_out, random_state=42).fit_transform(text_raw)
        pca_i = PCA(n_components=cfg.image_dim_out, random_state=42).fit_transform(image_raw)

        self.T_text = torch.zeros(self.num_items, cfg.text_dim_out, dtype=torch.float32)
        self.T_image = torch.zeros(self.num_items, cfg.image_dim_out, dtype=torch.float32)
        for idx in range(min(emb_n, self.num_items)):
            self.T_text[idx] = torch.from_numpy(pca_t[idx])
            self.T_image[idx] = torch.from_numpy(pca_i[idx])

        del text_raw, image_raw, pca_t, pca_i
        gc.collect()

    # ── Temporal splits ────────────────────────────────────────────────────

    def _build_temporal_splits(self):
        cfg = self.cfg
        self.pretrain_df = self.inter[self.inter["datetime"] < cfg.pretrain_cut].copy()

        cuts = [cfg.pretrain_cut] + cfg.period_cuts
        self.period_dfs = []
        for p in range(cfg.num_periods):
            lo = cuts[p]
            hi = cuts[p + 1] if p + 1 < len(cuts) else None
            pdf = (
                self.inter[self.inter["datetime"] >= lo]
                if hi is None
                else self.inter[
                    (self.inter["datetime"] >= lo) & (self.inter["datetime"] < hi)
                ]
            )
            self.period_dfs.append(pdf.copy().reset_index(drop=True))

    def chrono_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg
        n = len(df)
        n_tr = int(n * cfg.train_ratio)
        n_va = int(n * cfg.valid_ratio)
        return (
            df.iloc[:n_tr].copy(),
            df.iloc[n_tr: n_tr + n_va].copy(),
            df.iloc[n_tr + n_va:].copy(),
        )

    # ── Graph construction ─────────────────────────────────────────────────

    def make_graph(self, df: pd.DataFrame) -> tuple:
        """Return (adj, global_u, local_u_map) for the given interaction DataFrame."""
        uarr = df["u_idx"].values.astype(np.int64)
        iarr = df["i_idx"].values.astype(np.int64) + self.num_users
        row = np.concatenate([uarr, iarr])
        col = np.concatenate([iarr, uarr])
        N = self.num_users + self.num_items

        deg = np.bincount(row, minlength=N).astype(np.float32)
        deg_inv = np.where(deg > 0, deg**-0.5, 0.0)
        vals = deg_inv[row] * deg_inv[col]

        adj = SparseTensor(
            row=torch.from_numpy(row).long(),
            col=torch.from_numpy(col).long(),
            value=torch.from_numpy(vals),
            sparse_sizes=(N, N),
        )
        unique_u = np.unique(uarr)
        local_u_map = {int(g): l for l, g in enumerate(unique_u.tolist())}
        global_u = torch.from_numpy(unique_u).long()
        return adj, global_u, local_u_map

    def prep_tensors(
        self, train_df: pd.DataFrame, local_u_map: dict
    ) -> tuple[torch.Tensor, torch.Tensor, int, pd.DataFrame]:
        """Filter train_df to known users and return (pos_u_local, pos_i, N, filtered_df)."""
        mask = train_df["u_idx"].apply(lambda u: int(u) in local_u_map)
        tdf = train_df[mask].reset_index(drop=True)
        pu_local = torch.tensor(
            [local_u_map[int(v)] for v in tdf["u_idx"].values], dtype=torch.long
        )
        pi = torch.tensor(tdf["i_idx"].values, dtype=torch.long)
        return pu_local, pi, len(pu_local), tdf
