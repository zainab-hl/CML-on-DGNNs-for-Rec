
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from .config import Config
from .model import ItemFeatures

# A heterogeneous edge_index dict, plus the bookkeeping needed to map
# between local (period-specific) and global user indices.
EdgeIndexBundle = Tuple[Dict[Tuple[str, str, str], torch.Tensor],
                         torch.Tensor, int, Dict[int, int], int]
# (edge_dict, unique_global_user_ids, n_local_users, local_u_map, n_edges)


def parse_price(p) -> float:
    try:
        v = float(str(p).replace('$', '').replace(',', '').strip())
        return v if v > 0 else np.nan
    except Exception:
        return np.nan


def safe_scale(arr) -> np.ndarray:
    arr = np.array(arr, dtype=np.float32)
    return np.nan_to_num(
        (arr - np.nanmean(arr)) / (np.nanstd(arr) + 1e-8), nan=0.0
    )


class DataModule:
    """
    Loads raw data once, builds all static feature tensors, and exposes
    helpers (`make_edge_index`, `build_hist_tensor`, `chrono_split`)
    used by the trainer/evaluator/experiment layers.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load_raw()
        self._build_index_maps()
        self._build_item_features()
        self._cache_features_on_gpu()
        self._build_history_lookup()
        self._temporal_split()
        self._build_brand_item_edges()

    # ── 1. Raw load ─────────────────────────────────────────────────────
    def _load_raw(self) -> None:
        cfg = self.cfg
        print('\n── Loading data ──')
        self.items_df = pd.read_parquet(cfg.DATA / 'nodes_items.parquet')
        self.users_df = pd.read_parquet(cfg.DATA / 'nodes_users.parquet')
        self.brands_df = pd.read_parquet(cfg.DATA / 'nodes_brands.parquet')
        self.edges_df = pd.read_parquet(cfg.DATA / 'edges_purchased.parquet')
        self.bi_df = pd.read_parquet(cfg.DATA / 'edges_brand_item.parquet')
        self.emb_df = pd.read_parquet(cfg.root / 'embeddings_pca128.parquet')

        if 'item_id' in self.items_df.columns:
            self.items_df = self.items_df.set_index('item_id')
        if 'item_id' in self.emb_df.columns:
            self.emb_df = self.emb_df.set_index('item_id')

        print(f'  items={len(self.items_df):,}  users={len(self.users_df):,}  '
              f'brands={len(self.brands_df):,}')
        print(f'  purchased={len(self.edges_df):,}  brand-item={len(self.bi_df):,}')

    # ── 2. Index maps ────────────────────────────────────────────────────
    def _build_index_maps(self) -> None:
        self.item2idx = {it: i for i, it in enumerate(self.items_df.index.tolist())}
        self.brand2idx = {b: i for i, b in enumerate(self.brands_df['brand_id'].tolist())}
        self.user2idx = {u: i for i, u in enumerate(self.users_df['user_id'].tolist())}

        self.NUM_ITEMS = len(self.item2idx)
        self.NUM_BRANDS = len(self.brand2idx)
        self.NUM_USERS = len(self.user2idx)
        self.item_ids_ordered = list(self.item2idx.keys())

        print(f'  NUM_ITEMS={self.NUM_ITEMS:,}  NUM_BRANDS={self.NUM_BRANDS:,}  '
              f'NUM_USERS={self.NUM_USERS:,}')

    # ── 3. Item features (PCA-128 text/image, category, meta) ───────────
    def _build_item_features(self) -> None:
        cfg = self.cfg
        print('\n── Building item features (PCA-128) ──')
        N = self.NUM_ITEMS

        text_mat = np.zeros((N, cfg.CLIP_DIM), dtype=np.float32)
        image_mat = np.zeros((N, cfg.CLIP_DIM), dtype=np.float32)
        for iid, idx in self.item2idx.items():
            if iid in self.emb_df.index:
                row = self.emb_df.loc[iid]
                te = row['text_embedding']
                ie = row['image_embedding']
                if hasattr(te, '__len__'):
                    text_mat[idx] = np.array(te, dtype=np.float32)[:cfg.CLIP_DIM]
                    image_mat[idx] = np.array(ie, dtype=np.float32)[:cfg.CLIP_DIM]

        main_cats = self.items_df['main_category'].fillna('unknown').tolist()
        main_le = LabelEncoder().fit(main_cats)
        self.NUM_MAIN = len(main_le.classes_)

        all_subs: List[str] = []
        for s in self.items_df['sub_categories'].fillna(''):
            all_subs += [x.strip() for x in str(s).split('|') if x.strip()]
        top_subs = [s for s, _ in Counter(all_subs).most_common(cfg.TOP_K_SUBS)]
        sub2i = {s: i for i, s in enumerate(top_subs)}

        main_idx_arr = np.zeros(N, dtype=np.int64)
        sub_mat = np.zeros((N, cfg.TOP_K_SUBS), dtype=np.float32)
        for iid, idx in self.item2idx.items():
            if iid not in self.items_df.index:
                continue
            mc = self.items_df.at[iid, 'main_category']
            if isinstance(mc, str) and mc:
                main_idx_arr[idx] = int(main_le.transform([mc])[0])
            subs = self.items_df.at[iid, 'sub_categories']
            if isinstance(subs, str):
                for s in subs.split('|'):
                    if s.strip() in sub2i:
                        sub_mat[idx, sub2i[s.strip()]] = 1.0

        item_ids = self.item_ids_ordered
        prices = [parse_price(self.items_df.at[iid, 'price'])
                  if iid in self.items_df.index else np.nan for iid in item_ids]
        avgrat = [float(self.items_df.at[iid, 'average_rating'])
                  if iid in self.items_df.index else np.nan for iid in item_ids]
        ratnum = [float(self.items_df.at[iid, 'rating_number'])
                  if iid in self.items_df.index else np.nan for iid in item_ids]
        has_det = [1.0 if (iid in self.items_df.index
                           and isinstance(self.items_df.at[iid, 'details_json'], str)
                           and self.items_df.at[iid, 'details_json'])
                   else 0.0 for iid in item_ids]
        price_m = [1.0 if np.isnan(p) else 0.0 for p in prices]
        meta_mat = np.stack([
            safe_scale(prices), safe_scale(avgrat), safe_scale(ratnum),
            np.array(has_det), np.array(price_m)
        ], axis=1).astype(np.float32)

        self.T_text = torch.tensor(text_mat, dtype=torch.float32)
        self.T_image = torch.tensor(image_mat, dtype=torch.float32)
        self.T_sub = torch.tensor(sub_mat, dtype=torch.float32)
        self.T_meta = torch.tensor(meta_mat, dtype=torch.float32)
        self.T_main = torch.tensor(main_idx_arr, dtype=torch.long)

        print(f'  T_text={self.T_text.shape}  T_image={self.T_image.shape}')
        print(f'  T_sub={self.T_sub.shape}  T_meta={self.T_meta.shape}  '
              f'NUM_MAIN={self.NUM_MAIN}')

    # ── 4. Cache feature tensors on GPU (once) ───────────────────────────
    def _cache_features_on_gpu(self) -> None:
        cfg = self.cfg
        print('\n── Caching feature tensors on GPU ──')
        self.item_features = ItemFeatures(
            text=self.T_text.to(cfg.device),
            image=self.T_image.to(cfg.device),
            sub=self.T_sub.to(cfg.device),
            meta=self.T_meta.to(cfg.device),
            main_idx=self.T_main.to(cfg.device),
        )
        nbytes = self.item_features.text.nbytes + self.item_features.image.nbytes
        print(f'  Done. (text+image = {nbytes / 1e6:.1f} MB on GPU)')

    # ── 5. History lookup ─────────────────────────────────────────────────
    def _build_history_lookup(self) -> None:
        print('\n── Building history lookup ──')
        edges_df = self.edges_df
        edges_df['timestamp'] = pd.to_numeric(edges_df['timestamp'], errors='coerce')
        edges_df = edges_df.dropna(subset=['timestamp']).copy()
        edges_df['u_idx'] = edges_df['user_id'].map(self.user2idx)
        edges_df['i_idx'] = edges_df['item_id'].map(self.item2idx)
        edges_df = edges_df.dropna(subset=['u_idx', 'i_idx']).copy()
        edges_df = edges_df.astype({'u_idx': int, 'i_idx': int})
        if 'datetime' not in edges_df.columns:
            edges_df['datetime'] = pd.to_datetime(
                edges_df['timestamp'] // 1000, unit='s', utc=True
            )
        self.edges_df = edges_df.sort_values('timestamp').reset_index(drop=True)

        user_hist_str: Dict[int, str] = {}
        for row in self.edges_df[['u_idx', 'history']].itertuples(index=False):
            u = int(row.u_idx)
            h = str(row.history).strip() if pd.notna(row.history) else ''
            if h and h != 'nan':
                user_hist_str[u] = h

        lookup: Dict[int, np.ndarray] = {}
        for u, h_str in user_hist_str.items():
            idxs = [self.item2idx[iid] for iid in h_str.split() if iid in self.item2idx]
            if not idxs:
                continue
            idxs = idxs[-50:]
            t_idx = torch.tensor(idxs, dtype=torch.long)
            clip_mean = ((self.T_text[t_idx] + self.T_image[t_idx]) * 0.5).mean(0).numpy()
            lookup[u] = clip_mean.astype(np.float32)

        self.HIST_LOOKUP = lookup
        print(f'  Users with history: {len(lookup):,}  '
              f'({len(lookup) / self.NUM_USERS:.1%})')

    def build_hist_tensor(self, n_u_local: int,
                          local_u_map: Dict[int, int]) -> torch.Tensor:
        out = np.zeros((n_u_local, self.cfg.CLIP_DIM), dtype=np.float32)
        for global_u, local_u in local_u_map.items():
            if global_u in self.HIST_LOOKUP:
                out[local_u] = self.HIST_LOOKUP[global_u]
        return torch.tensor(out, dtype=torch.float32)

    # ── 6. Temporal split ─────────────────────────────────────────────────
    def _temporal_split(self) -> None:
        cfg = self.cfg
        print('\n── Temporal split ──')
        pretrain_cut = pd.Timestamp(cfg.PRETRAIN_CUT, tz='UTC')

        self.pretrain_df = self.edges_df[self.edges_df['datetime'] < pretrain_cut].copy()
        cl_df = self.edges_df[self.edges_df['datetime'] >= pretrain_cut].copy()
        cl_df = cl_df.reset_index(drop=True)
        cl_df['cl_period'] = pd.qcut(
            cl_df['timestamp'], q=cfg.NUM_CL_PERIODS, labels=False
        )
        self.cl_df = cl_df

        print(f'  Pretrain : {len(self.pretrain_df):,} edges')
        for p in range(cfg.NUM_CL_PERIODS):
            sub = cl_df[cl_df['cl_period'] == p]
            print(f'  CL period {p}: {len(sub):,} edges  '
                  f'{sub["datetime"].min().date()} → {sub["datetime"].max().date()}  '
                  f'users={sub["u_idx"].nunique():,}')

    def chrono_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg
        n = len(df)
        n_tr = int(n * cfg.TRAIN_RATIO)
        n_va = int(n * cfg.VALID_RATIO)
        return (df.iloc[:n_tr].copy(),
                df.iloc[n_tr:n_tr + n_va].copy(),
                df.iloc[n_tr + n_va:].copy())

    # ── 7. Brand-item edges (static, shared across periods) ─────────────
    def _build_brand_item_edges(self) -> None:
        bi_df = self.bi_df
        bi_df = bi_df[bi_df['brand_id'].isin(self.brand2idx)
                     & bi_df['item_id'].isin(self.item2idx)].copy()
        b_src = torch.tensor(bi_df['brand_id'].map(self.brand2idx).values, dtype=torch.long)
        i_dst = torch.tensor(bi_df['item_id'].map(self.item2idx).values, dtype=torch.long)
        self.B_SRC = b_src
        self.I_DST = i_dst
        self.BRAND_ITEM = torch.stack([b_src, i_dst])
        self.ITEM_BRAND = torch.stack([i_dst, b_src])

    def make_edge_index(self, df: pd.DataFrame) -> EdgeIndexBundle:
        """Build heterogeneous edge index from a user-item interaction dataframe."""
        unique_u = np.unique(df['u_idx'].values)
        local_u_map = {int(g): l for l, g in enumerate(unique_u.tolist())}
        u_loc = torch.tensor(
            [local_u_map[int(v)] for v in df['u_idx'].values], dtype=torch.long
        )
        i_idx = torch.tensor(df['i_idx'].values, dtype=torch.long)
        n_u_local = int(len(unique_u))
        edge_dict = {
            ('user', 'purchased', 'item'): torch.stack([u_loc, i_idx]),
            ('item', 'bought_by', 'user'): torch.stack([i_idx, u_loc]),
            ('brand', 'brand_of', 'item'): self.BRAND_ITEM,
            ('item', 'made_by', 'brand'): self.ITEM_BRAND,
        }
        n_edges = u_loc.numel() + i_idx.numel() + self.B_SRC.numel() + self.I_DST.numel()
        return edge_dict, torch.tensor(unique_u, dtype=torch.long), n_u_local, local_u_map, n_edges

    def to_gpu(self, ei: Dict, global_u: torch.Tensor,
              hist: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """Convenience: move an edge-index bundle + hist tensor to DEVICE."""
        device = self.cfg.device
        ei_gpu = {k: v.to(device) for k, v in ei.items()}
        return ei_gpu, global_u.to(device), hist.to(device)
