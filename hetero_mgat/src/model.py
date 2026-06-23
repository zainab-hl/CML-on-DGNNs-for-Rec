"""
model.py
========
Heterogeneous, multi-modal GAT model (user/item/brand, 4 typed edges).

Design note on `encode()`
--------------------------
The original script closed over module-level globals (T_text_gpu,
T_image_gpu, ...) cached once at startup. That works in a notebook but
creates a circular dependency in a package: model.py would need to
import data.py to get the tensors, while data.py has no reason to know
about the model.

Instead, `HeteroMGAT.encode(...)` takes an explicit `ItemFeatures`
bundle (and the graph / user tensors) as arguments. `DataModule` owns
and caches those tensors; callers (trainer, evaluator) pass them
through. This keeps model.py a pure "given tensors, produce
embeddings" module with no knowledge of where the data came from.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv

from .config import Config


@dataclass
class ItemFeatures:
    """GPU-resident, pre-cached item feature tensors (built once by
    DataModule and passed into every encode() call)."""
    text: torch.Tensor   # [N_items, CLIP_DIM]
    image: torch.Tensor  # [N_items, CLIP_DIM]
    sub: torch.Tensor    # [N_items, TOP_K_SUBS]
    meta: torch.Tensor   # [N_items, META_DIM]
    main_idx: torch.Tensor  # [N_items] long

    @property
    def num_items(self) -> int:
        return self.text.shape[0]


class HashEmbedding(nn.Module):
    def __init__(self, num_buckets: int, dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.table = nn.Embedding(num_buckets, dim)
        nn.init.xavier_uniform_(self.table.weight)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        return self.table(user_idx % self.num_buckets)


class HistoryEncoder(nn.Module):
    def __init__(self, hidden: int, clip_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(clip_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, hist_tensor: torch.Tensor) -> torch.Tensor:
        return self.norm(self.drop(F.relu(self.proj(hist_tensor))))


def reset_gate_weights(model: 'HeteroMGAT') -> None:
    """
    GATE-1 / GATE-3: zero out the final gate linear layer so that
    Softmax([0, 0, 0, 0]) = [0.25, 0.25, 0.25, 0.25] at the start of
    training (or immediately after a growth event copies stale weights).

    The gate MLP is:  Linear(4H, H) → ReLU → Linear(H, 4) → Softmax
    We zero the LAST Linear(H, 4) only — its weight and bias — which is
    sufficient to guarantee a uniform start without touching the first
    layer that actually learns a shared representation across modalities.
    """
    gate_seq = model.item_enc.gate
    final_linear = gate_seq[2]
    nn.init.zeros_(final_linear.weight)
    nn.init.zeros_(final_linear.bias)


class ModalityFusion(nn.Module):
    """
    4-modality gated fusion: text / image / category / meta → hidden.
    Gate weights [N, 4] are returned for the S2 (modal dominance) signal.

    GATE-2: meta projection uses a bottleneck (5 → META_BN_DIM → hidden)
    instead of a direct projection (5 → hidden), to reduce meta's
    expansion advantage and slow its convergence relative to the other
    modalities.

    GATE-1: gate final layer is zero-initialised so that the first
    forward pass produces uniform weights [0.25, 0.25, 0.25, 0.25].
    """
    def __init__(self, hidden: int, num_main: int, cat_emb: int,
                 sub_dim: int, meta_dim: int, clip_dim: int,
                 meta_bn_dim: int, dropout: float):
        super().__init__()
        self.main_emb = nn.Embedding(num_main, cat_emb)
        nn.init.xavier_uniform_(self.main_emb.weight)
        self.proj_text = nn.Linear(clip_dim, hidden)
        self.proj_image = nn.Linear(clip_dim, hidden)
        self.proj_cat = nn.Linear(cat_emb + sub_dim, hidden)

        # GATE-2: bottleneck replaces a single wide Linear(5, hidden)
        self.proj_meta_bn = nn.Linear(meta_dim, meta_bn_dim)
        self.proj_meta = nn.Linear(meta_bn_dim, hidden)

        self.gate = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4), nn.Softmax(dim=-1),
        )

        # GATE-1: zero-init the final gate linear so softmax starts uniform
        nn.init.zeros_(self.gate[2].weight)
        nn.init.zeros_(self.gate[2].bias)

        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, image, sub, meta, main_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        h_text = F.relu(self.proj_text(text))
        h_img = F.relu(self.proj_image(image))
        h_cat = F.relu(self.proj_cat(
            torch.cat([self.main_emb(main_idx), sub], dim=1)
        ))
        h_meta = F.relu(self.proj_meta(F.relu(self.proj_meta_bn(meta))))

        g = self.gate(torch.cat([h_text, h_img, h_cat, h_meta], dim=1))
        out = (g[:, 0:1] * h_text + g[:, 1:2] * h_img
               + g[:, 2:3] * h_cat + g[:, 3:4] * h_meta)
        return self.norm(self.dropout(out)), g  # [N,H], [N,4]


class NodeTypeAttention(nn.Module):
    """Learnable skip-connection gate per node type."""
    def __init__(self, node_types):
        super().__init__()
        self.gates = nn.ParameterDict(
            {t: nn.Parameter(torch.tensor(0.5)) for t in node_types}
        )

    def forward(self, x_before: Dict[str, torch.Tensor],
                x_after: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for t in x_after:
            g = torch.sigmoid(self.gates[t]) if t in self.gates else 1.0
            out[t] = g * x_after[t] + (1.0 - g) * x_before[t]
        return out


class HeteroMGAT(nn.Module):
    """
    Heterogeneous multi-modal GAT over (user, item, brand) with 4 typed
    edges. GAT depth is architecture-fixed at 2 conv layers; the model
    "grows" along two other axes instead: `hidden` width and attention
    `heads` (see growth.GrowthController).
    """
    def __init__(self, hidden: int, num_heads_l1: int, dropout: float,
                 num_brands: int, num_main: int, cat_emb: int, sub_dim: int,
                 meta_dim: int, clip_dim: int, hash_size: int, meta_bn_dim: int):
        super().__init__()
        self.hidden = hidden
        self.clip_dim = clip_dim
        self.user_hash = HashEmbedding(hash_size, hidden)
        self.hist_enc = HistoryEncoder(hidden, clip_dim, dropout)
        self.user_gate = nn.Parameter(torch.tensor(0.5))
        self.item_enc = ModalityFusion(
            hidden, num_main, cat_emb, sub_dim, meta_dim, clip_dim,
            meta_bn_dim, dropout,
        )
        self.brand_emb = nn.Embedding(num_brands, hidden)
        nn.init.xavier_uniform_(self.brand_emb.weight)

        self.conv1 = self._make_conv(hidden, num_heads_l1, dropout)
        self.conv2 = self._make_conv(hidden, 1, dropout)

        self.nta1 = NodeTypeAttention(['user', 'item', 'brand'])
        self.nta2 = NodeTypeAttention(['user', 'item', 'brand'])

        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _make_conv(hidden: int, heads: int, dropout: float) -> HeteroConv:
        return HeteroConv({
            ('user', 'purchased', 'item'): GATConv(
                hidden, hidden, heads=heads, concat=False,
                dropout=dropout, add_self_loops=False),
            ('item', 'bought_by', 'user'): GATConv(
                hidden, hidden, heads=heads, concat=False,
                dropout=dropout, add_self_loops=False),
            ('brand', 'brand_of', 'item'): GATConv(
                hidden, hidden, heads=heads, concat=False,
                dropout=dropout, add_self_loops=False),
            ('item', 'made_by', 'brand'): GATConv(
                hidden, hidden, heads=heads, concat=False,
                dropout=dropout, add_self_loops=False),
        }, aggr='mean')

    def _user_init(self, global_u_idx: torch.Tensor,
                   hist_gpu: torch.Tensor) -> torch.Tensor:
        h_hash = self.user_hash(global_u_idx)
        h_hist = self.hist_enc(hist_gpu)
        g = torch.sigmoid(self.user_gate)
        return g * h_hash + (1.0 - g) * h_hist

    def _item_init_full(self, feats: ItemFeatures) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.item_enc(
            feats.text, feats.image, feats.sub, feats.meta, feats.main_idx,
        )

    def encode(self, ei_gpu: Dict, global_u_gpu: torch.Tensor,
               hist_gpu: torch.Tensor, feats: ItemFeatures):
        """
        Full heterogeneous graph encode.

        Args:
            ei_gpu: dict of edge_type -> edge_index tensor (on DEVICE)
            global_u_gpu: global user indices for this local user set (on DEVICE)
            hist_gpu: per-user history-CLIP-mean tensor (on DEVICE)
            feats: pre-cached GPU item feature tensors

        Returns:
            x_cpu (dict of node-type -> embedding, on CPU),
            gate_weights [N_items, 4] (CPU),
            layer0_items (GPU, detached),
            layer2_items (GPU, detached)
        """
        item_init, gates = self._item_init_full(feats)
        x0 = {
            'user': self._user_init(global_u_gpu, hist_gpu),
            'item': item_init,
            'brand': self.brand_emb.weight,
        }
        layer0_items = item_init.detach()

        x1 = self.conv1(x0, ei_gpu)
        x1 = self.nta1(x0, x1)
        x1 = {k: F.elu(self.drop(v)) for k, v in x1.items()}

        x2 = self.conv2(x1, ei_gpu)
        x2 = self.nta2(x1, x2)
        x2 = {k: F.elu(v) for k, v in x2.items()}

        layer2_items = x2['item'].detach()
        x_cpu = {k: v.cpu() for k, v in x2.items()}
        return x_cpu, gates.detach().cpu(), layer0_items, layer2_items

    @torch.no_grad()
    def encode_eval(self, ei_gpu, global_u_gpu, hist_gpu, feats: ItemFeatures):
        self.eval()
        return self.encode(ei_gpu, global_u_gpu, hist_gpu, feats)

    def decode(self, x_dict, u_local_idx: torch.Tensor,
               i_idx: torch.Tensor, device: str) -> torch.Tensor:
        u = x_dict['user'][u_local_idx].to(device)
        i = x_dict['item'][i_idx].to(device)
        return self.predictor(torch.cat([u, i], dim=1)).squeeze(-1)

    @property
    def num_heads(self) -> int:
        return list(self.conv1.convs.values())[0].heads

    def gnn_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        skip = {'user_hash', 'brand_emb', 'item_enc.main_emb', 'hist_enc'}
        for name, p in self.named_parameters():
            if not any(name.startswith(s) for s in skip):
                yield name, p

    def graph_conv_parameters(self) -> Iterator[Tuple[str, nn.Parameter]]:
        prefixes = ('conv1', 'conv2', 'nta1', 'nta2')
        for name, p in self.named_parameters():
            if any(name.startswith(pf) for pf in prefixes):
                yield name, p

    def new_parameters(self, old_hidden: int) -> Iterator[nn.Parameter]:
        """Parameters that changed shape after hidden growth (for warm-up)."""
        for name, p in self.named_parameters():
            if 'item_enc' in name and ('proj' in name or 'norm' in name
                                       or 'gate' in name):
                yield p
            elif 'hist_enc' in name and p.shape[0] != old_hidden:
                yield p


def build_dynamic(cfg: Config, num_brands: int, num_main: int,
                  hidden: int = None, heads: int = None) -> HeteroMGAT:
    return HeteroMGAT(
        hidden=hidden or cfg.HIDDEN_INIT,
        num_heads_l1=heads or cfg.HEADS_INIT,
        dropout=cfg.DROPOUT,
        num_brands=num_brands, num_main=num_main,
        cat_emb=cfg.CAT_EMB_DIM, sub_dim=cfg.TOP_K_SUBS, meta_dim=cfg.META_DIM,
        clip_dim=cfg.CLIP_DIM, hash_size=cfg.HASH_SIZE,
        meta_bn_dim=cfg.META_BN_DIM,
    ).to(cfg.device)


def build_static(cfg: Config, num_brands: int, num_main: int) -> HeteroMGAT:
    return HeteroMGAT(
        hidden=cfg.HIDDEN_STATIC, num_heads_l1=cfg.HEADS_STATIC,
        dropout=cfg.DROPOUT,
        num_brands=num_brands, num_main=num_main,
        cat_emb=cfg.CAT_EMB_DIM, sub_dim=cfg.TOP_K_SUBS, meta_dim=cfg.META_DIM,
        clip_dim=cfg.CLIP_DIM, hash_size=cfg.HASH_SIZE,
        meta_bn_dim=cfg.META_BN_DIM,
    ).to(cfg.device)
