from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class ModalFusion(nn.Module):
    """Gated fusion of text and image item features."""

    def __init__(self, hidden: int, text_dim: int, image_dim: int, dropout: float):
        super().__init__()
        self.proj_text = nn.Linear(text_dim, hidden)
        self.proj_image = nn.Linear(image_dim, hidden)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Softmax(dim=-1),
        )
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor, image: torch.Tensor):
        h_t = F.relu(self.proj_text(text))
        h_i = F.relu(self.proj_image(image))
        g = self.gate(torch.cat([h_t, h_i], dim=-1))
        out = g[:, 0:1] * h_t + g[:, 1:2] * h_i
        return self.norm(self.drop(out)), g


class LightGCN(nn.Module):
    """
    Multimodal LightGCN with gated fusion and growable capacity.

    Parameters
    ----------
    hidden : embedding dimension (can grow between CL periods)
    K      : number of propagation layers (can grow between CL periods)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        T_text: torch.Tensor,
        T_image: torch.Tensor,
        cfg: Config,
        hidden: int,
        K: int,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.T_text = T_text
        self.T_image = T_image
        self.cfg = cfg
        self.hidden = hidden
        self.K = K

        self.fusion = ModalFusion(hidden, cfg.text_dim_out, cfg.image_dim_out, cfg.dropout)
        self.user_emb = nn.Embedding(num_users, cfg.user_emb_dim)
        self.user_proj = nn.Linear(cfg.user_emb_dim, hidden, bias=False)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.user_proj.weight)
        self.user_gate = nn.Parameter(torch.tensor(0.5))
        self.drop = nn.Dropout(cfg.dropout)

    # ── Encoding ───────────────────────────────────────────────────────────

    def _item_init(self):
        return self.fusion(
            self.T_text.to(self.cfg.device),
            self.T_image.to(self.cfg.device),
        )

    def _user_init(self, item_emb: torch.Tensor, adj_cpu, global_u_gpu: torch.Tensor):
        n_u = global_u_gpu.shape[0]
        u_rows = adj_cpu.storage.row()
        i_cols = adj_cpu.storage.col()
        mask = (u_rows < self.num_users) & (i_cols >= self.num_users)
        u_sel = u_rows[mask]
        i_sel = i_cols[mask] - self.num_users

        g_u_cpu = global_u_gpu.cpu()
        pos = torch.searchsorted(g_u_cpu, u_sel)
        valid = (pos < n_u) & (g_u_cpu[pos.clamp(max=n_u - 1)] == u_sel)
        pos = pos[valid]
        i_sel = i_sel[valid]

        device = self.cfg.device
        u_pool = torch.zeros(n_u, self.hidden, device=device)
        counts = torch.zeros(n_u, 1, device=device)
        i_feats = item_emb[i_sel.to(device)]
        u_pool.scatter_add_(0, pos.to(device).unsqueeze(1).expand(-1, self.hidden), i_feats)
        counts.scatter_add_(0, pos.to(device).unsqueeze(1), torch.ones(len(pos), 1, device=device))
        u_pool = u_pool / counts.clamp(min=1.0)

        u_emb = self.user_proj(self.user_emb(global_u_gpu))
        g = torch.sigmoid(self.user_gate)
        return self.drop(g * u_emb + (1 - g) * u_pool)

    def encode(self, adj, global_u_gpu: torch.Tensor):
        item0, gates = self._item_init()
        user0 = self._user_init(item0, adj, global_u_gpu)

        N = self.num_users + self.num_items
        X = torch.zeros(N, self.hidden, device=self.cfg.device)
        X[global_u_gpu] = user0
        X[self.num_users:] = item0

        adj_gpu = adj.to(self.cfg.device)
        layer_embs = [X]
        for _ in range(self.K):
            X = adj_gpu @ layer_embs[-1]
            layer_embs.append(X)

        X_final = torch.stack(layer_embs, dim=0).mean(0)
        x_user = X_final[global_u_gpu]
        x_item = X_final[self.num_users:]
        layer0_items = layer_embs[0][self.num_users:].detach()
        layerK_items = layer_embs[-1][self.num_users:].detach()
        return x_user, x_item, gates, layer0_items, layerK_items

    @torch.no_grad()
    def encode_eval(self, adj, global_u_gpu: torch.Tensor):
        self.eval()
        return self.encode(adj, global_u_gpu)

    def decode(self, x_user, x_item, u_local, i_idx):
        return (x_user[u_local] * x_item[i_idx]).sum(dim=-1)

    # ── Parameter helpers ──────────────────────────────────────────────────

    def lgcn_parameters(self):
        """All parameters except the user embedding table (excluded from EWC)."""
        for n, p in self.named_parameters():
            if not n.startswith("user_emb."):
                yield n, p

    def new_parameters(self, old_hidden: int):
        """Parameters that are new or resized after a growth step."""
        for n, p in self.named_parameters():
            if n == "user_proj.weight" and p.shape[0] != old_hidden:
                yield p
            elif "fusion" in n and any(k in n for k in ("proj", "norm", "gate")):
                yield p
