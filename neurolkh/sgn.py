import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

class SparseGraphConvLayer(nn.Module):

    def __init__(self, D: int = 128):
        super().__init__()
        self.D = D

        self.W_a = nn.Linear(D, D, bias=False)
        self.W_s = nn.Linear(D, D)
        self.W_n = nn.Linear(D, D, bias=False)
        self.W_f = nn.Linear(D, D)
        self.W_t = nn.Linear(D, D)
        self.W_o = nn.Linear(D, D)
        self.W_r = nn.Linear(D, D)
        self.p = nn.Parameter(torch.zeros(D))

        self.node_bn = nn.BatchNorm1d(D, track_running_stats=False)
        self.edge_bn = nn.BatchNorm1d(D, track_running_stats=False)

    def forward(self, v, e, edge_index, reverse_exists, reverse_index):
        src, tgt = edge_index[0], edge_index[1]

        N = v.shape[0]
        E = e.shape[0]

        attn_logits = self.W_a(e)

        max_per_src = torch.full((N, self.D), -float('inf'), device=v.device)
        max_per_src.index_reduce_(0, src, attn_logits, 'amax', include_self=False)
        attn_logits = attn_logits - max_per_src[src]

        attn_exp = torch.exp(attn_logits)

        sum_exp = torch.zeros(N, self.D, device=v.device)
        sum_exp.index_add_(0, src, attn_exp)
        sum_exp = torch.clamp(sum_exp, min=1e-10)
        attn = attn_exp / (sum_exp[src] + 1e-10)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        neighbor_msg = self.W_n(v[tgt])
        weighted_msg = attn * neighbor_msg
        aggregated = torch.zeros(N, self.D, device=v.device)
        aggregated.index_add_(0, src, weighted_msg)
        v_update = self.W_s(v) + aggregated
        v_new = v + F.relu(self.node_bn(v_update))
        v_new = torch.nan_to_num(v_new, nan=0.0, posinf=0.0, neginf=0.0)

        r = torch.zeros(E, self.D, device=e.device)
        if reverse_exists.any():
            r[reverse_exists] = self.W_r(e[reverse_index[reverse_exists]])
        if (~reverse_exists).any():
            r[~reverse_exists] = self.W_r(self.p).unsqueeze(0)

        e_update = self.W_f(v[src]) + self.W_t(v[tgt]) + self.W_o(e) + r
        e_new = e + F.relu(self.edge_bn(e_update))
        e_new = torch.nan_to_num(e_new, nan=0.0, posinf=0.0, neginf=0.0)

        return v_new, e_new


class SparseGraphNetwork(nn.Module):
    def __init__(self, D: int = 128, L: int = 30, gamma: int = 20, C: float = 10.0):
        super().__init__()
        self.D = D
        self.L = L
        self.gamma = gamma
        self.C = C

        self.node_encoder = nn.Linear(2, D)
        self.edge_encoder = nn.Linear(1, D)

        self.conv_layers = nn.ModuleList([
            SparseGraphConvLayer(D) for _ in range(L)
        ])

        self.edge_decoder = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU()
        )
        self.W_beta = nn.Linear(D, 1, bias=False)

        self.node_decoder = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU()
        )
        self.W_pi = nn.Linear(D, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def build_sparse_graph(self, coords: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        N = coords.shape[0]
        device = coords.device

        coords_noisy = coords + torch.randn_like(coords) * 1e-6

        dist = torch.cdist(coords_noisy, coords_noisy)
        dist.fill_diagonal_(float('inf'))

        edge_pairs = []
        edge_dist = []

        for i in range(N):
            k = min(self.gamma, N - 1)
            _, indices = torch.topk(dist[i], k, largest=False)
            for j in indices:
                edge_pairs.append((i, j.item()))
                edge_dist.append(dist[i, j].item())

        edge_index = torch.tensor(edge_pairs, device=device, dtype=torch.long).t()
        edge_dist = torch.tensor(edge_dist, device=device)

        edge_to_idx = {pair: idx for idx, pair in enumerate(edge_pairs)}
        reverse_index = torch.full((len(edge_pairs),), -1, dtype=torch.long, device=device)

        for idx, (i, j) in enumerate(edge_pairs):
            rev_idx = edge_to_idx.get((j, i), -1)
            reverse_index[idx] = rev_idx

        reverse_exists = reverse_index >= 0

        return edge_index, edge_dist, reverse_exists, reverse_index

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        N = coords.shape[0]
        device = coords.device

        edge_index, edge_dist, reverse_exists, reverse_index = self.build_sparse_graph(coords)
        src, tgt = edge_index[0], edge_index[1]
        E = edge_index.shape[1]

        v = self.node_encoder(coords)
        e = self.edge_encoder(edge_dist.unsqueeze(-1))

        v = torch.nan_to_num(v, nan=0.0)
        e = torch.nan_to_num(e, nan=0.0)

        for layer in self.conv_layers:
            v, e = layer(v, e, edge_index, reverse_exists, reverse_index)

        e_f = self.edge_decoder(e)
        e_f = torch.nan_to_num(e_f, nan=0.0)

        beta_logits = self.W_beta(e_f).squeeze(-1)

        beta = torch.zeros(E, device=device)
        for i in range(N):
            mask = src == i
            if mask.any():
                logits = beta_logits[mask]
                logits = logits - logits.max()
                exp_logits = torch.exp(torch.clamp(logits, min=-20, max=20))
                beta[mask] = exp_logits / (exp_logits.sum() + 1e-10)

        beta = torch.nan_to_num(beta, nan=1.0/E)

        v_f = self.node_decoder(v)
        v_f = torch.nan_to_num(v_f, nan=0.0)
        pi = self.C * torch.tanh(self.W_pi(v_f).squeeze(-1))
        pi = torch.nan_to_num(pi, nan=0.0)

        return beta, pi, edge_index, edge_dist

    def get_candidates(self, beta: torch.Tensor, edge_index: torch.Tensor, k: int = 5) -> List[Tuple[int, int, int]]:
        N = edge_index.max().item() + 1
        src, tgt = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        beta_np = beta.detach().cpu().numpy()

        candidates = []
        for i in range(N):
            mask = src == i
            edges = [(tgt[j], beta_np[j]) for j in np.where(mask)[0]]
            edges.sort(key=lambda x: x[1], reverse=True)
            for priority, (j, _) in enumerate(edges[:k]):
                candidates.append((i, j, priority))

        return candidates

    def transform_distances(self, coords: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(coords, coords)
        transformed = dist + pi.unsqueeze(0) + pi.unsqueeze(1)

        return torch.nan_to_num(transformed, nan=0.0, posinf=1e6, neginf=0.0)