"""Graph subset scorer used by text/spelling graph checkpoints.

Parameter names and forward computation match the graph experiment notebook.
"""

import torch
from torch import nn

from .models import MultiheadAttentionBlock, SetDecoder, SetTransformer


class GraphAttentionBlock(MultiheadAttentionBlock):
    def __init__(self, width, heads, dropout, edge_dim, edge_bias_scale):
        super().__init__(width, width, attn_heads=heads, dropout=dropout)
        self.edge_bias_scale = float(edge_bias_scale)
        if self.edge_bias_scale < 0:
            raise ValueError("edge_bias_scale must be nonnegative")
        self.edge_projection = nn.Linear(edge_dim, heads, bias=False)
        # Start from ordinary attention and learn which relationships to use.
        nn.init.zeros_(self.edge_projection.weight)

    def attention_bias(self, edges):
        bias = self.edge_bias_scale * torch.tanh(self.edge_projection(edges))
        # (B, N, N, H) -> (B * H, N, N), as required by MultiheadAttention.
        diagonal = torch.eye(edges.shape[1], device=edges.device, dtype=torch.bool)
        bias = bias.masked_fill(diagonal[None, :, :, None], 0)
        return bias.permute(0, 3, 1, 2).reshape(-1, edges.shape[1], edges.shape[2])

    def forward(self, x, edges):
        # x: (B, N, D); edges: (B, N, N, E)
        bias = None if self.edge_bias_scale == 0 else self.attention_bias(edges).to(x.dtype)
        attended = self.attn(x, x, x, attn_mask=bias, need_weights=False)[0]
        x = self.h_norm(x + attended)
        return self.mab_norm(self.residual_proj(x) + self.ff(x))


class GraphEncoder(nn.Module):
    def __init__(self, width, depth, heads, dropout, edge_dim, edge_bias_scale):
        super().__init__()
        self.mabs = nn.ModuleList([
            GraphAttentionBlock(width, heads, dropout, edge_dim, edge_bias_scale)
            for _ in range(depth)
        ])

    def forward(self, x, edges):
        for block in self.mabs:
            x = block(x, edges)
        return x


class GraphCrossAttentionBlock(MultiheadAttentionBlock):
    def __init__(self, width, heads, dropout, edge_dim, edge_bias_scale):
        super().__init__(width, width, attn_heads=heads, dropout=dropout)
        self.edge_bias_scale = float(edge_bias_scale)
        self.edge_projection = nn.Linear(edge_dim, heads, bias=False)
        nn.init.zeros_(self.edge_projection.weight)

    def forward(self, seeds, words, word_edges):
        # seeds: (B * C, 1, D); words: (B * C, 4, D); word_edges: (B * C, 4, E)
        bias = None
        if self.edge_bias_scale != 0:
            bias = self.edge_bias_scale * torch.tanh(self.edge_projection(word_edges))
            bias = bias.permute(0, 2, 1).reshape(-1, 1, words.shape[1]).to(seeds.dtype)
        attended = self.attn(seeds, words, words, attn_mask=bias, need_weights=False)[0]
        x = self.h_norm(seeds + attended)
        return self.mab_norm(self.residual_proj(x) + self.ff(x))


class GraphSetDecoder(SetDecoder):
    def __init__(self, width, heads, dropout, edge_dim, edge_bias_scale):
        super().__init__(width, width, width, num_seeds=1, mab_attn_heads=heads, dropout=dropout)
        original = self.mab
        self.mab = GraphCrossAttentionBlock(width, heads, dropout, edge_dim, edge_bias_scale)
        self.mab.load_state_dict(original.state_dict(), strict=False)

    def forward(self, x, edges):
        # x: (B * C, 4, D); edges: (B * C, 4, 4, E)
        words = self.ff1(x)
        seeds = self.seed_vectors.unsqueeze(0).expand(x.shape[0], -1, -1)
        # Diagonal edges are zero, so divide by the three other words.
        word_edges = edges.sum(dim=1) / (edges.shape[1] - 1)  # (B * C, 4, E)
        return self.mab(seeds, words, word_edges)


class GraphSetTransformer(SetTransformer):
    """Existing LTR scorer with edge-biased global and candidate attention."""
    def __init__(self, *, edge_dim=3, edge_bias_scale=1.0, **model_config):
        super().__init__(**model_config)
        self.edge_dim = edge_dim
        original = self.full_encoder
        first = original.mabs[0]
        self.full_encoder = GraphEncoder(
            first.attn.embed_dim, len(original.mabs), first.attn.num_heads,
            first.attn.dropout, edge_dim, edge_bias_scale,
        )
        # Preserve the original initialization; scale=0 is the same model.
        self.full_encoder.load_state_dict(original.state_dict(), strict=False)
        original = self.subset_encoder
        first = original.mabs[0]
        self.subset_encoder = GraphEncoder(
            first.attn.embed_dim, len(original.mabs), first.attn.num_heads,
            first.attn.dropout, edge_dim, edge_bias_scale,
        )
        self.subset_encoder.load_state_dict(original.state_dict(), strict=False)
        original = self.subset_decoder
        self.subset_decoder = GraphSetDecoder(
            original.mab.attn.embed_dim, original.mab.attn.num_heads,
            original.mab.attn.dropout, edge_dim, edge_bias_scale,
        )
        self.subset_decoder.load_state_dict(original.state_dict(), strict=False)

    def forward(self, x, group_idx=None, edge_features=None):
        B, N, _ = x.shape
        if edge_features is None or edge_features.shape != (B, N, N, self.edge_dim):
            raise ValueError("Expected edge_features with shape (B, N, N, edge_dim)")
        x = self.proj(self.input_norm(x))  # (B, N, D)
        x = self.full_encoder(x, edge_features)  # (B, N, D)
        idx = self.grouper.idx if group_idx is None else group_idx.to(x.device)
        x = self.grouper(x, group_idx=idx)  # (B, C, 4, D)
        B, C, group_size, D = x.shape
        # Select the same four indices on both edge axes for every candidate.
        candidate_edges = edge_features[:, idx[:, :, None], idx[:, None, :], :]
        candidate_edges = candidate_edges.reshape(B * C, group_size, group_size, self.edge_dim)
        x = x.reshape(B * C, group_size, D)
        x = self.subset_encoder(x, candidate_edges)  # (B * C, 4, D)
        x = self.subset_decoder(x, candidate_edges).squeeze(1)  # (B * C, D)
        return self.scorer(x.reshape(B, C, D))  # (B, C, 1)
