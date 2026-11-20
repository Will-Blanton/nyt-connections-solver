"""Learned set-to-set transformation edges and optional node fusion for GINE."""
import torch
from torch import nn
import torch.nn.functional as F
from .graph_models import StandaloneCandidateGINE


class TransformationGINE(StandaloneCandidateGINE):
    def __init__(self, *, text_dim, base_edge_dim, max_transformations=12,
                 transformation_width=64, learned_edge_dim=16, node_mode='text',
                 learned_edges=True, original_only=False, pair_hidden=64, pair_layers=2, **config):
        if pair_hidden < 1 or pair_layers not in (1, 2):
            raise ValueError('Pair matcher requires positive width and one or two linear layers')
        if node_mode not in {'text','fused','constant'}:
            raise ValueError('Unknown transformation node mode')
        if config['in_dim'] != text_dim + max_transformations * (text_dim + 7):
            raise ValueError('Transformation payload dimensions do not match')
        if config['edge_dim'] != base_edge_dim:
            raise ValueError('Saved edge_dim describes supplied base edge features')
        super().__init__(**{**config, 'in_dim': text_dim})
        self.text_dim, self.base_edge_dim = text_dim, base_edge_dim
        self.max_transformations, self.node_mode = max_transformations, node_mode
        self.learned_edges, self.learned_edge_dim = learned_edges, learned_edge_dim
        self.original_only = bool(original_only)
        # Extend only GINE's edge inputs. Preserve the original initialized
        # columns, and start the additional columns at zero in all arms.
        for i, old in enumerate(self.edge_encoder.edge_projections):
            new = nn.Linear(base_edge_dim+learned_edge_dim, old.out_features, bias=False)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :base_edge_dim].copy_(old.weight)
            self.edge_encoder.edge_projections[i] = new
        self.transform_projection = nn.Linear(text_dim, transformation_width)
        self.type_embedding = nn.Embedding(4, 8)
        self.transform_encoder = nn.Sequential(nn.Linear(transformation_width+8+5, transformation_width), nn.ReLU())
        # Symmetric comparisons, plus explicit structural compatibility.
        if pair_layers == 1:
            self.pair_encoder = nn.Sequential(nn.Linear(2*transformation_width+8, learned_edge_dim), nn.ReLU())
        else:
            self.pair_encoder = nn.Sequential(nn.Linear(2*transformation_width+8, pair_hidden), nn.ReLU(),
                                              nn.Linear(pair_hidden, learned_edge_dim), nn.ReLU())
        output_dim = self.edge_encoder.node_projection.in_features
        self.node_gate = nn.Linear(transformation_width, 1)
        self.node_fusion = nn.Sequential(nn.Linear(output_dim+2*transformation_width, output_dim), nn.ReLU())
        self.shared_node = nn.Parameter(torch.zeros(output_dim))
        if node_mode == 'constant':
            self.input_norm.requires_grad_(False)
            self.proj.requires_grad_(False)
        if node_mode != 'fused':
            self.node_gate.requires_grad_(False)
            self.node_fusion.requires_grad_(False)
        if node_mode != 'constant':
            self.shared_node.requires_grad_(False)
        if not learned_edges:
            self.transform_projection.requires_grad_(False)
            self.type_embedding.requires_grad_(False)
            self.transform_encoder.requires_grad_(False)
            self.pair_encoder.requires_grad_(False)

    def encode_transformations(self, x):
        slots = x[..., self.text_dim:].reshape(*x.shape[:-1], self.max_transformations, self.text_dim+7)
        valid = slots[..., -1] > .5
        positions = slots[..., self.text_dim:-1]
        types = positions[..., 0].long()
        if self.original_only:
            valid = valid & types.eq(0)
        projected = self.transform_projection(F.normalize(slots[..., :self.text_dim], dim=-1))
        z = self.transform_encoder(torch.cat([projected, self.type_embedding(types), positions[..., 1:]], -1))
        return z, valid, positions

    def pair_edges(self, z, valid, positions):
        # Compute N*(N-1)/2 pairs once per state, not once per candidate.
        B, N, K, D = z.shape
        a, b = torch.triu_indices(N, N, 1, device=z.device)
        left, right = z[:, a, :, None], z[:, b, None, :]
        p, q = positions[:, a, :, None], positions[:, b, None, :]
        delta = (p[..., 1:] - q[..., 1:]).abs()
        compatibility = torch.cat([delta, (p[..., :1] == q[..., :1]).float(),
            (p[..., 1:2] == q[..., 1:2]).float(), (p[..., 2:3] == q[..., 2:3]).float()], -1)
        pair = self.pair_encoder(torch.cat([(left-right).abs(), left*right, compatibility], -1))
        mask = valid[:, a, :, None] & valid[:, b, None, :]
        pooled = pair.masked_fill(~mask[..., None], -torch.inf).amax(dim=(-3,-2))
        pooled = torch.where(mask.any(dim=(-2,-1))[..., None], pooled, torch.zeros_like(pooled))
        result = z.new_zeros(B, N, N, self.learned_edge_dim)
        result[:, a, b] = pooled
        result[:, b, a] = pooled
        return result

    def node_vectors(self, x, z=None, valid=None):
        if self.node_mode == 'constant':
            return self.shared_node.expand(*x.shape[:-1], -1)
        text = super().encode_input_words(x[..., :self.text_dim])
        if self.node_mode == 'text':
            return text
        gate = self.node_gate(z).sigmoid() * valid[..., None]
        weighted = (z*gate).sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)
        maximum = z.masked_fill(~valid[..., None], -torch.inf).amax(-2)
        maximum = torch.where(valid.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
        return self.node_fusion(torch.cat([text, weighted, maximum], -1))

    def forward(self, x, group_idx=None, edge_features=None):
        B, N, _ = x.shape
        if edge_features is None or edge_features.shape != (B,N,N,self.base_edge_dim):
            raise ValueError('Expected supplied base edge features')
        z = valid = positions = None
        if self.learned_edges or self.node_mode == 'fused':
            z, valid, positions = self.encode_transformations(x)
        extra = self.pair_edges(z, valid, positions) if self.learned_edges else x.new_zeros(B,N,N,self.learned_edge_dim)
        full_edges = torch.cat([edge_features, extra], -1)
        words = self.node_vectors(x,z,valid)
        idx = self.grouper.idx if group_idx is None else group_idx.to(x.device)
        candidates = self.grouper(words, group_idx=idx)
        _, C, size, D = candidates.shape
        edges = full_edges[:, idx[:,:,None],idx[:,None,:],:]
        pooled = self.edge_encoder(candidates.reshape(B*C,size,D), edges.reshape(B*C,size,size,-1))
        return self.group_readout(pooled).reshape(B,C,1)

    def diagnostic_representations(self, nodes, edges):
        with torch.inference_mode():
            x = nodes[None]
            z, valid, _ = self.encode_transformations(x)
            words = self.node_vectors(x, z, valid)
            mean = (z * valid[..., None]).sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)
        return {'GINE input words': words[0], 'Transformation mean': mean[0]}
