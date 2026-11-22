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

    def encode_candidate_words(self, x, group_idx=None, edge_features=None, *, return_stages=False):
        B, N, _ = x.shape
        if edge_features is None or edge_features.shape != (B, N, N, self.edge_dim):
            raise ValueError("Expected edge_features with shape (B, N, N, edge_dim)")
        x = self.proj(self.input_norm(x))  # (B, N, D)
        projected = x if return_stages else None
        x = self.full_encoder(x, edge_features)  # (B, N, D)
        idx = self.grouper.idx if group_idx is None else group_idx.to(x.device)
        x = self.grouper(x, group_idx=idx)  # (B, C, 4, D)
        global_words = x if return_stages else None
        B, C, group_size, D = x.shape
        # Select the same four indices on both edge axes for every candidate.
        candidate_edges = edge_features[:, idx[:, :, None], idx[:, None, :], :]
        candidate_edges = candidate_edges.reshape(B * C, group_size, group_size, self.edge_dim)
        x = x.reshape(B * C, group_size, D)
        x = self.subset_encoder(x, candidate_edges)  # (B * C, 4, D)
        words = x.reshape(B, C, group_size, D)
        edges = candidate_edges.reshape(B, C, group_size, group_size, self.edge_dim)
        if return_stages:
            return words, edges, {"projected": self.grouper(projected, group_idx=idx),
                                  "global": global_words, "candidate": words}
        return words, edges

    def encode_candidates(self, x, group_idx=None, edge_features=None):
        words, edges = self.encode_candidate_words(x, group_idx, edge_features)
        B, C, group_size, D = words.shape
        x = words.reshape(B * C, group_size, D)
        candidate_edges = edges.reshape(B * C, group_size, group_size, self.edge_dim)
        x = self.subset_decoder(x, candidate_edges).squeeze(1)  # (B * C, D)
        return x.reshape(B, C, D)

    def forward(self, x, group_idx=None, edge_features=None):
        return self.scorer(self.encode_candidates(x, group_idx, edge_features))

    def diagnostic_representations(self, nodes, edges):
        """Optional per-word views used by diagnostics, independent of scoring."""
        projected = self.proj(self.input_norm(nodes[None]))
        return {"Learned input projection": projected[0],
                "Learned graph encoder": self.full_encoder(projected, edges[None])[0]}


class GatedEdgeResidual(GraphSetTransformer):
    """Frozen semantic scorer plus a candidate-local, gated edge correction.

    Six unordered pairs are encoded with a shared MLP and mean/max pooled.
    No edge message changes the pretrained semantic representations.
    """

    def __init__(self, *, edge_hidden=32, residual_edge_channels=None, **config):
        if config.get("edge_bias_scale", 0) != 0:
            raise ValueError("The residual experiment requires a text-only base")
        super().__init__(**config)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        mask = torch.ones(self.edge_dim)
        if residual_edge_channels is not None:
            if any(i < 0 or i >= self.edge_dim for i in residual_edge_channels):
                raise ValueError("Residual edge channels must be valid indices")
            mask.zero_()
            mask[list(residual_edge_channels)] = 1
        self.register_buffer("residual_edge_mask", mask, persistent=False)
        width = config.get("attn_width", 512)
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_dim, edge_hidden), nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden), nn.ReLU(),
        )
        self.edge_correction = nn.Linear(width + edge_hidden * 2, 1)
        self.edge_gate = nn.Linear(width + edge_hidden * 2, 1)
        nn.init.zeros_(self.edge_correction.weight)
        nn.init.zeros_(self.edge_correction.bias)

    def forward(self, x, group_idx=None, edge_features=None):
        with torch.no_grad():
            representation = self.encode_candidates(x, group_idx, edge_features)
            base_score = self.scorer(representation)
        idx = self.grouper.idx if group_idx is None else group_idx.to(x.device)
        pairs = torch.triu_indices(4, 4, offset=1, device=x.device)
        pair_edges = edge_features[:, idx[:, pairs[0]], idx[:, pairs[1]], :]
        encoded = self.edge_encoder(pair_edges * self.residual_edge_mask)
        pooled = torch.cat([encoded.mean(dim=2), encoded.amax(dim=2)], dim=-1)
        combined = torch.cat([representation, pooled], dim=-1)
        return base_score + self.edge_gate(combined).sigmoid() * self.edge_correction(combined)

    def train(self, mode=True):
        super().train(mode)
        for module in (self.input_norm, self.proj, self.full_encoder,
                       self.subset_encoder, self.subset_decoder, self.scorer):
            module.eval()
        return self


class CandidateGINE(nn.Module):
    """Dense GINE on candidates, without diagonal/self messages.

    h'_i = MLP((1 + eps) h_i + sum_{j != i} ReLU(h_j + W_e e_ij)).
    Each layer has its own edge projection, MLP, and trainable scalar epsilon.
    """

    def __init__(self, node_dim, edge_dim, hidden=32, layers=2, nonlinear_edges=False, dropout=0.):
        super().__init__()
        self.node_projection = nn.Linear(node_dim, hidden)
        self.edge_projections = nn.ModuleList([
            nn.Linear(edge_dim, hidden, bias=False) for _ in range(layers)
        ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU()) for _ in range(layers)
        ])
        if dropout:
            self.updates = nn.ModuleList([
                nn.Sequential(update[0], update[1], nn.Dropout(dropout),
                              update[2], update[3], nn.Dropout(dropout))
                for update in self.updates
            ])
        self.eps = nn.Parameter(torch.zeros(layers))
        if nonlinear_edges:
            # Construct extra weights last to preserve all other initializations
            # relative to the linear-edge variant under a matched seed.
            self.edge_projections = nn.ModuleList([
                nn.Sequential(projection, nn.ReLU(), nn.Linear(hidden, hidden, bias=False))
                for projection in self.edge_projections
            ])

    def forward(self, words, edges, edge_mask=None):
        h = self.node_projection(words)
        diagonal = torch.eye(h.shape[-2], device=h.device, dtype=torch.bool)
        excluded = diagonal if edge_mask is None else diagonal | ~edge_mask
        for layer, (edge_projection, update) in enumerate(zip(self.edge_projections, self.updates)):
            # Axes: (..., receiving word i, sending word j, hidden).
            messages = torch.relu(h.unsqueeze(-3) + edge_projection(edges))
            messages = messages.masked_fill(excluded[..., None], 0)
            h = update((1 + self.eps[layer]) * h + messages.sum(dim=-2))
        return torch.cat([h.mean(dim=-2), h.amax(dim=-2)], dim=-1)


class GatedGINEResidual(GatedEdgeResidual):
    """A frozen semantic scorer corrected by a gated candidate-local GINE."""

    def __init__(self, *, edge_hidden=32, gine_layers=2, nonlinear_edges=False,
                 gine_input_stage="candidate", gine_dropout=0., **config):
        if gine_input_stage not in {"projected", "global", "candidate"}:
            raise ValueError("Unknown GINE input stage")
        super().__init__(edge_hidden=edge_hidden, **config)
        self.gine_input_stage = gine_input_stage
        width = config.get("attn_width", 512)
        self.edge_encoder = CandidateGINE(width, self.edge_dim, edge_hidden, gine_layers, nonlinear_edges, gine_dropout)

    def forward(self, x, group_idx=None, edge_features=None):
        with torch.no_grad():
            if self.gine_input_stage == "candidate":
                words, edges = self.encode_candidate_words(x, group_idx, edge_features)
                branch_words = words
            else:
                words, edges, stages = self.encode_candidate_words(
                    x, group_idx, edge_features, return_stages=True)
                branch_words = stages[self.gine_input_stage]
            B, C, N, D = words.shape
            representation = self.subset_decoder(
                words.reshape(B * C, N, D), edges.reshape(B * C, N, N, self.edge_dim)
            ).reshape(B, C, D)
            base_score = self.scorer(representation)
        pooled = self.edge_encoder(branch_words, edges * self.residual_edge_mask)
        combined = torch.cat([representation, pooled], dim=-1)
        return base_score + self.edge_gate(combined).sigmoid() * self.edge_correction(combined)


class StandaloneCandidateGINE(GatedGINEResidual):
    """Score each candidate with GINE, optionally exposing raw edge summaries.

    No attention scores; edge_summary_width=0 preserves the original readout.
    """

    def __init__(self, *, edge_hidden=32, gine_layers=2, train_text_projection=False,
                 train_input_norm=False, raw_text_embeddings=False,
                 projection_width=None, adapter_hidden=None, single_layer_adapter=False,
                 gine_dropout=0., readout_width=128, sparse_edge_thresholds=None,
                 edge_coherence_scale=0., candidate_edge_channels=(), edge_summary_width=0, **config):
        if isinstance(edge_summary_width, bool) or not isinstance(edge_summary_width, int) or edge_summary_width < 0:
            raise ValueError('edge_summary_width must be a nonnegative integer')
        self.edge_summary_width = edge_summary_width
        if not 0 <= edge_coherence_scale < float('inf'):
            raise ValueError("edge_coherence_scale must be finite and nonnegative")
        self.edge_coherence_scale = float(edge_coherence_scale)
        channels = tuple(candidate_edge_channels)
        if len(set(channels)) != len(channels) or any(isinstance(c, bool) or not isinstance(c, int)
                or c < 0 or c >= config['edge_dim'] for c in channels):
            raise ValueError('Candidate edge channels must be distinct valid channel indices')
        self.candidate_edge_channels = channels
        if single_layer_adapter:
            if adapter_hidden is not None:
                raise ValueError("Choose either a single-layer adapter or a bottleneck adapter")
            adapter_hidden = config["in_dim"]
        super().__init__(edge_hidden=edge_hidden, gine_layers=gine_layers,
                         gine_input_stage="projected", gine_dropout=gine_dropout, **config)
        if sparse_edge_thresholds is not None:
            thresholds = torch.as_tensor(sparse_edge_thresholds)
            if thresholds.shape != (self.edge_dim,) or not torch.isfinite(thresholds).all() or (thresholds < 0).any():
                raise ValueError("Sparse thresholds must be finite nonnegative values, one per edge channel")
        # Stored in model_config; nonpersistent buffer preserves old checkpoint layouts.
        self.register_buffer("sparse_edge_thresholds", None if sparse_edge_thresholds is None else
                             torch.tensor(sparse_edge_thresholds, dtype=torch.float32), persistent=False)
        if projection_width is not None:
            if projection_width < 1 or raw_text_embeddings:
                raise ValueError("projection_width must be positive and requires projected embeddings")
            self.proj = nn.Sequential(nn.Linear(config["in_dim"], projection_width), nn.ReLU())
            self.edge_encoder = CandidateGINE(projection_width, self.edge_dim,
                                              edge_hidden, gine_layers, dropout=gine_dropout)
        self.train_text_projection = train_text_projection
        self.train_input_norm = train_input_norm
        self.raw_text_embeddings = raw_text_embeddings
        self.adapter_hidden = adapter_hidden
        if adapter_hidden is not None:
            if adapter_hidden < 0 or projection_width is not None or raw_text_embeddings:
                raise ValueError("Adapters require a nonnegative bottleneck and unprojected embeddings")
            self.input_norm.requires_grad_(True)
            self.train_input_norm = True
            self.edge_encoder = CandidateGINE(config["in_dim"], self.edge_dim,
                                              edge_hidden, gine_layers, dropout=gine_dropout)
        if train_text_projection:
            self.proj.requires_grad_(True)
        if train_input_norm:
            self.input_norm.requires_grad_(True)
        if raw_text_embeddings:
            self.edge_encoder = CandidateGINE(config["in_dim"], self.edge_dim,
                                              edge_hidden, gine_layers, dropout=gine_dropout)
        self.edge_gate.requires_grad_(False)
        self.edge_correction.requires_grad_(False)
        self.group_readout = nn.Sequential(
            nn.Linear(edge_hidden * 2, readout_width), nn.ReLU(), nn.Linear(readout_width, 1)
        )
        if gine_dropout:
            self.group_readout = nn.Sequential(self.group_readout[0], self.group_readout[1],
                                               nn.Dropout(gine_dropout), self.group_readout[2])
        if adapter_hidden:
            # Construct last so matched seeds retain identical initial GINE/readout tensors.
            if single_layer_adapter:
                self.embedding_adapter = nn.Sequential(
                    nn.Linear(config["in_dim"], config["in_dim"]), nn.ReLU(),
                )
            else:
                self.embedding_adapter = nn.Sequential(
                    nn.Linear(config["in_dim"], adapter_hidden), nn.ReLU(),
                    nn.Linear(adapter_hidden, config["in_dim"]),
                )
                nn.init.zeros_(self.embedding_adapter[-1].weight)
                nn.init.zeros_(self.embedding_adapter[-1].bias)
        if edge_summary_width:
            # Construct last: legacy GINE/readout initialization stays unchanged.
            self.edge_summary_encoder = nn.Sequential(
                nn.Linear(self.edge_dim * 3, edge_summary_width), nn.ReLU(),
                nn.Linear(edge_summary_width, edge_summary_width), nn.ReLU())
            original = self.group_readout[0]
            expanded = nn.Linear(original.in_features + edge_summary_width, original.out_features)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :original.in_features].copy_(original.weight)
                expanded.bias.copy_(original.bias)
            self.group_readout[0] = expanded

    def forward(self, x, group_idx=None, edge_features=None, candidate_edge_features=None):
        B, N, _ = x.shape
        if edge_features is None or edge_features.shape != (B, N, N, self.edge_dim):
            raise ValueError("Expected complete pairwise edge features")
        idx = self.grouper.idx if group_idx is None else group_idx.to(x.device)
        edges = edge_features[:, idx[:, :, None], idx[:, None, :], :]
        words = self.encode_input_words(x)
        candidates = self.grouper(words, group_idx=idx)
        B, C, group_size, D = candidates.shape
        candidate_edges = (edges * self.residual_edge_mask).reshape(B * C, group_size, group_size, self.edge_dim)
        if self.candidate_edge_channels:
            if candidate_edge_features is None or candidate_edge_features.shape != (B, C, len(self.candidate_edge_channels)):
                raise ValueError('Expected candidate_edge_features aligned with group_idx and configured channels')
            if not torch.isfinite(candidate_edge_features).all() or (candidate_edge_features < 0).any() or (candidate_edge_features > 1).any():
                raise ValueError('Candidate edge features must be finite and in [0, 1]')
            candidate_edges = candidate_edges.clone()
            values = candidate_edge_features.reshape(B*C, 1, 1, -1)
            values = values * self.residual_edge_mask[list(self.candidate_edge_channels)]
            off_diagonal = ~torch.eye(group_size, device=x.device, dtype=torch.bool)
            candidate_edges[..., list(self.candidate_edge_channels)] = values * off_diagonal[None,:,:,None]
        elif candidate_edge_features is not None:
            raise ValueError('Configure candidate_edge_channels before supplying candidate features')
        mask = None if self.sparse_edge_thresholds is None else self.edge_connectivity(candidate_edges)
        summary = (self.edge_summary_encoder(self.summarize_candidate_edges(candidate_edges))
                   if self.edge_summary_width else None)
        candidate_edges = self.modify_candidate_edges(candidate_edges)
        pooled = self.edge_encoder(candidates.reshape(B * C, group_size, D), candidate_edges, edge_mask=mask)
        if summary is not None:
            pooled = torch.cat([pooled, summary], dim=-1)
        return self.group_readout(pooled).reshape(B, C, 1)

    @staticmethod
    def summarize_candidate_edges(edges):
        """Per-channel mean, minimum, and weakest word's mean; exclude self edges.

        Uses raw masked candidate edges, including exact shared-witness overrides.
        Symmetric inputs therefore summarize the six unordered quartet pairs.
        Channel identity is retained; no pooling across relation dimensions.
        """
        count = edges.shape[-2]
        if count < 2 or edges.shape[-3] != count:
            raise ValueError('Expected a square candidate graph with at least two words')
        diagonal = torch.eye(count, device=edges.device, dtype=torch.bool)
        off_diagonal = edges.masked_fill(diagonal[..., None], 0)
        mean = off_diagonal.sum(dim=(-3, -2)) / (count * (count - 1))
        minimum = edges.masked_fill(diagonal[..., None], float('inf')).amin(dim=(-3, -2))
        weakest = (off_diagonal.sum(dim=-2) / (count - 1)).amin(dim=-2)
        return torch.cat([mean, minimum, weakest], dim=-1)

    def modify_candidate_edges(self, edges):
        """Amplify a channel when every candidate word supports that relation.

        Coherence is the minimum over words of their mean off-diagonal incoming
        edge value. Compute once on raw candidate edges, before GINE layers;
        preserve relation identity, connectivity, and the unmodified input tensor.
        The default zero scale returns the exact legacy tensor and computation.
        """
        if not self.edge_coherence_scale:
            return edges
        count = edges.shape[-2]
        if count < 2:
            raise ValueError("Coherence requires at least two candidate words")
        diagonal = torch.eye(count, device=edges.device, dtype=torch.bool)
        off_diagonal = edges.masked_fill(diagonal[..., None], 0)
        coherence = (off_diagonal.sum(dim=-2) / (count - 1)).amin(dim=-2)
        # Signed/experimental channels must not invert or attenuate messages.
        coherence = coherence.clamp(0, 1)
        return off_diagonal * (1 + self.edge_coherence_scale * coherence[..., None, None, :])

    def edge_connectivity(self, edges):
        """Keep a pair if any channel clears its training-fitted threshold.

        Zero is absence of evidence even for zero-valued thresholds. No self
        messages or nearest-neighbor fallback; isolated words retain their own
        representation through the usual GINE self term.
        """
        if self.sparse_edge_thresholds is None:
            keep = torch.ones(edges.shape[:-1], dtype=torch.bool, device=edges.device)
        else:
            keep = ((edges > 0) & (edges >= self.sparse_edge_thresholds)).any(dim=-1)
        diagonal = torch.eye(edges.shape[-2], device=edges.device, dtype=torch.bool)
        return keep & ~diagonal

    def encode_input_words(self, x):
        """Representations actually supplied to candidate-local GINE."""
        if self.adapter_hidden is not None:
            words = self.input_norm(x)
            if self.adapter_hidden:
                words = words + self.embedding_adapter(words)
        elif self.raw_text_embeddings:
            words = torch.nn.functional.layer_norm(x, (x.shape[-1],))
        else:
            words = self.proj(self.input_norm(x))
        return words

    def diagnostic_representations(self, nodes, edges):
        # Candidate GINE vectors depend on the other three words. There is no
        # single global GINE vector per word, so do not display inherited attention.
        return {"GINE input words": self.encode_input_words(nodes[None])[0]}

    def train(self, mode=True):
        super().train(mode)
        if self.train_text_projection:
            self.proj.train(mode)
        if self.train_input_norm:
            self.input_norm.train(mode)
        return self
