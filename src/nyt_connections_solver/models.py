"""Shared model architectures used by the notebook and search harness."""

from typing import Sequence

import torch
import torch.nn as nn


ATTENTION_DROPOUT = 0.3
SCORER_DROPOUT = 0.3
DEFAULT_MLP_DROPOUT = 0.1
DEFAULT_MLP_HIDDEN_DIMS = (256, 128)
SCORER_OUTPUT_DIM = 1

DEFAULT_ATTENTION_HEADS = 8
DEFAULT_ATTENTION_LAYERS = 1
BASE_PROJECTION_DIM = 256
DEFAULT_BASE_SCORER_LAYERS = (512, 256, 256)

DEFAULT_SET_LIN_LAYERS = 3
DEFAULT_SET_LIN_WIDTH = 1024
DEFAULT_SET_ATTN_WIDTH = 512
DEFAULT_SET_ATTN_HEADS = 4
GROUP_SUMMARY_SEEDS = 1
SEED_INTERACTION_LAYERS = 1


class MultiheadAttentionBlock(torch.nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            attn_heads=DEFAULT_ATTENTION_HEADS,
            dropout=ATTENTION_DROPOUT,
        ):
        super().__init__()

        # following the notation from the set transformer paper
        self.attn = torch.nn.MultiheadAttention(
            in_dim, num_heads=attn_heads, batch_first=True, dropout=dropout
        )
        self.h_norm = torch.nn.LayerNorm(in_dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU()
        )
        self.residual_proj = (
            torch.nn.Identity() if in_dim == out_dim else torch.nn.Linear(in_dim, out_dim, bias=False)
        )
        self.mab_norm = torch.nn.LayerNorm(out_dim)

    def forward(self, x, y):
        # H: self-attention with residual
        x = self.h_norm(x + self.attn(x, y, y)[0])

        # MAB. Project the residual whenever this block changes width.
        return self.mab_norm(self.residual_proj(x) + self.ff(x))


class SetEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dims: Sequence[int],
        num_mabs: int,
        attn_heads: Sequence[int],
        dropout: float = ATTENTION_DROPOUT,
    ):
        super().__init__()
        assert num_mabs == len(out_dims) == len(attn_heads), "num_mabs, out_dims, attn_heads must match in length"

        mabs = []
        cur_in = in_dim
        for i in range(num_mabs):
            mabs.append(MultiheadAttentionBlock(
                in_dim=cur_in,
                out_dim=out_dims[i],
                attn_heads=attn_heads[i],
                dropout=dropout,
            ))
            cur_in = out_dims[i]
        self.mabs = nn.ModuleList(mabs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mab in self.mabs:
            x = mab(x, x)
        return x


class SetDecoder(nn.Module):
    def __init__(
            self,
            in_dim,
            h1,
            out_dim,
            num_seeds,
            h2 = None,
            mab_attn_heads=DEFAULT_ATTENTION_HEADS,
            sab_attn_heads=DEFAULT_ATTENTION_HEADS,
            dropout: float = ATTENTION_DROPOUT,
        ):
        """
            Args:
                k: number of seed vectors
        """
        super().__init__()

        self.num_seeds = num_seeds

        # TODO: use Xavier intiializations?
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, h1))

        self.ff1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h1),
            torch.nn.ReLU()
        )

        self.mab = MultiheadAttentionBlock(
                in_dim=h1,
                out_dim=h2 if num_seeds > 1 else out_dim,
                attn_heads=mab_attn_heads,
                dropout=dropout,
        )

        # only need to use the set encoder when more than 1 vector is produced (model interactions between)
        if num_seeds > 1:
            self.sab = SetEncoder(
                in_dim=h2,
                out_dims=[h2] * SEED_INTERACTION_LAYERS,
                num_mabs=SEED_INTERACTION_LAYERS,
                attn_heads=[sab_attn_heads] * SEED_INTERACTION_LAYERS,
                dropout=dropout,
            )

            self.ff2 = torch.nn.Sequential(
                torch.nn.Linear(h2, out_dim),
                torch.nn.ReLU()
            )

    def forward(self, x):

        x = self.ff1(x)
        seeds = self.seed_vectors.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = self.mab(seeds, x)

        if self.num_seeds > 1:
            x = self.sab(x)
            x = self.ff2(x)

        return x


class MLPScorer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = DEFAULT_MLP_HIDDEN_DIMS,
        dropout: float = DEFAULT_MLP_DROPOUT,
    ):
        """
        Generic configurable MLP scorer.

        Args:
            input_dim:  size of input feature vector (e.g., 4*D)
            hidden_dims: list/tuple of hidden layer sizes
            output_dim: size of output (1 for scalar score)
            dropout: dropout probability between hidden layers
        """
        super().__init__()
        layers = []
        prev_dim = in_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, SCORER_OUTPUT_DIM))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, input_dim)
        return self.mlp(x)


class Grouper(nn.Module):
    def __init__(self, group_idx: torch.Tensor):
        super().__init__()
        # Ensure long dtype and register as a buffer so .to(device)/.cuda() moves it
        self.num_groups, self.group_size = group_idx.shape
        self.register_buffer("idx", group_idx.to(dtype=torch.long), persistent=False)

    def forward(self, x, group_idx=None):  #
        """
        Group items in x into all combinations of 4 items
        """
        B, _, D = x.shape
        idx = self.idx if group_idx is None else group_idx.to(device=x.device, dtype=torch.long)
        num_groups, group_size = idx.shape
        idx_flat = idx.reshape(-1)
        out = torch.index_select(x, dim=1, index=idx_flat)
        return out.view(B, num_groups, group_size, D)


# start with a simple model and build our way up
class Baseline(nn.Module):
    def __init__(self, in_dim, group_idx, layers: Sequence[int] = DEFAULT_BASE_SCORER_LAYERS):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, BASE_PROJECTION_DIM),
            nn.ReLU()
        )

        self.grouper = Grouper(group_idx)
        self.scorer = MLPScorer(
            BASE_PROJECTION_DIM * self.grouper.group_size,
            hidden_dims=layers,
            dropout=SCORER_DROPOUT,
        )

    def forward(self, x, group_idx=None):

        x = self.proj(x)
        x = self.grouper(x, group_idx=group_idx) # (B, num_groups, 4, D)

        # concatenate for scoring
        x = x.flatten(2, 3) # (B, 1820, D * 4)

        return self.scorer(x)  # (B, num_groups, 1)


# start with a simple model and build our way up
class AttentionModel(nn.Module):
    def __init__(
        self,
        in_dim,
        group_idx,
        layers: Sequence[int] = DEFAULT_BASE_SCORER_LAYERS,
        attn_layers=DEFAULT_ATTENTION_LAYERS,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, BASE_PROJECTION_DIM),
            nn.ReLU()
        )

        self.grouper = Grouper(group_idx)
        self.encoder = SetEncoder(
            BASE_PROJECTION_DIM,
            [BASE_PROJECTION_DIM] * attn_layers,
            attn_layers,
            [DEFAULT_ATTENTION_HEADS] * attn_layers
        )
        self.scorer = MLPScorer(
            BASE_PROJECTION_DIM * self.grouper.group_size,
            hidden_dims=layers,
            dropout=SCORER_DROPOUT,
        )

    def forward(self, x, group_idx=None):

        x = self.proj(x)
        x = self.encoder(x)
        x = self.grouper(x, group_idx=group_idx) # (B, num_groups, 4, D)

        # concatenate for scoring
        x = x.flatten(2, 3) # (B, 1820, D * 4)

        return self.scorer(x)  # (B, num_groups, 1)


class SetTransformer(torch.nn.Module):
    def __init__(
            self,
            in_dim,
            group_idx,
            lin_layers=DEFAULT_SET_LIN_LAYERS,
            lin_width=DEFAULT_SET_LIN_WIDTH,
            attn_layers=DEFAULT_ATTENTION_LAYERS,
            attn_width=DEFAULT_SET_ATTN_WIDTH,
            attn_heads=DEFAULT_SET_ATTN_HEADS,
            dropout=SCORER_DROPOUT,
        ):
        super().__init__()

        self.input_norm = nn.LayerNorm(in_dim)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, attn_width),
            nn.ReLU()
        )

        self.full_encoder = SetEncoder(
            attn_width,
            [attn_width] * attn_layers,
            attn_layers,
            [attn_heads] * attn_layers,
            dropout=dropout,
        )

        self.grouper = Grouper(group_idx)

        self.subset_encoder = SetEncoder(
            in_dim=attn_width,
            out_dims=[attn_width] * attn_layers,
            num_mabs=attn_layers,
            attn_heads=[attn_heads] * attn_layers,
            dropout=dropout,
        )

        # decode each subset to a single "group vector" to summarize group coherence
        self.subset_decoder = SetDecoder(
            in_dim=attn_width,
            h1=attn_width,
            out_dim=attn_width,
            num_seeds=GROUP_SUMMARY_SEEDS,
            mab_attn_heads=attn_heads,
            dropout=dropout,
        )

        self.scorer = MLPScorer(
            attn_width, hidden_dims=[lin_width] * lin_layers, dropout=dropout
        )

    def forward(self, x, group_idx=None):
        x = self.input_norm(x)
        x = self.proj(x)
        x = self.full_encoder(x)
        x = self.grouper(x, group_idx=group_idx) # (B, num_groups, 4, D)

        B, num_groups, group_size, D = x.shape
        x = x.view(B * num_groups, group_size, D)
        x = self.subset_encoder(x)
        x = self.subset_decoder(x).squeeze(1)
        x = x.view(B, num_groups, D)

        return self.scorer(x)  # (B, num_groups, 1)
