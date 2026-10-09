"""Interface for searching complete Connections partitions with a saved model."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer

from .models import AttentionModel, Baseline, SetTransformer


MODEL_TYPES = {
    "Baseline": Baseline,
    "AttentionModel": AttentionModel,
    "SetTransformer": SetTransformer,
}

@dataclass(frozen=True)
class SearchPath:
    """Groups chosen so far, in search order, and their aggregate score."""

    groups: tuple[tuple[str, ...], ...]
    score: float


@dataclass(frozen=True)
class SearchResult:
    """Best complete partition and retained paths after each scored choice."""

    best_groups: tuple[tuple[str, ...], ...]
    beams_by_depth: dict[int, tuple[SearchPath, ...]]


class SearchHarness:
    """Load a saved scorer once, ready for search over complete puzzles.

    The checkpoint must contain ``model_class``, ``model_config``,
    ``model_state_dict``, and ``embedding_model_name``. Evaluation checkpoints
    also carry ``split_dates`` and ``dataset_path`` for the CLI.

    Beam search scores states with 16, 12, and 8 words. ``solve`` returns the
    best four groups and retained paths at depths 1, 2, and 3; the fourth
    group is forced by elimination.
    """

    def __init__(
        self,
        weights_path: str | Path,
        *,
        beam_width: int = 10,
        device: str | torch.device = "cpu",
        state_batch_size: int = 4,
    ) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        if state_batch_size < 1:
            raise ValueError("state_batch_size must be positive")
        self.weights_path = Path(weights_path)
        self.beam_width = beam_width
        self.state_batch_size = state_batch_size
        self.device = torch.device(device)

        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.weights_path}")
        self.checkpoint = torch.load(
            self.weights_path,
            map_location="cpu",
            weights_only=True,
        )

        try:
            model_type = MODEL_TYPES[self.checkpoint["model_class"]]
            model_config = dict(self.checkpoint["model_config"])
            model_state_dict = self.checkpoint["model_state_dict"]
            embedding_model_name = self.checkpoint["embedding_model_name"]
        except (KeyError, TypeError) as error:
            required = "model_class, model_config, model_state_dict, embedding_model_name"
            raise ValueError(f"Checkpoint must contain {required}") from error

        # Forward calls supply their own candidate indices. This placeholder
        # only satisfies the model constructor's required group_idx argument.
        placeholder_group_idx = torch.tensor(list(combinations(range(4), 4)))
        model_config["group_idx"] = placeholder_group_idx
        self.model = model_type(**model_config).to(self.device)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
        self.input_dim = model_config["in_dim"]
        self.group_indices = {
            count: torch.tensor(
                list(combinations(range(count), 4)),
                dtype=torch.long,
                device=self.device,
            )
            for count in (16, 12, 8)
        }
        self.word_embedder = SentenceTransformer(
            embedding_model_name,
            device=str(self.device),
        )

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        word_embedder: SentenceTransformer,
        *,
        input_dim: int,
        beam_width: int = 10,
        device: str | torch.device = "cpu",
        state_batch_size: int = 4,
    ) -> "SearchHarness":
        """Build a harness around in-memory objects, useful for model selection."""
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        if state_batch_size < 1:
            raise ValueError("state_batch_size must be positive")

        harness = cls.__new__(cls)
        harness.weights_path = Path("<in-memory>")
        harness.beam_width = beam_width
        harness.state_batch_size = state_batch_size
        harness.device = torch.device(device)
        harness.checkpoint = {}
        harness.model = model.to(harness.device).eval()
        harness.input_dim = input_dim
        harness.group_indices = {
            count: torch.tensor(
                list(combinations(range(count), 4)),
                dtype=torch.long,
                device=harness.device,
            )
            for count in (16, 12, 8)
        }
        harness.word_embedder = word_embedder
        return harness

    def solve(self, words: Sequence[str]) -> SearchResult:
        """Return the highest-scoring complete partition and each retained beam."""
        words = tuple(words)
        if len(words) != 16 or any(not isinstance(word, str) for word in words):
            raise ValueError("Expected exactly 16 words")
        if len(set(words)) != 16:
            raise ValueError("Puzzle words must be distinct")

        with torch.inference_mode():
            embeddings = self.word_embedder.encode(
                list(words), convert_to_tensor=True, show_progress_bar=False
            ).to(self.device)  # (16, D)
            if embeddings.shape != (16, self.input_dim):
                raise ValueError(
                    f"Checkpoint expects {self.input_dim} features per word, but "
                    f"the embedder returned {tuple(embeddings.shape)}; checkpoints "
                    "with additional features are not yet supported"
                )

            remaining = torch.arange(16, device=self.device).unsqueeze(0)  # (1, 16)
            beam_scores = torch.zeros(1, device=self.device)  # (1,)
            beam_paths: list[tuple[tuple[int, ...], ...]] = [()]
            beams_by_depth: dict[int, tuple[SearchPath, ...]] = {}

            for depth, count in enumerate((16, 12, 8), start=1):
                candidates = self.group_indices[count]  # (C, 4), C = choose(count, 4)
                score_batches = []
                for start in range(0, len(remaining), self.state_batch_size):
                    end = start + self.state_batch_size
                    state_embeddings = embeddings[remaining[start:end]]  # (B, count, D)
                    logits = self.model(
                        state_embeddings, group_idx=candidates
                    ).squeeze(-1)  # (B, C)
                    log_probs = torch.log_softmax(logits.float(), dim=1)  # (B, C)
                    score_batches.append(
                        log_probs + beam_scores[start:end, None]
                    )  # (B, C)

                all_scores = torch.cat(score_batches).flatten()  # (beam_size * C,)
                top_scores, flat_indices = torch.topk(
                    all_scores, min(self.beam_width, all_scores.numel())
                )  # each (K,)
                parent_indices = flat_indices // len(candidates)  # (K,)
                chosen_positions = candidates[flat_indices % len(candidates)]  # (K, 4)
                parent_words = remaining[parent_indices]  # (K, count)
                chosen_words = parent_words.gather(1, chosen_positions)  # (K, 4)

                keep = torch.ones_like(parent_words, dtype=torch.bool)  # (K, count)
                keep.scatter_(1, chosen_positions, False)
                remaining = parent_words[keep].view(-1, count - 4)  # (K, count - 4)

                # Keep scoring and selection on GPU; only path metadata moves to CPU.
                parent_ids = parent_indices.cpu().tolist()
                chosen_ids = chosen_words.cpu().tolist()
                beam_paths = [
                    beam_paths[parent] + (tuple(group),)
                    for parent, group in zip(parent_ids, chosen_ids)
                ]
                beams_by_depth[depth] = tuple(
                    SearchPath(
                        groups=tuple(tuple(words[index] for index in group) for group in path),
                        score=score,
                    )
                    for path, score in zip(beam_paths, top_scores.cpu().tolist())
                )
                beam_scores = top_scores
                # TODO: prune equivalent remaining-word states if duplicates matter.

            best_indices = beam_paths[0] + (tuple(remaining[0].cpu().tolist()),)
            best_groups = tuple(
                tuple(words[index] for index in group) for group in best_indices
            )
            return SearchResult(best_groups, beams_by_depth)
