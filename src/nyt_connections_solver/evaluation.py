"""Puzzle loading and end-to-end search evaluation utilities."""

import argparse
import ast
import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import torch

from .harness import SearchHarness, SearchResult


@dataclass(frozen=True)
class Puzzle:
    date: str
    words: tuple[str, ...]
    groups: frozenset[frozenset[str]]


def load_puzzles(dataset_path: Path, selected_dates: Sequence[str]) -> list[Puzzle]:
    """Load each selected date once and validate its four groups of four."""
    selected = set(selected_dates)
    by_date: dict[str, list[tuple[str, ...]]] = {}
    with dataset_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            date = row["date"]
            if date in selected:
                group = ast.literal_eval(row["connections"])
                if not isinstance(group, list) or len(group) != 4 or not all(isinstance(w, str) for w in group):
                    raise ValueError(f"Invalid four-word group on {date}")
                by_date.setdefault(date, []).append(tuple(group))

    missing = selected - by_date.keys()
    if missing:
        raise ValueError(f"Dataset is missing {len(missing)} selected dates")

    puzzles = []
    for date in selected_dates:
        groups = by_date[date]
        words = tuple(word for group in groups for word in group)
        if len(groups) != 4 or len(set(words)) != 16:
            raise ValueError(f"Expected four disjoint groups of four on {date}")
        puzzles.append(Puzzle(date, words, frozenset(map(frozenset, groups))))
    return puzzles


def _validate_result(result: SearchResult, puzzle: Puzzle) -> None:
    if len(result.best_groups) != 4:
        raise ValueError(f"Expected four predicted groups on {puzzle.date}")
    predicted_words = [word for group in result.best_groups for word in group]
    if any(len(group) != 4 for group in result.best_groups) or set(predicted_words) != set(puzzle.words):
        raise ValueError(f"Predicted groups do not partition {puzzle.date}")
    if len(predicted_words) != 16:
        raise ValueError(f"Predicted groups repeat words on {puzzle.date}")

    for depth in (1, 2, 3):
        paths = result.beams_by_depth.get(depth)
        if not paths:
            raise ValueError(f"Missing depth-{depth} beam on {puzzle.date}")
        for path in paths:
            path_words = [word for group in path.groups for word in group]
            if len(path.groups) != depth or any(len(group) != 4 for group in path.groups):
                raise ValueError(f"Invalid depth-{depth} path on {puzzle.date}")
            if len(path_words) != 4 * depth or not set(path_words) <= set(puzzle.words):
                raise ValueError(f"Overlapping or unknown words in depth-{depth} path on {puzzle.date}")


def evaluate_puzzles(harness: SearchHarness, puzzles: Sequence[Puzzle]) -> dict:
    """Report exact puzzle solves and survival of a correct path in each beam."""
    if not puzzles:
        raise ValueError("No puzzles selected for evaluation")

    solved = 0
    correct_groups = 0
    oracle_counts = {depth: 0 for depth in (1, 2, 3)}
    for puzzle in puzzles:
        result = harness.solve(puzzle.words)
        _validate_result(result, puzzle)
        predicted = frozenset(map(frozenset, result.best_groups))
        solved += predicted == puzzle.groups
        correct_groups += len(predicted & puzzle.groups)
        for depth, paths in result.beams_by_depth.items():
            if depth in oracle_counts and any(
                all(frozenset(group) in puzzle.groups for group in path.groups)
                for path in paths
            ):
                oracle_counts[depth] += 1

    count = len(puzzles)
    return {
        "puzzles": count,
        "solved": solved,
        "solve_rate": solved / count,
        "mean_correct_groups": correct_groups / count,
        "oracle_beam_coverage": {
            depth: oracle_counts[depth] / count for depth in oracle_counts
        },
    }


def evaluate_oracle_next_group_ranking(
    harness: SearchHarness,
    puzzles: Sequence[Puzzle],
    *,
    top_ks: Sequence[int] = (1, 5, 20),
) -> dict[int, dict[str, float | int]]:
    """Rank the best true next group on clean oracle residual states.

    This separates state-level choice quality from beam path competition. Each
    puzzle contributes 1, 4, and 6 oracle states with 4, 3, and 2 groups left.
    """
    if not puzzles:
        raise ValueError("No puzzles selected for evaluation")
    if any(k < 1 for k in top_ks):
        raise ValueError("top_ks values must be positive")

    device = harness.device
    states: dict[int, list[tuple[torch.Tensor, tuple[int, ...]]]] = {4: [], 3: [], 2: []}
    for puzzle in puzzles:
        embeddings = harness.word_embedder.encode(
            list(puzzle.words), convert_to_tensor=True, show_progress_bar=False
        ).to(device)
        if embeddings.shape != (16, harness.input_dim):
            raise ValueError(
                f"Checkpoint expects {harness.input_dim} features per word, but "
                f"the embedder returned {tuple(embeddings.shape)}"
            )

        for remaining_count in (4, 3, 2):
            for group_indices in combinations(range(4), remaining_count):
                word_indices = [
                    group_index * 4 + offset
                    for group_index in group_indices
                    for offset in range(4)
                ]
                true_target_indices = tuple(
                    group_indices.index(group_index)
                    for group_index in group_indices
                )
                states[remaining_count].append(
                    (embeddings[word_indices], true_target_indices)
                )

    summaries = {}
    with torch.inference_mode():
        for remaining_count, state_rows in states.items():
            word_count = remaining_count * 4
            candidates = harness.group_indices[word_count]
            candidate_lookup = {
                tuple(candidate): position
                for position, candidate in enumerate(candidates.cpu().tolist())
            }
            target_candidates = torch.tensor(
                [
                    [
                        candidate_lookup[tuple(range(group_index * 4, group_index * 4 + 4))]
                        for group_index in target_indices
                    ]
                    for _, target_indices in state_rows
                ],
                dtype=torch.long,
                device=device,
            )
            best_ranks = []
            batch_size = harness.state_batch_size
            for start in range(0, len(state_rows), batch_size):
                batch_rows = state_rows[start : start + batch_size]
                x = torch.stack([row[0] for row in batch_rows])
                logits = harness.model(x, group_idx=candidates).squeeze(-1).float()
                batch_targets = target_candidates[start : start + len(batch_rows)]
                target_scores = logits.gather(1, batch_targets)
                ranks = 1 + (logits.unsqueeze(1) > target_scores.unsqueeze(2)).sum(dim=2)
                best_ranks.extend(ranks.min(dim=1).values.cpu().tolist())

            rank_tensor = torch.tensor(best_ranks, dtype=torch.float)
            summaries[remaining_count] = {
                "oracle_states": len(best_ranks),
                "best_true_group_mean_rank": rank_tensor.mean().item(),
                "best_true_group_mrr": rank_tensor.reciprocal().mean().item(),
                **{
                    f"best_true_group_top{k}": (rank_tensor <= k).float().mean().item()
                    for k in top_ks
                },
            }

    return summaries


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Notebook checkpoint (.pt)")
    parser.add_argument("--dataset", type=Path, help="Override the dataset path saved in the checkpoint")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    harness = SearchHarness(args.weights, beam_width=args.beam_width, device=args.device)
    checkpoint = harness.checkpoint
    if not isinstance(checkpoint, dict) or "split_dates" not in checkpoint:
        parser.error("Checkpoint must contain split_dates from the notebook")
    split_dates = checkpoint["split_dates"].get(args.split)
    if not split_dates:
        parser.error(f"Checkpoint contains no {args.split} dates")
    if args.dataset is None and "dataset_path" not in checkpoint:
        parser.error("Checkpoint must contain dataset_path or pass --dataset")
    dataset = args.dataset or Path(checkpoint["dataset_path"])
    puzzles = load_puzzles(dataset, split_dates)
    metrics = evaluate_puzzles(harness, puzzles)
    ranking_top_ks = tuple(sorted({1, 5, args.beam_width}))
    oracle_ranking = evaluate_oracle_next_group_ranking(
        harness, puzzles, top_ks=ranking_top_ks
    )

    print(f"{args.split}: {metrics['solved']}/{metrics['puzzles']} fully solved ({metrics['solve_rate']:.1%})")
    print(f"Mean correct groups: {metrics['mean_correct_groups']:.2f}/4")
    for depth, coverage in metrics["oracle_beam_coverage"].items():
        print(f"Oracle beam coverage after choice {depth}: {coverage:.1%}")
    print("Oracle next-group ranking (best true group among valid choices):")
    for remaining_count, summary in oracle_ranking.items():
        top_k = ", ".join(
            f"top-{k}: {summary[f'best_true_group_top{k}']:.1%}"
            for k in ranking_top_ks
        )
        print(
            f"  {remaining_count} groups left ({summary['oracle_states']} states): "
            f"mean rank {summary['best_true_group_mean_rank']:.1f}, "
            f"MRR {summary['best_true_group_mrr']:.3f}, {top_k}"
        )
