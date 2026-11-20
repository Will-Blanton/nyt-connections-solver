#!/usr/bin/env python3
"""Evaluate a saved graph checkpoint with feedback search on its held-out split."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nyt_connections_solver.checkpoints import load_graph_harness
from nyt_connections_solver.evaluation.feedback import evaluate_feedback_games
from nyt_connections_solver.evaluation.search import Puzzle
from nyt_connections_solver.feedback_search import FeedbackSearchHarness, FeedbackConstraints
from nyt_connections_solver.feedback_policies import PartitionBeliefPolicy
from nyt_connections_solver.data_loader import load_connections


class RankedPolicy:
    """Record where the chosen guess sat in the current state's raw ranking."""
    def __init__(self, policy):
        self.policy = policy
        self.games = []
        self.partition_games = []

    def choose(self, state):
        if not state.history:
            self.games.append([])
            self.partition_games.append([])
        chosen = self.policy.choose(state)
        candidates = self.policy.last_trace.candidates
        ordered = sorted(candidates, key=lambda row: row.log_probability, reverse=True)
        selected = frozenset(chosen)
        raw_rank = next(i for i, row in enumerate(ordered, 1)
                        if frozenset(row.words) == selected)
        feasible = FeedbackConstraints(state).condition_candidates(candidates)
        feasible_ordered = sorted(feasible, key=lambda row: row.log_probability, reverse=True)
        feasible_rank = next(i for i, row in enumerate(feasible_ordered, 1)
                             if frozenset(row.words) == selected)
        self.games[-1].append({'raw_rank': raw_rank, 'feasible_rank': feasible_rank,
                               'candidate_count': len(candidates), 'feasible_count': len(feasible)})
        depth = self.policy.last_trace.beams_by_depth
        paths = depth[-2] if len(depth) > 1 else depth[-1]
        partitions = []
        for path in paths:
            used = set().union(*(set(group) for group in path.groups))
            tail = tuple(word for word in state.remaining if word not in used)
            groups = (*path.groups, *((tail,) if tail else ()))
            partitions.append((path.log_probability, frozenset(map(frozenset, groups))))
        self.partition_games[-1].append((frozenset(state.remaining), partitions))
        return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--beam-width', type=int, default=20)
    parser.add_argument('--full-board-weight', type=float, default=0.0)
    parser.add_argument('--score-mode', choices=('path', 'full_board'), default='full_board')
    parser.add_argument('--weakest-group-weight', type=float, default=0.0)
    parser.add_argument('--policy', choices=('partition', 'marginal', 'information'), default='partition')
    parser.add_argument('--max-mistakes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    dates = checkpoint['split_dates'][args.split]
    if not dates:
        raise ValueError(f'Checkpoint has no {args.split} dates; choose a nonempty held-out split')
    frame = load_connections(Path(checkpoint['dataset_path']))
    puzzles = []
    for date in dates:
        rows = frame.loc[frame['date'] == date, 'connections'].tolist()
        if len(rows) != 4:
            raise ValueError(f'Expected four groups for {date}, got {len(rows)}')
        groups = tuple(tuple(row) for row in rows)
        puzzles.append(Puzzle(date, tuple(word for group in groups for word in group),
                              frozenset(map(frozenset, groups)), groups))
    harness = load_graph_harness(args.checkpoint, beam_width=args.beam_width)
    if args.policy == 'partition':
        search = FeedbackSearchHarness.from_harness(
            harness, beam_width=args.beam_width, capture_trace=True,
            full_board_weight=args.full_board_weight, score_mode=args.score_mode,
            weakest_group_weight=args.weakest_group_weight)
    else:
        search = PartitionBeliefPolicy(harness, beam_width=args.beam_width,
                                       information_weight=0.25 if args.policy == 'information' else 0.0)
    policy = RankedPolicy(search)
    result = evaluate_feedback_games(policy, puzzles, max_mistakes=args.max_mistakes, seed=args.seed)
    colors = ('purple', 'blue', 'green', 'yellow')
    details = []
    for game, puzzle, ranks, partition_steps in zip(
            result.pop('games'), puzzles, policy.games, policy.partition_games):
        groups = {frozenset(group): color for group, color in zip(puzzle.ordered_groups, colors)}
        history = []
        for guess, rank, (remaining, partitions) in zip(game.pop('history'), ranks, partition_steps):
            chosen = frozenset(guess.guess)
            overlaps = {color: len(chosen & group) for group, color in groups.items()}
            gold_remaining = frozenset(group for group in puzzle.groups if group <= remaining)
            ranked_partitions = sorted(partitions, key=lambda row: row[0], reverse=True)
            gold_ranks = [i for i, (_, candidate_partition) in enumerate(ranked_partitions, 1)
                          if candidate_partition == gold_remaining]
            history.append({'words': list(guess.guess), 'outcome': guess.outcome.value, **rank,
                            'max_gold_overlap': max(overlaps.values()),
                            'gold_partition_in_beam': bool(gold_ranks),
                            'gold_partition_beam_rank': gold_ranks[0] if gold_ranks else None,
                            'distinct_beam_partitions': len({partition for _, partition in partitions}),
                            'color': groups.get(chosen)})
        details.append({**game, 'history': history,
                        'solved_colors': [groups[frozenset(row['words'])] for row in history
                                          if row['outcome'] == 'correct']})
        print(f"{puzzle.date}: {'W' if game['won'] else 'L'} "
              f"{game['correct_groups']}/4, {game['mistakes']} mistakes", flush=True)
    payload = {'checkpoint': str(args.checkpoint), 'split': args.split, 'policy': args.policy,
               'full_board_weight': args.full_board_weight,
               'score_mode': args.score_mode,
               'weakest_group_weight': args.weakest_group_weight,
               'beam_width': args.beam_width, 'max_mistakes': args.max_mistakes,
               'shuffle_seed': args.seed, 'summary': result, 'games': details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    print(f'Result: {args.output}', flush=True)


if __name__ == '__main__':
    main()
