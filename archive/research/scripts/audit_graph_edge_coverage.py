#!/usr/bin/env python3
"""Audit existing full-board validation features and missed groups; no training."""
import argparse
import json
import re
from pathlib import Path
from statistics import mean

import torch
from sentence_transformers import SentenceTransformer

from run_graph_feature_experiment import (
    load_connections, make_artifacts, score_batch,
)
from nyt_connections_solver.data_loader import ConnectionsDataset
from nyt_connections_solver.graph_features import build_graph_features
from nyt_connections_solver.graph_models import GatedGINEResidual
from nyt_connections_solver.edge_features.resources import phrase_form


COLORS = ("purple", "blue", "green", "yellow")


def candidate_evidence(edges, idx, positive):
    """Return six-pair support and tie-aware mean-evidence percentiles.

    Percentile 0.5 means tied with all distractors, not strong discrimination.
    All true groups are excluded from the distractor reference distribution.
    """
    pairs = torch.triu_indices(4, 4, offset=1)
    values = edges[idx[:, pairs[0]], idx[:, pairs[1]], :]
    pooled = values.mean(1)
    negative = pooled[~positive]
    percentiles = []
    for row in pooled[positive]:
        percentiles.append(((negative < row).float() + .5 * (negative == row)).mean(0))
    return values, pooled, torch.stack(percentiles)


def phrase_witnesses(words, index):
    completions = [set(index.get(phrase_form(word), ())) for word in words]
    common = set.intersection(*completions)
    all_pairs = all(completions[i] & completions[j] for i in range(4) for j in range(i+1, 4))
    return {"all_six_pairs_supported": bool(all_pairs),
            "common_to_all_four": sorted(common)}


def closed_compound_witnesses(words, vocabulary):
    """Attested concatenations missed by a space-separated phrase index.

    Require single alphabetic puzzle entries and a completion that is itself
    an English vocabulary word of at least two letters. No invented compounds.
    """
    words = [phrase_form(word) for word in words]
    if not all(word.isalpha() for word in words):
        return {"preceding": [], "following": []}
    first = words[0]
    following, preceding = set(), set()
    for entry in vocabulary:
        if entry.startswith(first):
            suffix = entry[len(first):]
            if len(suffix) >= 2 and suffix in vocabulary and all(word + suffix in vocabulary for word in words):
                following.add(suffix)
        if entry.endswith(first):
            prefix = entry[:-len(first)]
            if len(prefix) >= 2 and prefix in vocabulary and all(prefix + word in vocabulary for word in words):
                preceding.add(prefix)
    return {"preceding": sorted(preceding), "following": sorted(following)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("experiments/graph-features"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/edge-coverage"))
    parser.add_argument("--variant", default="gated-gine-residual",
                        help="Existing long-budget GINE variant to audit across seeds 0-3.")
    parser.add_argument("--reference-checkpoint", type=Path,
                        default=Path("weights/experiments/20261023T150230Z-text-only.pt"))
    args = parser.parse_args()
    reference = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=True)
    dates = set(reference["split_dates"]["val"])
    frame = load_connections(Path(reference["dataset_path"]))
    frame = frame[frame.date.isin(dates)]
    if frame.date.nunique() != len(dates):
        raise ValueError("Incomplete validation split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    options = reference["feature_metadata"]["options"]
    embedder = SentenceTransformer(reference["embedding_model_name"],
                                   revision=options["revision"], device=str(device))
    nodes, edges, metadata = build_graph_features(embedder, reference["embedding_model_name"],
                                                 cache_dir="data/feature_cache/graph", **options)
    if metadata != reference["feature_metadata"]:
        raise ValueError("Feature metadata differs from the reference")
    lexicon = next(provider.lexicon for provider in edges.providers if hasattr(provider, "lexicon"))
    boards = ConnectionsDataset.prepare(frame, embedder, None, nodes, edges)
    embedder.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    artifact = make_artifacts()[4]
    idx, positive = artifact["idx"], artifact["y"]
    targets = positive.nonzero().flatten()
    names = list(edges.feature_names)
    categories = {str(date): groups["category"].tolist() for date, groups in frame.groupby("date")}
    groups, pooled_by_date = [], {}
    for board in boards:
        values, pooled, percentiles = candidate_evidence(board["edge_features"], idx, positive)
        pooled_by_date[str(board["date"])] = pooled
        for color_index, target in enumerate(targets):
            words = board["words"][color_index*4:color_index*4+4]
            support = values[target] > 0
            groups.append({
                "date": str(board["date"]), "color": COLORS[color_index], "words": words,
                "category": categories[str(board["date"])][color_index],
                "pair_support_fraction": support.float().mean(0).tolist(),
                "any_pair": support.any(0).tolist(), "all_six_pairs": support.all(0).tolist(),
                "mean_evidence": pooled[target].tolist(),
                "incorrect_candidate_percentile": percentiles[color_index].tolist(),
                "preceding": phrase_witnesses(words, lexicon.preceding),
                "following": phrase_witnesses(words, lexicon.following),
                "closed_compounds": closed_compound_witnesses(words, lexicon.words),
            })
    checkpoints = {}
    for path in sorted(args.results.glob("*.json")):
        data = json.loads(path.read_text())
        config = data.get("config", {})
        if config.get("variant") == args.variant and config.get("updates") == 3000:
            checkpoints[config["seed"]] = Path("weights/experiments") / (data["run_name"] + ".pt")
    if set(checkpoints) != {0, 1, 2, 3}:
        raise ValueError("Expected four existing long-budget GINE checkpoints for the requested variant")
    model_rows = []
    for seed, path in sorted(checkpoints.items()):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["split_dates"] != reference["split_dates"] or checkpoint["feature_metadata"] != metadata:
            raise ValueError("Audit checkpoints must share the feature definitions and split")
        model = GatedGINEResidual(group_idx=idx, **checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device).eval()
        for start in range(0, len(boards), 16):
            batch_boards = boards[start:start+16]
            batch = {"x": torch.stack([b["x"] for b in batch_boards]),
                     "edge_features": torch.stack([b["edge_features"] for b in batch_boards])}
            with torch.inference_mode():
                logits = score_batch(model, batch, idx.to(device), device).squeeze(-1).cpu()
            orders = logits.argsort(dim=1, descending=True)
            ranks = orders.argsort(dim=1) + 1
            for board, order, rank in zip(batch_boards, orders, ranks):
                hard = order[~positive[order]][:4]
                pooled = pooled_by_date[str(board["date"])]
                for color_index, target in enumerate(targets):
                    true = pooled[target]
                    negatives = pooled[hard]
                    hard_percentile = ((negatives < true).float() + .5 * (negatives == true)).mean(0)
                    model_rows.append({"seed": seed, "date": str(board["date"]),
                                       "color": COLORS[color_index], "rank": int(rank[target]),
                                       "hard_negative_percentile": hard_percentile.tolist()})
        del model
        print(f"Audited seed {seed}: {path}", flush=True)

    lookup = {(row["date"], row["color"]): row for row in groups}
    summary = {}
    for color in COLORS:
        static = [row for row in groups if row["color"] == color]
        missed = [row for row in model_rows if row["color"] == color and row["rank"] > 4]
        channels = {}
        for channel, name in enumerate(names):
            channels[name] = {
                "any_pair_rate": mean(row["any_pair"][channel] for row in static),
                "all_six_pairs_rate": mean(row["all_six_pairs"][channel] for row in static),
                "mean_pair_support": mean(row["pair_support_fraction"][channel] for row in static),
                "mean_incorrect_candidate_percentile": mean(row["incorrect_candidate_percentile"][channel] for row in static),
                "missed_mean_incorrect_candidate_percentile": mean(lookup[row["date"], color]["incorrect_candidate_percentile"][channel] for row in missed),
                "missed_hard_negative_percentile": mean(row["hard_negative_percentile"][channel] for row in missed),
            }
        templates = [row for row in static if re.search(r"_{2,}", row["category"])]
        summary[color] = {"groups": len(static), "missed_top4_appearances": len(missed), "channels": channels,
                          "explicit_blank_templates": len(templates),
                          "templates_with_shared_completion": sum(bool(row["preceding"]["common_to_all_four"] or row["following"]["common_to_all_four"]) for row in templates),
                          "templates_with_closed_compound_completion": sum(bool(row["closed_compounds"]["preceding"] or row["closed_compounds"]["following"]) for row in templates),
                          "phrase": {direction: {
                              "any_pair_count": sum(row["any_pair"][names.index("shared_preceding_word" if direction == "preceding" else "shared_following_word")] for row in static),
                              "all_six_pair_count": sum(row[direction]["all_six_pairs_supported"] for row in static),
                              "shared_four_count": sum(bool(row[direction]["common_to_all_four"]) for row in static),
                              "all_pairs_without_shared_four_count": sum(row[direction]["all_six_pairs_supported"] and not row[direction]["common_to_all_four"] for row in static),
                          } for direction in ("preceding", "following")}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"split": "val", "audited_variant": args.variant, "feature_names": names, "feature_metadata": metadata,
              "checkpoints": {seed: str(path) for seed, path in checkpoints.items()},
              "summary": summary, "groups": groups, "model_group_ranks": model_rows}
    (args.output_dir / "coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Existing edge feature coverage", "",
             "Full-board validation only: 121 puzzles, four true groups per puzzle. Model misses use "
             f"four existing long-budget `{args.variant}` seeds. No training or test evaluation.", "",
             "A miss means the specific true group ranks below the first four candidates. "
             "Coverage is counted once per puzzle/group; missed diagnostics count group appearances "
             "across model seeds and are correlated. Nonzero support is not proof of a shared explanation.", "",
             "Mean-evidence percentiles compare each true group's mean of six edge values with all "
             "1,816 incorrect quads, excluding all four true groups. Ties receive half credit; "
             "0.5 is uninformative. Hard-negative percentiles use the four highest model-ranked incorrect "
             "quads. These univariate diagnostics cannot establish a multichannel model's feature ceiling.", "",
             "## Purple coverage and discrimination", "",
             "Dense semantic channels can be nonzero everywhere. Sparse channels that fire on only "
             "one pair may not explain all four words. All-six support also need not have one common witness.", "",
             "| Channel | Any pair | All six pairs | True vs incorrect percentile | Missed: vs incorrect | Missed: vs hard negatives |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in names:
        row = summary["purple"]["channels"][name]
        lines.append(f"| {name} | {row['any_pair_rate']:.1%} | {row['all_six_pairs_rate']:.1%} | "
                     f"{row['mean_incorrect_candidate_percentile']:.3f} | {row['missed_mean_incorrect_candidate_percentile']:.3f} | {row['missed_hard_negative_percentile']:.3f} |")
    lines += ["", "## Phrase evidence by color", "",
              "| Color | Direction | Any pair /121 | All six pairs /121 | Shared completion for all four /121 | All pairs without shared completion |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for color in reversed(COLORS):
        for direction, row in summary[color]["phrase"].items():
            lines.append(f"| {color} | {direction} | {row['any_pair_count']} | {row['all_six_pair_count']} | {row['shared_four_count']} | {row['all_pairs_without_shared_four_count']} |")
    lines += ["", "## Explicit blank-template coverage", "",
              "Labels containing two or more underscores are a narrow diagnostic subset, "
              "not a comprehensive classifier of phrase-based categories. Labels are not model inputs.", "",
              "| Color | Explicit templates | Current phrase index: common completion | Closed compounds in same vocabulary |",
              "| --- | ---: | ---: | ---: |"]
    for color in reversed(COLORS):
        lines.append(f"| {color} | {summary[color]['explicit_blank_templates']} | {summary[color]['templates_with_shared_completion']} | {summary[color]['templates_with_closed_compound_completion']} |")
    lines += ["", "## Purple examples with a common phrase completion", "",
              "Ranks are of the true group among all candidates, in seed order 0, 1, 2, 3.", ""]
    for row in groups:
        if row["color"] == "purple" and any(row[d]["common_to_all_four"] for d in ("preceding", "following")):
            ranks = [r["rank"] for r in model_rows if r["date"] == row["date"] and r["color"] == "purple"]
            lines.append(f"- {row['date']}: {row['category']}; {row['words']}; "
                         f"preceding={row['preceding']['common_to_all_four']}; following={row['following']['common_to_all_four']}; ranks={ranks}")
    lines += ["", "## Closed-compound evidence omitted by the current phrase index", "",
              "Diagnostic only: all four concatenations and the completion itself must occur in the "
              "same WordNet English vocabulary. No new features were added. Prefixes that are not "
              "standalone vocabulary words are excluded; this is a conservative coverage check.", ""]
    for row in groups:
        if row["color"] == "purple" and any(row["closed_compounds"].values()):
            lines.append(f"- {row['date']}: {row['category']}; {row['words']}; {row['closed_compounds']}")
    lines += ["", "Raw per-group evidence, all color/channel summaries, and per-seed group ranks "
              "are stored in `coverage.json`. Category descriptions are inspected only for diagnosis; "
              "they are not inputs to feature construction or the trained models."]
    (args.output_dir / "coverage.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
