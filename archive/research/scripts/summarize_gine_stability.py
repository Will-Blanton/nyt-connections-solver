#!/usr/bin/env python3
"""Summarize matched GINE edge/control seeds on the fixed validation split."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("experiments/graph-features"))
    parser.add_argument("--output", type=Path, default=Path("experiments/gine-stability.md"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--allow-early-stop", action="store_true")
    parser.add_argument("--variants", nargs=2,
                        default=["gated-gine-zero-edge-control", "gated-gine-residual"])
    args = parser.parse_args()
    variants = tuple(args.variants)
    runs = {}
    for path in sorted(args.results.glob("*.json")):
        data = json.loads(path.read_text())
        config = data.get("config", {})
        if config.get("variant") not in variants or config.get("seed") not in args.seeds:
            continue
        if config.get("updates") != args.updates:
            continue
        if not args.allow_early_stop and data["training"]["updates_ran"] != args.updates:
            continue
        runs[config["seed"], config["variant"]] = (path, data)
    for seed in args.seeds:
        for variant in variants:
            if (seed, variant) not in runs:
                raise ValueError(f"Missing full-budget run for seed {seed}, {variant}")
    reference = runs[args.seeds[0], variants[0]][1]["config"]
    for _, data in runs.values():
        for key in ("reference_checkpoint", "dataset", "lr", "weight_decay", "batch_size", "updates", "eval_every"):
            if data["config"][key] != reference[key]:
                raise ValueError(f"Unmatched configuration: {key}")
        model_config = dict(data["config"]["model_config"])
        model_config.pop("residual_edge_channels", None)
        expected = dict(reference["model_config"])
        expected.pop("residual_edge_channels", None)
        if model_config != expected:
            raise ValueError("Unmatched GINE architecture")

    def value(seed, variant, size, metric):
        return runs[seed, variant][1]["training"]["validation"][size][metric]

    def spread(values, percent=False):
        scale = 100 if percent else 1
        precision = 2 if percent else 4
        suffix = "%" if percent else ""
        sd = stdev(values) if len(values) > 1 else 0
        return f"{mean(values)*scale:.{precision}f} ± {sd*scale:.{precision}f}{suffix}"

    lines = ["# GINE stability across matched branch seeds", "",
             f"Fixed text-only checkpoint and date split; up to {args.updates:,} updates per branch. "
             "Only branch initialization and batch/state order vary across seeds. "
             "Each edge/control pair shares its seed. The text backbone stays frozen.", "",
             f"Variants: `{variants[0]}` (control), `{variants[1]}` (edges). "
             "Each run selects its best checkpoint by validation s4 Hit@1, then s4 MRR. "
             "All levels below use that same checkpoint. Update 0 is eligible. "
             "No test split or search-harness games are evaluated.", "",
             "## Full-board results by seed", "",
             "| Seed | Control update | Edge update | Control Hit@1 | Edge Hit@1 | Delta (pp) | Control MRR | Edge MRR |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for seed in args.seeds:
        updates = [runs[seed, variant][1]["training"]["best_update"] for variant in variants]
        h = [value(seed, variant, "s4", "hit1") for variant in variants]
        m = [value(seed, variant, "s4", "mrr") for variant in variants]
        lines.append(f"| {seed} | {updates[0]} | {updates[1]} | {h[0]:.2%} | {h[1]:.2%} | {(h[1]-h[0])*100:+.2f} | {m[0]:.4f} | {m[1]:.4f} |")
    lines += ["", "## Ranking across all levels", "",
              "Mean ± sample standard deviation across seeds. Delta is the paired edge-minus-control mean; "
              "positive-seed counts exclude ties. Percentage deltas are percentage points.", "",
              "| Level | Metric | Zero-edge control | With edges | Paired delta | Positive seeds |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for size in ("s4", "s3", "s2"):
        for metric in ("hit1", "mrr", "hit4"):
            control = [value(seed, variants[0], size, metric) for seed in args.seeds]
            edges = [value(seed, variants[1], size, metric) for seed in args.seeds]
            delta = [b-a for a, b in zip(control, edges)]
            percent = metric != "mrr"
            lines.append(f"| {size} | {metric} | {spread(control, percent)} | {spread(edges, percent)} | "
                         f"{mean(delta)*(100 if percent else 1):+.4f} | {sum(d > 0 for d in delta)}/{len(delta)} |")
    lines += ["", "## Per-color group ranking", "",
              "These measure each color's specific true group, rather than any correct group.", "",
              "| Level | Color | Control top1 | Edge top1 | Control top4 | Edge top4 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for size in ("s4", "s3", "s2"):
        for color in ("yellow", "green", "blue", "purple"):
            cells = [spread([runs[seed, variant][1]["training"]["validation"][size]
                             ["by_color"][color][metric] for seed in args.seeds], True)
                     for metric in ("top1", "top4") for variant in variants]
            lines.append(f"| {size} | {color} | " + " | ".join(cells) + " |")
    lines += ["", "## Scope and limitations", "",
              "This measures branch/optimization stability conditional on one pretrained backbone and one "
              "validation split. It does not measure uncertainty across puzzle samples or backbone retraining. "
              "The same validation puzzles are reused across seeds; treat them as paired repeats, not independent "
              "new puzzle observations. s3/s2 are oracle residual states formed by removing true groups, "
              f"not visited search trajectories. Results are conditional on the {args.updates:,}-update "
              "budget and the recorded stopping policy.", "", "## Run artifacts", "",
              "Reference: `" + reference["reference_checkpoint"] + "`.", "",
              "Reproduce a run with `nix develop -c uv run python scripts/run_graph_feature_experiment.py "
              "--variant VARIANT --reference-checkpoint weights/experiments/20261023T150230Z-text-only.pt "
              f"--updates {args.updates} --min-updates-before-stop 1500 --patience 8 --seed SEED`. "
              "See each JSON configuration for exact stopping settings.", ""]
    for seed in args.seeds:
        for variant in variants:
            path, _ = runs[seed, variant]
            lines.append(f"- Seed {seed}, `{variant}`: `{path}`")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
