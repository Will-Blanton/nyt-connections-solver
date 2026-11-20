"""Snapshot small research reports without committing generated experiment outputs.

Run on the research machine, where experiments/ and its selection files exist.
Existing snapshots are never deleted automatically. No training or evaluation runs.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "experiments"
    destination = ROOT / "notes" / "experiments"
    reports = sorted(source.rglob("*.md"))
    if not reports:
        raise SystemExit("No local experiment reports found; existing snapshots left intact.")

    # Read required provenance before writing, so a deployment-only checkout cannot
    # silently replace the manifest with an incomplete local inventory.
    refit = source / "gine-v1-refit"
    selection = json.loads((refit / "selection.json").read_text())
    sealed = json.loads((refit / "sealed-checkpoints.json").read_text())
    checkpoint = next(item for item in sealed if item["path"] == selection["checkpoint"])
    cv = json.loads((source / "gine-coherence-cv" / "best-baseline.json").read_text())
    protocol = json.loads((refit / "manifest.json").read_text())
    run = json.loads((refit / f"{selection['winner']}-seed-{selection['primary_seed']}.json").read_text())
    if run["status"] != "complete" or run["checkpoint"] != selection["checkpoint"]:
        raise SystemExit("Selected run is incomplete or does not match the checkpoint record.")

    inventory = []
    for report in reports:
        relative = report.relative_to(source)
        payload = report.read_bytes()
        target = destination / "reports" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        title = next(
            (line.lstrip("# ") for line in payload.decode().splitlines() if line.startswith("# ")),
            report.stem,
        )
        inventory.append({
            "title": title,
            "local_source": report.relative_to(ROOT).as_posix(),
            "snapshot": target.relative_to(destination).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        })

    manifest = {
        "schema_version": 1,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Markdown report snapshots, not an inventory of every optimizer trial or raw run",
        "raw_artifacts": "experiments/ (ignored; research machine only)",
        "report_count": len(inventory),
        "selected_baseline": {
            **selection,
            "checkpoint_sha256": checkpoint["sha256"],
            "cv_selection_record": cv,
            "refit_report": "reports/gine-v1-refit/report.md",
            "model_config": run["config"],
            "fixed_refit_updates": run["training"]["updates_ran"],
            "training_puzzles": len(protocol["train_dates"]),
            "test_puzzles": len(protocol["test_dates"]),
            "batch_size": protocol["batch_size"],
            "loss": {
                key: run["training"][key]
                for key in ("loss", "floor_weight", "floor_beta")
            },
            "primary_checkpoint_test_solves": run["evaluation"]["harness"]["solved"],
            "primary_checkpoint_test_hit1": run["evaluation"]["ranking"]["s4"]["hit1"],
            "harness": protocol["harness"],
        },
        "known_incomplete_studies": [{
            "report": "reports/gine-transformation-capacity.md",
            "status": "interrupted; report is a historical progress snapshot, not a completed result",
        }],
        "reports": inventory,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(inventory)} reports ({sum(item['bytes'] for item in inventory):,} bytes).")
    print("No raw histories, logs, weights, datasets, or rendered charts copied.")


if __name__ == "__main__":
    main()
