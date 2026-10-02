#!/usr/bin/env python3
"""Summarize completed training trials and select the best validation result."""

import argparse
import json
import logging
from pathlib import Path

from config import configure_logging, load_search_config

LOGGER = logging.getLogger("summarize")


def summarize(output_dir, config):
    """Verify every trial and select the checkpoint with the lowest RMSE."""
    results = []
    model_paths = {}
    for index, trial in enumerate(config["trials"]):
        # New sweeps use task-index directories. Older sweeps used a Slurm
        # job-id/task-id directory with a nested trial directory.
        trial_dir = output_dir / str(index)
        if not trial_dir.is_dir():
            legacy_dirs = sorted(path for path in output_dir.rglob(f"trial-{index:03d}")
                                 if path.is_dir())
            if len(legacy_dirs) != 1:
                raise ValueError(f"expected one completed trial {index} under {output_dir}, "
                                 f"found {len(legacy_dirs)}")
            trial_dir = legacy_dirs[0]
        metrics_path = trial_dir / "metrics.json"
        model_path = trial_dir / "model.pt"
        if not metrics_path.is_file() or not model_path.is_file():
            raise ValueError(f"missing completed trial {index}: {trial_dir}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics["trial_index"] != index:
            raise ValueError(f"trial index mismatch in {metrics_path}")
        if (metrics["search_id"] != config["search_id"] or
                metrics["seed"] != config["seed"] or
                metrics["epochs"] != config["epochs"] or
                any(metrics[name] != trial[name] for name in trial)):
            raise ValueError(f"search configuration mismatch in {metrics_path}")
        relative_model_path = str(model_path.relative_to(output_dir))
        model_paths[str(index)] = relative_model_path
        # Each summary row names its model relative to the sweep directory.
        results.append({**metrics, "model_path": relative_model_path})
        LOGGER.info("trial=%d model_path=%s val_rmse=%.4f", index,
                    relative_model_path, metrics["validation_rmse"])
    # Reject a sweep assembled from different datasets or training settings.
    setups = {(item["dataset_id"], item["feature_count"], item["search_id"],
               item["train_samples"], item["validation_samples"], item["test_samples"])
              for item in results}
    if len(setups) != 1:
        raise ValueError("trials used different datasets or search plans")
    best = min(results, key=lambda item: item["validation_rmse"])
    with (output_dir / "trials.jsonl").open("w", encoding="utf-8") as file:
        for item in results:
            file.write(json.dumps(item) + "\n")
    summary = {"trial_count": len(results), "dataset_id": best["dataset_id"],
               "search_id": config["search_id"],
               "feature_count": best["feature_count"],
               "hidden_width": best["hidden_width"], "best_trial": best,
               "model_paths": model_paths,
               "best_checkpoint": best["model_path"]}
    (output_dir / "best.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("best trial=%d val_rmse=%.4f val_mae=%.4f checkpoint=%s",
                best["trial_index"], best["validation_rmse"], best["validation_mae"],
                summary["best_checkpoint"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-config", type=Path, required=True,
                        help="JSON search plan used by the training jobs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory containing all completed trial results")
    args = parser.parse_args()
    config = load_search_config(args.search_config)
    LOGGER.info("summarizing %d trials from %s", len(config["trials"]), args.output_dir)
    summarize(args.output_dir, config)


if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except (OSError, ValueError) as exc:
        LOGGER.error("summarization failed: %s", exc)
        raise SystemExit(1) from exc
