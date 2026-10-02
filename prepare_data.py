#!/usr/bin/env python3
"""Download California Housing and save fixed train/validation/test splits."""

import argparse
import logging
import os
import tempfile
from pathlib import Path

from config import configure_logging
from dataset import prepare_dataset, split_paths, write_dataset_split

LOGGER = logging.getLogger("prepare_data")


def main():
    """Fetch once and publish three Parquet files in a chosen directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True,
                        help="dataset directory containing train.parquet, test.parquet, and validate.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path,
                        help="optional scikit-learn download cache directory")
    args = parser.parse_args()
    paths = split_paths(args.output_path)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise ValueError(f"dataset split already exists: {existing[0]}; choose a new path")

    LOGGER.info("preparing California Housing output=%s seed=%d cache_dir=%s",
                args.output_path, args.seed, args.cache_dir)
    data = prepare_dataset(args.seed, args.cache_dir)
    args.output_path.mkdir(parents=True, exist_ok=True)
    temporary_paths = {}
    published_paths = []
    try:
        # Stage every split first. Hard links publish without overwriting an
        # existing file; a failed publication removes only links we created.
        for split, path in paths.items():
            with tempfile.NamedTemporaryFile("wb", dir=path.parent,
                                             prefix=f".{path.name}.", suffix=".tmp",
                                             delete=False) as file:
                temporary_paths[split] = Path(file.name)
            write_dataset_split(data, split, temporary_paths[split])
            with temporary_paths[split].open("rb") as file:
                os.fsync(file.fileno())
        for split, path in paths.items():
            os.link(temporary_paths[split], path)
            published_paths.append(path)
    except FileExistsError as exc:
        for path in published_paths:
            path.unlink(missing_ok=True)
        raise ValueError(f"dataset split already exists; choose a new dataset directory: {args.output_path}") from exc
    except Exception:
        for path in published_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
    LOGGER.info("dataset ready path=%s id=%s features=%d train=%d validation=%d test=%d",
                args.output_path, data["dataset_id"], data["feature_count"],
                len(data["train"]), len(data["validation"]), len(data["test"]))
    for split, path in paths.items():
        LOGGER.info("split=%s rows=%d path=%s", split, len(data[split]), path)


if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except (OSError, ValueError) as exc:
        LOGGER.error("dataset preparation failed: %s", exc)
        raise SystemExit(1) from exc
