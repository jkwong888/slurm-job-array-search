"""Parquet storage for a fixed California Housing train/validation/test split."""

import hashlib
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


FORMAT_VERSION = 2
DATASET_KIND = "california-housing"
SPLITS = ("train", "validation", "test")
SPLIT_SUFFIXES = {"train": "train", "validation": "validate", "test": "test"}


def split_paths(path):
    """Derive the three Parquet filenames inside one dataset directory."""
    return {split: Path(path) / f"{SPLIT_SUFFIXES[split]}.parquet" for split in SPLITS}


def fingerprint(data):
    """Identify the exact split, row order, features, and targets in a dataset."""
    payload = {key: data[key] for key in
               ("kind", "seed", "feature_names", "train", "validation", "test")}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def prepare_dataset(seed, cache_dir=None):
    """Fetch the public source data and make reproducible 80/10/10 splits."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    housing = fetch_california_housing(data_home=cache_dir)
    features = housing.data
    targets = housing.target
    # Split off test before validation so neither appears in training data.
    # For this 20,640-row dataset, one tenth is exactly 2,064 rows.
    holdout_size = len(features) // 10
    train_val_x, test_x, train_val_y, test_y = train_test_split(
        features, targets, test_size=holdout_size, random_state=seed)
    train_x, val_x, train_y, val_y = train_test_split(
        train_val_x, train_val_y, test_size=holdout_size, random_state=seed)

    def rows(feature_values, target_values):
        """Convert NumPy rows into plain Python values for Parquet storage."""
        return [values + [target] for values, target in
                zip(feature_values.tolist(), target_values.tolist())]

    data = {"format_version": FORMAT_VERSION, "kind": DATASET_KIND,
            "seed": seed, "feature_names": list(housing.feature_names),
            "train": rows(train_x, train_y),
            "validation": rows(val_x, val_y),
            "test": rows(test_x, test_y)}
    data["feature_count"] = len(data["feature_names"])
    data["dataset_id"] = fingerprint(data)
    return data


def write_dataset_split(data, split, path):
    """Store one split with enough metadata to verify the complete dataset."""
    if split not in SPLITS:
        raise ValueError(f"unknown dataset split: {split}")
    feature_names = data["feature_names"]
    rows = data[split]
    # The filename and metadata identify the split, so rows need only values.
    fields = ([pa.field(name, pa.float64()) for name in feature_names] +
              [pa.field("target", pa.float64())])
    metadata = {
        b"format_version": str(data["format_version"]).encode(),
        b"kind": data["kind"].encode(),
        b"seed": str(data["seed"]).encode(),
        b"feature_names": json.dumps(feature_names).encode(),
        b"dataset_id": data["dataset_id"].encode(),
        b"split": split.encode(),
    }
    schema = pa.schema(fields, metadata=metadata)
    columns = []
    columns.extend(pa.array([row[index] for row in rows], type=pa.float64())
                   for index in range(len(feature_names)))
    columns.append(pa.array([row[-1] for row in rows], type=pa.float64()))
    pq.write_table(pa.Table.from_arrays(columns, schema=schema), path)


def load_dataset(path):
    """Read all three files and verify their schema, shared ID, and values."""
    data = None
    for split, split_path in split_paths(path).items():
        try:
            table = pq.read_table(split_path)
        except (OSError, pa.ArrowException) as exc:
            raise ValueError(f"cannot read Parquet split: {split_path}: {exc}") from exc
        metadata = table.schema.metadata or {}
        try:
            current = {
                "format_version": int(metadata[b"format_version"]),
                "kind": metadata[b"kind"].decode("utf-8"),
                "seed": int(metadata[b"seed"]),
                "feature_names": json.loads(metadata[b"feature_names"]),
                "dataset_id": metadata[b"dataset_id"].decode("utf-8"),
            }
            stored_split = metadata[b"split"].decode("utf-8")
        except (KeyError, UnicodeDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid dataset metadata: {split_path}") from exc
        if current["format_version"] != FORMAT_VERSION or current["kind"] != DATASET_KIND:
            raise ValueError(f"unsupported dataset format: {split_path}")
        if stored_split != split:
            raise ValueError(f"split metadata mismatch: {split_path}")
        if data is None:
            data = current
        elif current != {key: data[key] for key in current}:
            raise ValueError(f"dataset metadata mismatch: {split_path}")
        names = current["feature_names"]
        if (not isinstance(names, list) or len(names) != 8 or
                any(not isinstance(name, str) or not name for name in names) or
                len(set(names)) != len(names)):
            raise ValueError(f"invalid feature names: {split_path}")
        expected_schema = pa.schema([pa.field(name, pa.float64()) for name in names] +
                                    [pa.field("target", pa.float64())])
        if not table.schema.equals(expected_schema, check_metadata=False):
            raise ValueError(f"invalid dataset columns: {split_path}")
        columns = table.to_pydict()
        rows = []
        for index in range(table.num_rows):
            row = [columns[name][index] for name in names] + [columns["target"][index]]
            if any(type(value) is not float or not math.isfinite(value) for value in row):
                raise ValueError(f"invalid {split} row: {index}")
            rows.append(row)
        if not rows:
            raise ValueError(f"dataset split is empty: {split_path}")
        data[split] = rows
    data["feature_count"] = len(data["feature_names"])
    if data["dataset_id"] != fingerprint(data):
        raise ValueError(f"dataset fingerprint mismatch: {path}")
    return data
