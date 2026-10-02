#!/usr/bin/env python3
"""PyTorch regression hyperparameter search on California Housing data."""

import argparse
import glob
import json
import logging
import os
import time
from pathlib import Path

import torch
from torch import nn

from config import configure_logging, load_search_config
from dataset import load_dataset

LOGGER = logging.getLogger("train_search")


def select_device(requested):
    """Choose CPU or CUDA, failing early when CUDA was explicitly requested."""
    if requested == "cuda" and not torch.cuda.is_available():
        details = (f"PyTorch {torch.__version__}, built with CUDA {torch.version.cuda!r}; "
                   f"CUDA device count={torch.cuda.device_count()}; "
                   f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}; "
                   f"NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES')!r}; "
                   f"/dev/nvidia*={glob.glob('/dev/nvidia*')}")
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA GPU. " + details)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def make_model(feature_count, width, device):
    """Build a one-hidden-layer regressor for house value."""
    return nn.Sequential(nn.Linear(feature_count, width), nn.Tanh(), nn.Linear(width, 1)).to(device)


def train(model, features, targets, epochs, batch_size, learning_rate, device):
    """Fit one hyperparameter candidate with shuffled mini-batches."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    started = time.monotonic()
    # Keep Slurm logs useful without writing a line for every mini-batch.
    report_every = max(1, epochs // 8)
    model.train()
    for epoch in range(1, epochs + 1):
        report_progress = epoch == 1 or epoch % report_every == 0 or epoch == epochs
        epoch_loss_sum = torch.zeros((), device=device) if report_progress else None
        order = torch.randperm(len(features), device=device)
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features[indices]), targets[indices])
            if report_progress:
                # Weight each batch by its size so the logged mean covers all rows.
                epoch_loss_sum += loss.detach() * len(indices)
            loss.backward()
            optimizer.step()
        if report_progress:
            LOGGER.info("training progress epoch=%d/%d train_mse=%.4f elapsed_seconds=%.2f",
                        epoch, epochs, (epoch_loss_sum / len(features)).item(),
                        time.monotonic() - started)


@torch.no_grad()
def evaluate(model, features, targets):
    """Measure regression error in the dataset's $100,000 target units."""
    model.eval()
    predictions = model(features)
    errors = predictions - targets
    rmse = errors.square().mean().sqrt()
    mae = errors.abs().mean()
    return {"rmse": rmse.item(), "mae": mae.item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="sweep directory; each trial writes into its task-index subdirectory")
    parser.add_argument("--dataset-path", type=Path, required=True,
                        help="directory containing train.parquet, test.parquet, and validate.parquet")
    parser.add_argument("--search-config", type=Path, required=True,
                        help="JSON file listing every trial and shared training settings")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--trial-index", type=int, help="run one zero-based trial, for a Slurm array")
    args = parser.parse_args()

    config = load_search_config(args.search_config)
    trial_count = len(config["trials"])
    if args.trial_index is not None and not 0 <= args.trial_index < trial_count:
        parser.error(f"--trial-index must be between 0 and {trial_count - 1}")
    LOGGER.info("search plan path=%s id=%s seed=%d epochs=%d trials=%s",
                args.search_config, config["search_id"], config["seed"], config["epochs"],
                json.dumps(config["trials"], separators=(",", ":")))
    if args.trial_index is not None:
        assigned = config["trials"][args.trial_index]
        LOGGER.info("assigned trial index=%d learning_rate=%s batch_size=%d hidden_width=%d",
                    args.trial_index, assigned["learning_rate"], assigned["batch_size"],
                    assigned["hidden_width"])
    dataset = load_dataset(args.dataset_path)
    LOGGER.info("dataset path=%s id=%s features=%d train=%d validation=%d test=%d",
                args.dataset_path, dataset["dataset_id"], dataset["feature_count"],
                len(dataset["train"]), len(dataset["validation"]), len(dataset["test"]))
    device = select_device(args.device)
    LOGGER.info("device=%s%s", device,
                f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = dataset["train"], dataset["validation"]
    # Fit scaling on training rows only; reuse it for validation and prediction.
    train_features = torch.tensor([row[:-1] for row in train_rows], dtype=torch.float32)
    feature_mean = train_features.mean(dim=0)
    feature_scale = train_features.std(dim=0, unbiased=False).clamp_min(1e-12)
    train_features = ((train_features - feature_mean) / feature_scale).to(device)
    train_targets = torch.tensor([[row[-1]] for row in train_rows], dtype=torch.float32, device=device)
    val_features = torch.tensor([row[:-1] for row in val_rows], dtype=torch.float32)
    val_features = ((val_features - feature_mean) / feature_scale).to(device)
    val_targets = torch.tensor([[row[-1]] for row in val_rows], dtype=torch.float32, device=device)
    dataset_id = dataset["dataset_id"]
    indices = (range(trial_count) if args.trial_index is None else [args.trial_index])
    for index in indices:
        start = time.monotonic()
        trial = config["trials"][index]
        learning_rate = trial["learning_rate"]
        batch_size = trial["batch_size"]
        hidden_width = trial["hidden_width"]
        LOGGER.info("starting trial %d/%d: learning_rate=%s batch_size=%d hidden_width=%d",
                    index + 1, trial_count, learning_rate, batch_size, hidden_width)
        # Reset the RNG for a reproducible start and batch order per trial.
        torch.manual_seed(config["seed"])
        model = make_model(dataset["feature_count"], hidden_width, device)
        train(model, train_features, train_targets, config["epochs"], batch_size, learning_rate, device)
        metrics = evaluate(model, val_features, val_targets)
        # The array task ID is the trial index, so all tasks in one sweep
        # publish directly under <sweep>/<task-index>/.
        trial_dir = args.output_dir / str(index)
        trial_dir.mkdir(parents=True, exist_ok=True)
        # Persist preprocessing with the weights so the model can score raw rows.
        checkpoint = {
            "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "feature_names": dataset["feature_names"],
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
        }
        temporary_model = trial_dir / f"model.{os.getpid()}.tmp"
        # Publish complete artifacts atomically for the later summary step.
        torch.save(checkpoint, temporary_model)
        temporary_model.replace(trial_dir / "model.pt")
        result = {"trial_index": index, "seed": config["seed"], "dataset_id": dataset_id,
                  "search_id": config["search_id"], "search_config": str(args.search_config),
                  "dataset_path": str(args.dataset_path), "dataset_seed": dataset["seed"],
                  "feature_count": dataset["feature_count"],
                  "train_samples": len(train_rows), "validation_samples": len(val_rows),
                  "test_samples": len(dataset["test"]),
                  "hidden_width": hidden_width, "learning_rate": learning_rate,
                  "batch_size": batch_size, "epochs": config["epochs"],
                  "device": str(device), "pytorch_version": torch.__version__,
                  "elapsed_seconds": round(time.monotonic() - start, 3),
                  "validation_rmse": metrics["rmse"], "validation_mae": metrics["mae"]}
        temporary_metrics = trial_dir / f"metrics.{os.getpid()}.tmp"
        temporary_metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary_metrics.replace(trial_dir / "metrics.json")
        LOGGER.info("finished trial %d/%d: val_rmse=%.4f val_mae=%.4f saved_to=%s",
                    index + 1, trial_count, metrics["rmse"], metrics["mae"], trial_dir)


if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("training failed: %s", exc)
        raise SystemExit(1) from exc
