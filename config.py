"""Shared search-plan validation and command-line logging setup."""

import hashlib
import json
import logging
import math
import sys


def configure_logging():
    """Write timestamped INFO and ERROR messages to the job's stdout log."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stdout,
    )


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise ValueError("must be a positive integer")
    return number


def load_search_config(path):
    """Return a validated plan and its content ID for sweep consistency."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"format_version", "seed", "epochs", "trials"}:
        raise ValueError(f"invalid search configuration: {path}")
    if type(config["format_version"]) is not int or config["format_version"] != 1:
        raise ValueError("unsupported search configuration version")
    if type(config["seed"]) is not int:
        raise ValueError("search seed must be an integer")
    if type(config["epochs"]) is not int or config["epochs"] <= 0:
        raise ValueError("search epochs must be a positive integer")
    trials = config["trials"]
    if not isinstance(trials, list) or not trials:
        raise ValueError("search must contain at least one trial")
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict) or set(trial) != {"learning_rate", "batch_size", "hidden_width"}:
            raise ValueError(f"invalid trial {index}")
        rate = trial["learning_rate"]
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"trial {index} learning_rate must be positive and finite")
        for name in ("batch_size", "hidden_width"):
            if type(trial[name]) is not int or trial[name] <= 0:
                raise ValueError(f"trial {index} {name} must be a positive integer")
    # All tasks record this ID; the summary rejects results from different plans.
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    config["search_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return config
