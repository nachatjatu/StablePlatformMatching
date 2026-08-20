import math
from typing import Any
import numpy as np
from pathlib import Path
import json
from importlib.metadata import PackageNotFoundError, version
import gzip
import os

def encode_nonfinite(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
        return value

    if isinstance(value, dict):
        return {
            str(key): encode_nonfinite(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [encode_nonfinite(item) for item in value]

    if isinstance(value, (set, frozenset)):
        return [
            encode_nonfinite(item)
            for item in sorted(value)
        ]

    return value

def find_nonfinite(
    value: Any,
    path: str = "payload",
) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            found.append((path, value))
        return found

    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(
                find_nonfinite(
                    item,
                    f"{path}[{key!r}]",
                )
            )
        return found

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(
                find_nonfinite(
                    item,
                    f"{path}[{index}]",
                )
            )
        return found

    if isinstance(value, (set, frozenset)):
        for index, item in enumerate(value):
            found.extend(
                find_nonfinite(
                    item,
                    f"{path}[set_item_{index}]",
                )
            )

    return found

def json_default(value: Any) -> Any:
    """Convert supported non-standard values into JSON-compatible data."""
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (set, frozenset)):
        return sorted(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return list(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )

def save_json_gz_atomic(
    payload: dict[str, Any],
    save_path: Path,
) -> None:
    """
    Atomically write a gzip-compressed JSON checkpoint.

    Existing files are replaced only after the new file has been written
    successfully.
    """
    if not save_path.name.endswith(".json.gz"):
        raise ValueError("save_path must end with .json.gz")

    temporary_path = save_path.with_name(
        save_path.name + ".tmp"
    )

    try:
        with gzip.open(
            temporary_path,
            "wt",
            encoding="utf-8",
            compresslevel=6,
        ) as file:
            json.dump(
                payload,
                file,
                default=json_default,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )

        temporary_path.replace(save_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def get_solver_threads() -> int:
    for var in ("SLURM_CPUS_PER_TASK", "GUROBI_THREADS"):
        value = os.environ.get(var)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return 1


def package_version(package: str) -> str | None:
    """Return an installed package's version, if available."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None

