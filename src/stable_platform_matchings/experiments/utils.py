import math
from typing import Any
import numpy as np
from pathlib import Path
import json
from importlib.metadata import PackageNotFoundError, version
import gzip
import os

def encode_nonfinite(value: Any) -> Any:
    """Helper function that helps process nonfinite values for downstream file writing"""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
        return value

    if isinstance(value, dict):
        return {str(key): encode_nonfinite(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [encode_nonfinite(item) for item in value]

    if isinstance(value, (set, frozenset)):
        return [encode_nonfinite(item) for item in sorted(value)]

    return value

def find_nonfinite(value: Any, path: str = "payload") -> list[tuple[str, Any]]:
    """Helper function that searches for nonfinite values in an object"""
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
    """
    Convert supported non-standard values into JSON-compatible data.

    Args:
        value (Any): an input object for writing to JSON

    Raises:
        TypeError: object is not JSON serializable

    Returns:
        Any: JSON-compatible representation of `value`.
    """
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

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

def save_json_gz_atomic(payload: dict[str, Any], save_path: Path) -> None:
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
    """Gets the number of available solver threads for Gurobi, defaults to 1."""
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



def snapshot_status_quo_quantity(intermediary: dict[str, Any]) -> float:
    """
    Read an intermediary's recorded status-quo quantity from an instance snapshot.

    Snapshots written before the field was renamed use the ambiguous key
    `status_quo_quantity`; newer ones use `original_status_quo_quantity`. Either is
    accepted so existing result files stay readable.

    Note this is the quantity of the intermediary's *recorded* schedules, averaged
    over them. When a solve derives the status quo differently (see
    `SolverOptions.hist_set_method`), this is not the quantity its stability
    constraints were built against.

    Args:
        intermediary (dict[str, Any]): one entry of `instance_snapshot["intermediaries"]`.

    Raises:
        KeyError: neither key is present.

    Returns:
        float: the recorded status-quo quantity.
    """
    for key in ("original_status_quo_quantity", "status_quo_quantity"):
        if key in intermediary:
            return intermediary[key]

    raise KeyError(
        "snapshot intermediary has neither 'original_status_quo_quantity' nor "
        f"'status_quo_quantity'; keys are {sorted(intermediary)}"
    )


def encode_and_check(run: dict[str, Any], label: str = "payload") -> Any:
    """
    Encode one run for writing and reject anything that would not survive JSON.

    Non-finite Python floats are encoded as strings; anything still non-finite after
    that (e.g. a numpy float32 infinity, which `encode_nonfinite` does not convert)
    is reported and rejected.

    Args:
        run (dict[str, Any]): the run payload.
        label (str): name used when reporting offending values.

    Raises:
        ValueError: the payload contains non-finite values.

    Returns:
        Any: the encoded payload, ready to be written.
    """
    encoded = encode_nonfinite(run)

    nonfinite_values = find_nonfinite(encoded)
    if nonfinite_values:
        print(f"Found non-finite values in {label}:")
        for path, value in nonfinite_values:
            print(f"  {path} = {value!r}")

        raise ValueError(
            f"{label} contains {len(nonfinite_values)} non-finite value(s)"
        )

    return encoded


class JobPayload:
    """
    Accumulates one SLURM job's runs and writes them atomically.

    Replaces the encode / check / append / count / save sequence that each experiment
    script previously repeated inline. Runs are validated as they are added, so a
    non-finite value is reported against the run that produced it rather than after
    the whole job has finished.

    The written document is ``{**job_fields, "n_runs": int, "runs": [...]}``.
    """

    def __init__(self, **job_fields: Any) -> None:
        """
        Args:
            **job_fields: job-level fields to record alongside the runs, e.g.
                `experiment_metadata`, the instance file, or the swept values that
                identify this job's cell of the grid.
        """
        self._job_fields = job_fields
        self._runs: list[Any] = []

    @property
    def n_runs(self) -> int:
        """Number of runs added so far."""
        return len(self._runs)

    def add_run(self, run: dict[str, Any]) -> None:
        """
        Encode, validate and store one run.

        Args:
            run (dict[str, Any]): the run payload.

        Raises:
            ValueError: the run contains non-finite values, which would not survive
                a round trip through JSON.
        """
        self._runs.append(
            encode_and_check(run, label=f"run {len(self._runs)}")
        )

    def save(self, save_path: Path) -> None:
        """
        Atomically write the job document.

        Args:
            save_path (Path): destination, which must end in `.json.gz`.
        """
        save_json_gz_atomic(
            payload={
                **self._job_fields,
                "n_runs": len(self._runs),
                "runs": self._runs,
            },
            save_path=save_path,
        )
