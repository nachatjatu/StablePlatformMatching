from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import os
import json
import gzip
import math
from pprint import pprint

from stable_platform_matchings import Optimizer, InstanceGenerator
from stable_platform_matchings.optimization.options import OptimizerParams, SolverOptions
from stable_platform_matchings.domain.instance import Instance


BASE_SEED = 20260806

VRP_TIME_LIMIT_SECONDS = 900

DATA_DIR = Path("data")

FARMERS_FULL_CSV_PATH = DATA_DIR / Path("farmers.csv")
FARMERS_14_CSV_PATH = DATA_DIR / Path("farmers_14.csv")
INTERMEDIARIES_CSV_PATH = DATA_DIR / Path("intermediaries.csv")
GRAPH_PKL_PATH = DATA_DIR / Path("graph_0-14960_00_new.pickle")
ALPHA_JSON_PATH = DATA_DIR / Path("precomputed_alpha.json")
SIGMAS_JSON_PATH = DATA_DIR / Path("precomputed_sigmas.json")


EPSILONS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
N_HIST_SETS = [1, 2, 3, 4, 5, 6, 7, 8]
N_CYCLES = 10
N_INTS = 12
CYCLE_LENGTH = 14


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


def set_epsilons(
    instance: Instance,
    epsilon: float,
) -> dict[str, float]:
    """Apply a fixed epsilon to every intermediary."""
    return {
        intermediary.id: float(epsilon)
        for intermediary in instance.intermediaries
    }


def run_one(
    *,
    job_id: int,
    generator: InstanceGenerator,
    n_hist_sets: int,
    solver_threads: int,
    epsilon: float,
    aggregate: bool
) -> dict[str, Any]:
    """
    Run one reproducible experiment and return the InstanceSummary together
    with the information required to reproduce the sampled inputs.
    """
    seed_sequence = np.random.SeedSequence(
        [BASE_SEED, job_id, n_hist_sets]
    )

    sampling_seed_sequence, optimizer_seed_sequence = (
        seed_sequence.spawn(2)
    )

    optimizer_seed = int(
        optimizer_seed_sequence.generate_state(1, dtype=np.uint32)[0]
    )

    instance = generator.gen_instance(
        instance_id=str(job_id),
        day=N_CYCLES * CYCLE_LENGTH - 1,
        n_hist_sets=n_hist_sets,
    )

    pprint(instance.to_snapshot()["intermediaries"])

    epsilons = set_epsilons(
        instance=instance,
        epsilon=epsilon,
    )

    het_costs = {
        intermediary.id: float(2.0 * instance.dist_to_mill[intermediary.id])
        for intermediary in instance.intermediaries
    }

    params = OptimizerParams(
        het_costs=het_costs,
        epsilons=epsilons,
        backend="gurobi",
        vrp_mode="approximate",
        vrp_time_limit_seconds=VRP_TIME_LIMIT_SECONDS,
        threads=solver_threads,
    )

    optimizer = Optimizer(
        instance=instance,
        params=params,
    )

    options = SolverOptions(
        strategy="heuristic_optimized",
        structured_farmer_payments=False,
        dominance_constraints=False,
        early_stop=False,
        aggregate=aggregate,
        pay_unmatched=False,
        seed=optimizer_seed,
    )

    summary = optimizer.solve(options)

    return {
        "schema_version": 1,
        "metadata": {
            "n_hist_sets": n_hist_sets,
            "optimizer_seed": optimizer_seed,
            "seed_sequence_state": seed_sequence.state,
            "sampling_seed_sequence_state": (
                sampling_seed_sequence.state
            ),
            "optimizer_seed_sequence_state": (
                optimizer_seed_sequence.state
            ),
            "aggregate": aggregate
        },
        "sampled_inputs": {
            "epsilons": epsilons,
            "het_costs": het_costs,
        },
        "summary": summary.return_dict(),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python experiment.py JOB_ID")

    job_id = int(sys.argv[1])

    results_path = Path("results") / "exp_5" / f"job_{job_id}"
    results_path.mkdir(parents=True, exist_ok=True)

    job_seed_sequence = np.random.SeedSequence([BASE_SEED, job_id])
    (
        intermediary_seed_sequence,
        calendar_seed_sequence,
        epsilon_seed_sequence,
    ) = job_seed_sequence.spawn(3)

    intermediary_seed = int(
        intermediary_seed_sequence.generate_state(1, dtype=np.uint32)[0]
    )
    calendar_seed = int(
        calendar_seed_sequence.generate_state(1, dtype=np.uint32)[0]
    )

    epsilon_rng = np.random.default_rng(epsilon_seed_sequence)
    epsilon = float(epsilon_rng.choice(EPSILONS))

    generator = InstanceGenerator(
        farmers_full_csv_path=FARMERS_FULL_CSV_PATH,
        farmers_14_csv_path=FARMERS_14_CSV_PATH,
        intermediaries_csv_path=INTERMEDIARIES_CSV_PATH,
        graph_pkl_path=GRAPH_PKL_PATH,
        alpha_json_path=ALPHA_JSON_PATH,
        sigmas_json_path=SIGMAS_JSON_PATH,
    )

    generator.gen_intermediaries(
        n_intermediaries=N_INTS,
        seed=intermediary_seed,
    )

    generator.gen_calendar(
        n_cycles=N_CYCLES,
        cycle_length=CYCLE_LENGTH,
        seed=calendar_seed,
    )

    solver_threads = get_solver_threads()

    experiment_metadata = {
        "experiment": "exp_5",
        "base_seed": BASE_SEED,
        "job_id": job_id,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "stable_platform_matchings_version": package_version(
            "stable-platform-matchings"
        ),
        "gurobipy_version": package_version("gurobipy"),
        "solver_threads": solver_threads,
        "sampled_inputs": {
            "epsilon": epsilon,
            "intermediary_seed": intermediary_seed,
            "calendar_seed": calendar_seed,
        },
        "constants": {
            "vrp_time_limit_seconds": VRP_TIME_LIMIT_SECONDS,
            "epsilons_pool": EPSILONS,
            "n_hist_sets_values": N_HIST_SETS,
            "n_cycles": N_CYCLES,
            "n_intermediaries": N_INTS,
            "cycle_length": CYCLE_LENGTH,
        },
    }

    save_path = results_path / f"job_{job_id}.json.gz"

    job_payload: dict[str, Any] = {
        "schema_version": 1,
        "job_id": job_id,
        "experiment_metadata": experiment_metadata,
        "hist_sets_counts": 0,
        "n_hist_sets": [],
    }

    for n_hist_sets in N_HIST_SETS:
        run_payload = run_one(
            job_id=job_id,
            generator=generator,
            n_hist_sets=n_hist_sets,
            solver_threads=solver_threads,
            epsilon=epsilon,
            aggregate=False
        )

        safe_run_payload = encode_nonfinite(run_payload)

        nonfinite_values = find_nonfinite(safe_run_payload)

        if nonfinite_values:
            print("Found non-finite values:")
            for path, value in nonfinite_values:
                print(f"  {path} = {value!r}")

            raise ValueError(
                f"Payload contains {len(nonfinite_values)} non-finite value(s)"
            )

        job_payload["n_hist_sets"].append(safe_run_payload)
        job_payload["hist_sets_counts"] = len(job_payload["n_hist_sets"])

        save_json_gz_atomic(
            payload=job_payload,
            save_path=save_path,
        )

        print(
            f"Saved n_hist_sets {n_hist_sets} "
            f"({job_payload['hist_sets_counts']}/{len(N_HIST_SETS) + 1}) "
            f"to {save_path}"
        )

    max_n_hist_sets = max(N_HIST_SETS)
    run_payload = run_one(
        job_id=job_id,
        generator=generator,
        n_hist_sets=max_n_hist_sets,
        solver_threads=solver_threads,
        epsilon=epsilon,
        aggregate=True
    )

    safe_run_payload = encode_nonfinite(run_payload)
    
    nonfinite_values = find_nonfinite(safe_run_payload)

    if nonfinite_values:
        print("Found non-finite values:")
        for path, value in nonfinite_values:
            print(f"  {path} = {value!r}")

        raise ValueError(
            f"Payload contains {len(nonfinite_values)} non-finite value(s)"
        )

    job_payload["n_hist_sets"].append(safe_run_payload)
    job_payload["hist_sets_counts"] = len(job_payload["n_hist_sets"])

    save_json_gz_atomic(
        payload=job_payload,
        save_path=save_path,
    )

    print(
        f"Saved n_hist_sets {max_n_hist_sets} "
        f"({job_payload['hist_sets_counts']}/{len(N_HIST_SETS) + 1}) "
        f"to {save_path}"
    )


if __name__ == "__main__":
    main()