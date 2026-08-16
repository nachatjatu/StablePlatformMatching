from __future__ import annotations

import pickle
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import os
import time
import json
import gzip
import math

from stable_platform_matchings import Optimizer
from stable_platform_matchings.optimization.options import OptimizerParams, SolverOptions
from stable_platform_matchings.domain.instance import Instance
from stable_platform_matchings.graphs.road_graphs import RoadGraph


N_RUNS = 1
BASE_SEED = 20260806

MIN_QUANTITY = 0.5
MAX_QUANTITY = 2.8

MIN_HET_COST = 200_000.0
MAX_HET_COST = 1_200_000.0

HIGH_PROB = 0.28

VRP_TIME_LIMIT_SECONDS = 10 * 60 # 10 minutes

EPSILON = 2

FRUIT_THRESH = 3.5

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


def sample_quantities(
    instance: Instance,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Perturb, round down, and clip each farmer's quantity."""
    quantities = {}

    for farmer in instance.farmers:
        random_quantity = rng.uniform(MIN_QUANTITY, MAX_QUANTITY)
        rounded_down = np.floor(random_quantity * 10.0) / 10.0
        clipped = np.clip(
            rounded_down,
            MIN_QUANTITY,
            MAX_QUANTITY,
        )

        quantities[farmer.id] = float(clipped)

    return quantities


def set_epsilons(
    instance: Instance,
) -> dict[str, float]:
    """Sample an epsilon for every intermediary."""
    return {
        intermediary.id: float(EPSILON)
        for intermediary in instance.intermediaries
    }


def sample_het_costs(
    instance: Instance,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample heterogeneous costs for every intermediary."""
    return {
        intermediary.id: float(
            rng.uniform(MIN_HET_COST, MAX_HET_COST)
        )
        for intermediary in instance.intermediaries
    }

def reset_relationships(
    instance: Instance,
    rng: np.random.Generator
):
    hist_matching = {
        intermediary.id: set() 
        for intermediary in instance.intermediaries
    }
    farmer_quants = {
        farmer.id: farmer.quantity 
        for farmer in instance.farmers
    }
    unmatched_farmers = set(farmer_quants.keys())

    high_int = {
        intermediary.id for intermediary in instance.intermediaries 
        if rng.random() < HIGH_PROB
    }
    low_int = {
        intermediary.id for intermediary in instance.intermediaries 
        if intermediary.id not in high_int
    }

    intermediary_ids = [intermediary.id for intermediary in instance.intermediaries]

    while len(unmatched_farmers) > 0:
        sampled_farmer = str(rng.choice(sorted(unmatched_farmers)))
        sampled_int = str(rng.choice(sorted(intermediary_ids)))
        
        sum_fruit = sum(
            farmer_quants[f_id] for f_id in hist_matching[sampled_int]
        ) + farmer_quants[sampled_farmer]

        if (sampled_int in high_int) or (sampled_int in low_int and sum_fruit <= FRUIT_THRESH):
            hist_matching[sampled_int].add(sampled_farmer)
            unmatched_farmers.remove(sampled_farmer)


    for intermediary in instance.intermediaries:
        intermediary.hist_sets = [frozenset(hist_matching[intermediary.id])]

    return hist_matching


def sample_locs(
    instance: Instance,
    rng: np.random.Generator,
    G
):  
    avail_locations = {
        str(node):(data['lat'], data['lon']) 
        for node, data in G.nodes(data=True) 
        if 'lat' in data and 'lon' in data
    }
    farmer_locations = {}
    for farmer in instance.farmers:
        node = rng.choice(list(avail_locations.keys()))
        farmer_locations[farmer.id] = node
        farmer.location = avail_locations[node]


def run_one(
    *,
    job_id: int,
    instance_path: Path,
    graph: Any,
    solver_threads: int,
    process_start
) -> dict[str, Any]:
    """
    Run one reproducible experiment and return the InstanceSummary together
    with the information required to reproduce the sampled inputs.
    """

    print("Building instance...")
    seed_sequence = np.random.SeedSequence(
        [BASE_SEED, job_id]
    )

    sampling_seed_sequence, optimizer_seed_sequence = (
        seed_sequence.spawn(2)
    )

    rng = np.random.default_rng(sampling_seed_sequence)

    optimizer_seed = int(
        optimizer_seed_sequence.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )

    initial_instance = Instance.from_yaml(instance_path)

    quantities = sample_quantities(
        instance=initial_instance,
        rng=rng,
    )
    print("Loading instance...")
    instance = Instance.from_yaml(
        instance_path,
        force_quantities=quantities,
    )

    print("Sampling locations...")
    sample_locs(instance, rng, graph)

    print("Setting graph...")
    instance.set_graph(RoadGraph(graph))

    print("Setting relationships...")
    reset_relationships(instance, rng)

    epsilons = set_epsilons(
        instance=instance,
    )

    het_costs = sample_het_costs(
        instance=instance,
        rng=rng,
    )

    params = OptimizerParams(
        het_costs=het_costs,
        epsilons=epsilons,
        backend="gurobi",
        vrp_mode="approximate",
        vrp_time_limit_seconds=VRP_TIME_LIMIT_SECONDS,
        threads=solver_threads
    )

    print("Initializing optimizer...")
    optimizer = Optimizer(
        instance=instance,
        params=params,
    )

    options = SolverOptions(
        strategy="heuristic_optimized",
        structured_farmer_payments=False,
        dominance_constraints=False,
        early_stop=True,
        aggregate=True,
        pay_unmatched=False,
        seed=optimizer_seed,
    )

    summary_early_stop = optimizer.solve(options)


    return {
        "schema_version": 1,
        "metadata": {
            "optimizer_seed": optimizer_seed,
            "seed_sequence_state": seed_sequence.state,
            "sampling_seed_sequence_state": (
                sampling_seed_sequence.state
            ),
            "optimizer_seed_sequence_state": (
                optimizer_seed_sequence.state
            ),
            "instance_file": instance_path.name,
        },
        "sampled_inputs": {
            "quantities": quantities,
            "epsilons": epsilons,
            "het_costs": het_costs,
        },
        "summary_early_stop": summary_early_stop.return_dict(),
    }

def main() -> None:

    process_start = time.time()

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python experiment.py JOB_ID")
    
    job_id = int(sys.argv[1])

    data_path = Path("data")
    instances_path = data_path / "anon_14_day_instances"
    graph_path = data_path / "graph_0-14960_00_new.pickle"

    results_path = Path("results") / "exp_6_heuristic" / f"job_{job_id}"

    if not instances_path.is_dir():
        raise FileNotFoundError(f"Instance directory does not exist: {instances_path}")

    if not graph_path.is_file():
        raise FileNotFoundError(f"Graph file does not exist: {graph_path}")

    results_path.mkdir(parents=True, exist_ok=True)

    instance_paths = sorted(
        path for path in instances_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".yaml", ".yml"}
        and path.name.startswith("aggregate_instance_")
    )

    if not instance_paths:
        raise FileNotFoundError(f"No YAML instance files found in {instances_path}")

    with graph_path.open("rb") as file:
        graph = pickle.load(file)

    solver_threads = get_solver_threads()

    experiment_metadata = {
        "experiment": "exp_6_heuristic",
        "base_seed": BASE_SEED,
        "job_id": job_id,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "stable_platform_matchings_version": package_version(
            "stable-platform-matchings"
        ),
        "gurobipy_version": package_version("gurobipy"),
        "solver_threads": solver_threads,
        "constants": {
            "min_quantity": MIN_QUANTITY,
            "max_quantity": MAX_QUANTITY,
            "min_het_cost": MIN_HET_COST,
            "max_het_cost": MAX_HET_COST,
            "vrp_time_limit_seconds": (
                VRP_TIME_LIMIT_SECONDS
            ),
        },
    }
    instance_path = instance_paths[job_id%14]

    save_path = results_path / f"job_{job_id}.json.gz"

    job_payload: dict[str, Any] = {
        "schema_version": 1,
        "job_id": job_id,
        "experiment_metadata": experiment_metadata,
        "n_runs": 0,
        "runs": [],
    }

    run_payload = run_one(
        job_id=job_id,
        instance_path=instance_path,
        graph=graph,
        solver_threads=solver_threads,
        process_start=process_start
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

    job_payload["runs"].append(safe_run_payload)
    job_payload["n_runs"] = len(job_payload["runs"])


    save_json_gz_atomic(
        payload=job_payload,
        save_path=save_path,
    )

    print(
        f"Saved run"
        f"to {save_path}"
    )


if __name__ == "__main__":
    main()