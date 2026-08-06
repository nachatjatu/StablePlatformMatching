from __future__ import annotations

import pickle
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import os

from stable_platform_matchings import Optimizer
from stable_platform_matchings.optimization.options import OptimizerParams, SolverOptions
from stable_platform_matchings.domain.instance import Instance
from stable_platform_matchings.graphs.road_graphs import RoadGraph


N_RUNS = 10
BASE_SEED = 20260806

MIN_QUANTITY = 0.1
MAX_QUANTITY = 9.0
MAX_PERTURB = 0.5

MIN_EPSILON = 0.0
MAX_EPSILON = 6.0

HET_COST_MEAN = 0.0
HET_COST_SD = 100_000.0

VRP_TIME_LIMIT_SECONDS = 1800

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
        perturbation = rng.uniform(-MAX_PERTURB, MAX_PERTURB)
        perturbed = farmer.quantity + perturbation
        rounded_down = np.floor(perturbed * 10.0) / 10.0
        clipped = np.clip(
            rounded_down,
            MIN_QUANTITY,
            MAX_QUANTITY,
        )

        quantities[farmer.id] = float(clipped)

    return quantities


def sample_epsilons(
    instance: Instance,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample an epsilon for every intermediary."""
    return {
        intermediary.id: float(
            rng.uniform(MIN_EPSILON, MAX_EPSILON)
        )
        for intermediary in instance.intermediaries
    }


def sample_het_costs(
    instance: Instance,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample heterogeneous costs for every intermediary."""
    return {
        intermediary.id: float(
            2.0 * instance.dist_to_mill[intermediary.id]
            + rng.normal(HET_COST_MEAN, HET_COST_SD)
        )
        for intermediary in instance.intermediaries
    }


def run_one(
    *,
    job_id: int,
    run_index: int,
    instance_paths: list[Path],
    graph: Any,
    solver_threads: int
) -> dict[str, Any]:
    """
    Run one reproducible experiment and return the InstanceSummary together
    with the information required to reproduce the sampled inputs.
    """
    seed_sequence = np.random.SeedSequence(
        [BASE_SEED, job_id, run_index]
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

    instance_index = int(rng.integers(len(instance_paths)))
    instance_path = instance_paths[instance_index]

    initial_instance = Instance.from_yaml(instance_path)

    quantities = sample_quantities(
        instance=initial_instance,
        rng=rng,
    )

    instance = Instance.from_yaml(
        instance_path,
        force_quantities=quantities,
    )

    instance.set_graph(RoadGraph(graph))

    epsilons = sample_epsilons(
        instance=instance,
        rng=rng,
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

    options = SolverOptions(
        strategy="heuristic_optimized",
        structured_farmer_payments=False,
        dominance_constraints=False,
        early_stop=False,
        aggregate=True,
        pay_unmatched=False,
        seed=optimizer_seed
    )

    optimizer = Optimizer(
        instance=instance,
        params=params,
    )

    summary = optimizer.solve(options)

    return {
        "metadata": {
            "run_index": run_index,
            "optimizer_seed": optimizer_seed,
            "seed_sequence_state": seed_sequence.state,
            "sampling_seed_sequence_state": (
                sampling_seed_sequence.state
            ),
            "optimizer_seed_sequence_state": (
                optimizer_seed_sequence.state
            ),
            "instance_file": instance_path.name,
            "instance_index": instance_index,
        },
        "sampled_inputs": {
            "quantities": quantities,
            "epsilons": epsilons,
            "het_costs": het_costs,
        },
        "summary": summary,
    }


def save_pickle_exclusively(
    payload: dict[str, Any],
    save_path: Path,
) -> None:
    """
    Save without silently overwriting an existing result.

    A temporary file is written first and then atomically moved into place.
    """
    temporary_path = save_path.with_suffix(
        save_path.suffix + ".tmp"
    )

    try:
        with temporary_path.open("xb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

        # refuse to replace an existing completed result.
        if save_path.exists():
            raise FileExistsError(f"Result already exists: {save_path}")

        temporary_path.replace(save_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python experiment.py JOB_ID")
    
    job_id = int(sys.argv[1])

    data_path = Path("data")
    instances_path = data_path / "anon_14_day_instances"
    graph_path = data_path / "graph_0-14960_00_new.pickle"

    results_path = Path("experiments") / "exp_1" / "results" / f"job_{job_id}"

    if not instances_path.is_dir():
        raise FileNotFoundError(f"Instance directory does not exist: {instances_path}")

    if not graph_path.is_file():
        raise FileNotFoundError(f"Graph file does not exist: {graph_path}")

    results_path.mkdir(parents=True, exist_ok=True)

    instance_paths = sorted(
        path for path in instances_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".yaml", ".yml"}
        and not path.name.startswith("aggregate")
    )

    if not instance_paths:
        raise FileNotFoundError(f"No YAML instance files found in {instances_path}")

    with graph_path.open("rb") as file:
        graph = pickle.load(file)

    solver_threads = get_solver_threads()

    experiment_metadata = {
        "experiment": "exp_1",
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
            "max_perturb": MAX_PERTURB,
            "min_epsilon": MIN_EPSILON,
            "max_epsilon": MAX_EPSILON,
            "het_cost_mean": HET_COST_MEAN,
            "het_cost_sd": HET_COST_SD,
            "vrp_time_limit_seconds": (
                VRP_TIME_LIMIT_SECONDS
            ),
        },
    }

    for run_index in range(N_RUNS):
        payload = run_one(
            job_id=job_id,
            run_index=run_index,
            instance_paths=instance_paths,
            graph=graph,
            solver_threads=solver_threads
        )

        payload["experiment_metadata"] = experiment_metadata

        save_path = results_path / f"job_{job_id}_run_{run_index}.pkl"

        save_pickle_exclusively(payload=payload, save_path=save_path,)

        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()