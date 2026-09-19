from __future__ import annotations

import pickle
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import copy

from stable_platform_matchings import Optimizer
from stable_platform_matchings.optimization.options import OptimizerParams, SolverOptions
from stable_platform_matchings.domain.instance import Instance
from stable_platform_matchings.graphs.road_graphs import RoadGraph
import stable_platform_matchings.experiments.utils as utils


BASE_SEED = 20260918
VRP_TIME_LIMIT_SECONDS = 900
EPSILON_ELL = 0.5

EPSILON_HS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
EPSILON_HALVES = [EPSILON_HS[:5], EPSILON_HS[5:]]
TOP_N = [1, 2, 3, 4, 5, 6, 7]

def set_epsilons(
    instance: Instance,
    treatment_ids: set,
    epsilon_h: float,
    epsilon_ell: float,
) -> dict[str, float]:
    """Sample an epsilon for every intermediary."""
    epsilons = {}
    for intermediary in instance.intermediaries:
        if intermediary.id in treatment_ids:
            epsilons[intermediary.id] = epsilon_h
        else:
            epsilons[intermediary.id] = epsilon_ell
    return epsilons


def set_het_costs(
    instance: Instance,
    treatment_ids: set,
    margin: float = 1
) -> dict[str, float]:
    """Sample heterogeneous costs for every intermediary."""
    base_het_costs = {
        intermediary.id: float(2.0 * instance.dist_to_mill[intermediary.id])
        for intermediary in instance.intermediaries
    }

    # get min treatment and max control sigmas
    min_treatment_sigma = min(
        het_cost for intermediary_id, het_cost in base_het_costs.items()
        if intermediary_id in treatment_ids
    )
    max_control_sigma = min(
        het_cost for intermediary_id, het_cost in base_het_costs.items()
        if intermediary_id not in treatment_ids
    )

    # scale treatment sigmas
    scale_factor = max(1, (max_control_sigma + margin) / min_treatment_sigma)

    het_costs = {}
    for intermediary_id in base_het_costs:
        if intermediary_id in treatment_ids:
            het_costs[intermediary_id] = base_het_costs[intermediary_id] * scale_factor
        else:
            het_costs[intermediary_id] = base_het_costs[intermediary_id]

    return het_costs


def run_one(
    *,
    job_id: int,
    solver_threads: int,
    treatment_ids: set,
    instance: Instance,
    epsilon_ell: float,
    epsilon_h: float,
) -> dict[str, Any]:
        
    # set random seeding
    print("Setting random seeding...")
    seed_sequence = np.random.SeedSequence(
        [BASE_SEED, job_id]
    )
    optimizer_seed_sequence = (
        seed_sequence.spawn(1)
    )[0]
    optimizer_seed = int(
        optimizer_seed_sequence.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


    # sample inputs
    print("Sampling inputs...")
    epsilons = set_epsilons(
        instance=instance,
        treatment_ids=treatment_ids,
        epsilon_ell=epsilon_ell,
        epsilon_h=epsilon_h,
    )
    het_costs = set_het_costs(
        instance=instance,
        treatment_ids=treatment_ids,
    )

    # initialize optimizer
    print("Initializing optimizer...")
    params = OptimizerParams(
        het_costs=het_costs,
        epsilons=epsilons,
        backend="gurobi",
        vrp_mode="approximate",
        vrp_time_limit_seconds=VRP_TIME_LIMIT_SECONDS,
        threads=solver_threads
    )
    optimizer = Optimizer(
        instance=instance,
        params=params,
    )

    # solve
    print("Solving...")
    options = SolverOptions(
        strategy="exact",
        structured_farmer_payments=False,
        dominance_constraints=False,
        early_stop_threshold=float(1e-4),
        hist_set_method="instance_farmers",
        pay_unmatched=False,
        seed=optimizer_seed,
        stabilize_final_solution=True
    )
    summary = optimizer.solve(options)

    return {
        "metadata": {
            "epsilon_h": epsilon_h,
            "epsilon_ell": epsilon_ell,
            "optimizer_seed": optimizer_seed,
            "seed_sequence_state": seed_sequence.state,
            "optimizer_seed_sequence_state": (
                optimizer_seed_sequence.state
            ),

        },
        "sampled_inputs": {
            "epsilons": epsilons,
            "het_costs": het_costs,
        },
        "summary": summary.return_dict(),
        "treatment_ids": treatment_ids,
    }

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python experiment.py JOB_ID")
    
    job_id = int(sys.argv[1])

    # get data paths
    data_path = Path("data")
    instances_path = data_path / "anon_14_day_instances"
    graph_path = data_path / "graph_0-14960_00_new.pickle"

    # check if data is well-formed
    if not instances_path.is_dir():
        raise FileNotFoundError(f"Instance directory does not exist: {instances_path}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"Graph file does not exist: {graph_path}")

    # load instance paths
    instance_paths = sorted(
        path for path in instances_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".yaml", ".yml"}
        and not path.name.startswith("aggregate")
    )
    if not instance_paths:
        raise FileNotFoundError(f"No YAML instance files found in {instances_path}")

    # make results path
    results_path = Path("results") / "exp_7" / f"job_{job_id}"
    results_path.mkdir(parents=True, exist_ok=True)

    with graph_path.open("rb") as file:
        graph = pickle.load(file)

    solver_threads = utils.get_solver_threads()
    experiment_metadata = {
        "experiment": "exp_7",
        "base_seed": BASE_SEED,
        "job_id": job_id,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "gurobipy_version": utils.package_version("gurobipy"),
        "solver_threads": solver_threads,
        "constants": {
            "vrp_time_limit_seconds": (
                VRP_TIME_LIMIT_SECONDS
            ),
        },
    }

    top_n_idx    = job_id % 7      # 0..6   (innermost, varies fastest)
    job_id      //= 7

    eps_half_idx = job_id % 2      # 0..1
    job_id      //= 2

    instance_idx = job_id          # 0..13  (outermost, varies slowest)

    # load instance
    print("Loading instance...")
    instance_path = instance_paths[instance_idx]
    instance = Instance.from_yaml(instance_path)
    
    print(f"Loaded instance {instance_path}, setting graph...")
    instance.set_graph(RoadGraph(graph))

    top_n = TOP_N[top_n_idx]
    status_quo_sorted = sorted(instance.status_quo_quantities.items(), key=lambda x: x[1])
    treatment_ids = {item[0] for item in status_quo_sorted[:top_n]}

    save_path = results_path / f"job_{job_id}.json.gz"

    job_payload: dict[str, Any] = {
        "experiment_metadata": experiment_metadata,
        "n_runs": 0,
        "runs": [],
        "instance_file": instance_path.name,
        "instance_index": instance_idx,
    }

    for epsilon_h in EPSILON_HALVES[eps_half_idx]:
        run_payload = run_one(
            job_id=job_id,
            solver_threads=solver_threads,
            treatment_ids=treatment_ids,
            instance=instance,
            epsilon_ell=EPSILON_ELL,
            epsilon_h=epsilon_h,
        )

        # format results for correctness
        safe_run_payload = utils.encode_nonfinite(run_payload)
        nonfinite_values = utils.find_nonfinite(safe_run_payload)
        if nonfinite_values:
            print("Found non-finite values:")
            for path, value in nonfinite_values:
                print(f"  {path} = {value!r}")

            raise ValueError(
                f"Payload contains {len(nonfinite_values)} non-finite value(s)"
            )

        # add results to the job payload and save
        job_payload["runs"].append(safe_run_payload)
        job_payload["n_runs"] = len(job_payload["runs"])
        utils.save_json_gz_atomic(
            payload=job_payload,
            save_path=save_path,
        )

        print(f"Saved epsilon_h={epsilon_h} to {save_path}")


if __name__ == "__main__":
    main()