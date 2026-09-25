from __future__ import annotations

import pickle
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stable_platform_matchings import Optimizer
from stable_platform_matchings.optimization.options import OptimizerParams, SolverOptions
from stable_platform_matchings.domain.hist_sets import (
    status_quo_quantities_for_method,
)
from stable_platform_matchings.domain.instance import Instance
from stable_platform_matchings.graphs.road_graphs import RoadGraph
import stable_platform_matchings.experiments.utils as utils


BASE_SEED = 20260918
VRP_TIME_LIMIT_SECONDS = 900

EPSILON_ELL = 0.5
EPSILON_HS = [0, 1, 2, 3, 4, 5, 6, 7]
TOP_N = 3
HIST_SET_METHOD = "instance_farmers"

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


def clip_het_costs(
    instance: Instance,
    treatment_ids: set,
    margin: float = 10000
) -> dict[str, float]:
    """Sample heterogeneous costs for every intermediary."""
    base_het_costs = {
        intermediary.id: float(2.0 * instance.dist_to_mill[intermediary.id])
        for intermediary in instance.intermediaries
    }

    # get max control sigma; every treated intermediary is clipped above this
    max_control_sigma = max(
        het_cost for intermediary_id, het_cost in base_het_costs.items()
        if intermediary_id not in treatment_ids
    )

    # Give each treated intermediary its own floor rather than a shared one. A
    # single shared floor collapses every treated intermediary whose base cost
    # falls below it onto the identical value, making them interchangeable.
    # Spreading the floors by rank keeps the design intent (all treated strictly
    # above all controls, relative order among treated preserved) while leaving
    # the costs distinct.
    treated_by_base_cost = sorted(
        treatment_ids, key=lambda intermediary_id: base_het_costs[intermediary_id]
    )
    treated_floor = {
        intermediary_id: max_control_sigma + margin * (rank + 1)
        for rank, intermediary_id in enumerate(treated_by_base_cost)
    }

    het_costs = {}
    for intermediary_id in base_het_costs:
        if intermediary_id in treatment_ids:
            het_costs[intermediary_id] = max(
                base_het_costs[intermediary_id], treated_floor[intermediary_id]
            )
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
    het_costs = clip_het_costs(
        instance=instance,
        treatment_ids=treatment_ids,
    )

    # initialize optimizer
    print("Initializing optimizer...")
    params = OptimizerParams(
        het_costs=het_costs,
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
        strategy="lagrangian_bnp",
        structured_farmer_payments=False,
        dominance_constraints=False,
        early_stop_threshold=float(1e-4),
        hist_set_method=HIST_SET_METHOD,
        pay_unmatched=False,
        seed=optimizer_seed,
        stabilize_final_solution=True
    )
    summary = optimizer.solve(options, epsilons=epsilons)

    # epsilons and het_costs are recorded by the summary itself (summary.params);
    # epsilon_h/epsilon_ell and treatment_ids are job-level and recorded once there.
    return {
        "metadata": {
            "optimizer_seed": optimizer_seed,
            "seed_sequence_state": seed_sequence.state,
            "optimizer_seed_sequence_state": (
                optimizer_seed_sequence.state
            ),
        },
        "summary": summary.return_dict(),
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

    # decode job_id into (replicate, instance, epsilon_h) without clobbering
    # job_id; one job_id selects exactly one epsilon_h and one instance. job_ids
    # beyond one full sweep of the grid wrap around and become additional
    # replicates of each condition.
    if job_id < 0:
        raise SystemExit(f"JOB_ID must be non-negative, got {job_id}")

    remainder = job_id

    eps_h_idx = remainder % len(EPSILON_HS)     # innermost, varies fastest
    remainder //= len(EPSILON_HS)

    instance_idx = remainder % len(instance_paths)

    epsilon_h = float(EPSILON_HS[eps_h_idx])
    epsilon_ell = float(EPSILON_ELL)

    # load instance
    print("Loading instance...")
    instance_path = instance_paths[instance_idx]
    instance = Instance.from_yaml(instance_path)
    
    print(f"Loaded instance {instance_path}, setting graph...")
    instance.set_graph(RoadGraph(graph))

    top_n = TOP_N
    # Rank by the status quo the stability constraints actually enforce under
    # HIST_SET_METHOD, not by instance.original_status_quo_quantities (which averages
    # the recorded routes and so understates multi-route intermediaries). Ties are
    # broken by intermediary ID so the treatment group is reproducible.
    status_quo_quantities = status_quo_quantities_for_method(instance, HIST_SET_METHOD)
    status_quo_sorted = sorted(
        status_quo_quantities.items(),
        key=lambda item: (-item[1], item[0]),
    )
    treatment_ids = {item[0] for item in status_quo_sorted[:top_n]}

    save_path = results_path / f"job_{job_id}.json.gz"

    job_payload = utils.JobPayload(
        experiment_metadata=experiment_metadata,
        instance_file=instance_path.name,
        instance_index=instance_idx,
        top_n=top_n,
        treatment_ids=treatment_ids,
        epsilon_ell=epsilon_ell,
        epsilon_h=epsilon_h,
        epsilon_h_index=eps_h_idx,
    )

    run_payload = run_one(
        job_id=job_id,
        solver_threads=solver_threads,
        treatment_ids=treatment_ids,
        instance=instance,
        epsilon_ell=epsilon_ell,
        epsilon_h=epsilon_h,
    )

    job_payload.add_run(run_payload)
    job_payload.save(save_path)

    print(f"Saved epsilon_h={epsilon_h} to {save_path}")


if __name__ == "__main__":
    main()