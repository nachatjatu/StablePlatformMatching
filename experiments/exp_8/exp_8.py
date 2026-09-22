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

EPSILON_ELLS = [0]
TOP_N = [1, 2, 3]
HIST_SET_METHOD = "instance_farmers"
EPSILON_H_GROUPS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
]

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
    margin: float = 100
) -> dict[str, float]:
    """Sample heterogeneous costs for every intermediary."""
    base_het_costs = {
        intermediary.id: float(2.0 * instance.dist_to_mill[intermediary.id])
        for intermediary in instance.intermediaries
    }

    # get min treatment and max control sigmas
    max_control_sigma = max(
        het_cost for intermediary_id, het_cost in base_het_costs.items()
        if intermediary_id not in treatment_ids
    )

    het_costs = {}
    for intermediary_id in base_het_costs:
        if intermediary_id in treatment_ids:
            het_costs[intermediary_id] = max(base_het_costs[intermediary_id], max_control_sigma + margin)
        else:
            het_costs[intermediary_id] = base_het_costs[intermediary_id]

    return het_costs


def run_one(
    *,
    job_id: int,
    optimizer: Optimizer,
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
    # solve
    print("Solving...")
    options = SolverOptions(
        strategy="exact",
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
    # epsilon_ell and treatment_ids are job-level and recorded once there. epsilon_h
    # stays because it varies across the runs within a job.
    return {
        "metadata": {
            "epsilon_h": epsilon_h,
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
    results_path = Path("results") / "exp_8" / f"job_{job_id}"
    results_path.mkdir(parents=True, exist_ok=True)

    with graph_path.open("rb") as file:
        graph = pickle.load(file)

    solver_threads = utils.get_solver_threads()
    experiment_metadata = {
        "experiment": "exp_8",
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

    # decode job_id into (replicate, instance, epsilon_ell, epsilon_h group,
    # top_n) without clobbering job_id; job_ids beyond one full sweep of the
    # grid wrap around and become additional replicates of each condition.
    if job_id < 0:
        raise SystemExit(f"JOB_ID must be non-negative, got {job_id}")

    remainder = job_id

    top_n_idx = remainder % len(TOP_N)          # innermost, varies fastest
    remainder //= len(TOP_N)

    eps_h_group_idx = remainder % len(EPSILON_H_GROUPS)
    remainder //= len(EPSILON_H_GROUPS)

    eps_ell_idx = remainder % len(EPSILON_ELLS)
    remainder //= len(EPSILON_ELLS)

    instance_idx = remainder % len(instance_paths)
    remainder //= len(instance_paths)

    replicate_idx = remainder                   # outermost, varies slowest

    # load instance
    print("Loading instance...")
    instance_path = instance_paths[instance_idx]
    instance = Instance.from_yaml(instance_path)

    print(f"Loaded instance {instance_path}, setting graph...")
    instance.set_graph(RoadGraph(graph))

    top_n = TOP_N[top_n_idx]
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

    epsilon_hs = EPSILON_H_GROUPS[eps_h_group_idx]

    job_payload = utils.JobPayload(
        experiment_metadata=experiment_metadata,
        instance_file=instance_path.name,
        instance_index=instance_idx,
        top_n=top_n,
        treatment_ids=treatment_ids,
        epsilon_ell=EPSILON_ELLS[eps_ell_idx],
        epsilon_h_group_index=eps_h_group_idx,
        epsilon_h_group=epsilon_hs,
        replicate_index=replicate_idx,
    )

    epsilon_ell = EPSILON_ELLS[eps_ell_idx]

    # het_costs depend only on treatment_ids, so one Optimizer serves the whole
    # epsilon_h group: its VRP routing costs are computed once rather than per run.
    het_costs = clip_het_costs(
        instance=instance,
        treatment_ids=treatment_ids,
    )
    print("Initializing optimizer...")
    optimizer = Optimizer(
        instance=instance,
        params=OptimizerParams(
            het_costs=het_costs,
            backend="gurobi",
            vrp_mode="approximate",
            vrp_time_limit_seconds=VRP_TIME_LIMIT_SECONDS,
            threads=solver_threads,
        ),
    )

    for epsilon_h in epsilon_hs:
        run_payload = run_one(
            job_id=job_id,
            optimizer=optimizer,
            treatment_ids=treatment_ids,
            instance=instance,
            epsilon_ell=epsilon_ell,
            epsilon_h=float(epsilon_h),
        )

        # save after each run so a timeout keeps the completed ones
        job_payload.add_run(run_payload)
        job_payload.save(save_path)

        print(f"Saved epsilon_h={epsilon_h} to {save_path}")


if __name__ == "__main__":
    main()
