from __future__ import annotations

import pickle
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stable_platform_matchings import Optimizer
from stable_platform_matchings.optimization.options import OptimizerParams, SolverOptions
from stable_platform_matchings.domain.instance import Instance
from stable_platform_matchings.graphs.road_graphs import RoadGraph
import stable_platform_matchings.experiments.utils as utils

N_RUNS = 10
BASE_SEED = 20260806
MIN_QUANTITY = 0.1
MAX_QUANTITY = 9.0
MAX_PERTURB = 0.5
HET_COST_MEAN = 0.0
HET_COST_SD = 100_000.0
VRP_TIME_LIMIT_SECONDS = 900
EPSILONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def perturb_quantities(
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
    epsilon = rng.choice(EPSILONS)
    return {
        intermediary.id: float(epsilon)
        for intermediary in instance.intermediaries
    }


def perturb_het_costs(
    instance: Instance,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample heterogeneous costs for every intermediary."""
    return {
        intermediary.id: float(
            4.0 * instance.dist_to_mill[intermediary.id]
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
        
    # set random seeding
    print("Setting random seeding...")
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

    # load instance and perturb quantities
    print("Loading instance...")
    instance_index = int(rng.integers(len(instance_paths)))
    instance_path = instance_paths[instance_index]
    initial_instance = Instance.from_yaml(instance_path)
    quantities = perturb_quantities(
        instance=initial_instance,
        rng=rng,
    )
    instance = Instance.from_yaml(
        instance_path,
        force_quantities=quantities,
    )
    print(f"Loaded instance {instance_path}, setting graph...")
    instance.set_graph(RoadGraph(graph))

    # sample inputs
    print("Sampling inputs...")
    epsilons = sample_epsilons(
        instance=instance,
        rng=rng,
    )
    het_costs = perturb_het_costs(
        instance=instance,
        rng=rng,
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
    print("Solving vanilla no pay...")
    vanilla_no_pay_options = SolverOptions(
        strategy="paper_bnb",
        structured_farmer_payments=False,
        dominance_constraints=False,
        early_stop_threshold=float(1e-4),
        hist_set_method="instance_farmers",
        pay_unmatched=False,
        seed=optimizer_seed,
        stabilize_final_solution=True
    )
    summary_vanilla_no_pay = optimizer.solve(
        vanilla_no_pay_options,
        epsilons=epsilons,
    )

    print("Solving vanilla pay...")
    vanilla_pay_options = SolverOptions(
        strategy="paper_bnb",
        structured_farmer_payments=False,
        dominance_constraints=False,
        early_stop_threshold=float(1e-4),
        hist_set_method="instance_farmers",
        pay_unmatched=True,
        seed=optimizer_seed,
        stabilize_final_solution=True
    )
    summary_vanilla_pay = optimizer.solve(
        vanilla_pay_options,
        epsilons=epsilons,
    )

    print("Solving structured no pay...")
    structured_no_pay_options = SolverOptions(
        strategy="paper_bnb",
        structured_farmer_payments=True,
        dominance_constraints=False,
        early_stop_threshold=float(1e-4),
        hist_set_method="instance_farmers",
        pay_unmatched=False,
        stabilize_final_solution=True
    )
    summary_structured_no_pay = optimizer.solve(
        structured_no_pay_options,
        epsilons=epsilons,
    )

    print("Solving structured pay...")
    structured_pay_options = SolverOptions(
        strategy="paper_bnb",
        structured_farmer_payments=True,
        dominance_constraints=False,
        early_stop_threshold=float(1e-4),
        hist_set_method="instance_farmers",
        pay_unmatched=True,
        seed=optimizer_seed,
        stabilize_final_solution=True
    )
    summary_structured_pay = optimizer.solve(
        structured_pay_options,
        epsilons=epsilons,
    )

    print("Solving dominance no pay...")
    dominance_no_pay_options = SolverOptions(
        strategy="paper_bnb",
        structured_farmer_payments=False,
        dominance_constraints=True,
        early_stop_threshold=float(1e-4),
        hist_set_method="instance_farmers",
        pay_unmatched=False,
        seed=optimizer_seed,
        stabilize_final_solution=True
    )
    summary_dominance_no_pay = optimizer.solve(
        dominance_no_pay_options,
        epsilons=epsilons,
    )

    print("Solving dominance pay...")
    dominance_pay_options = SolverOptions(
        strategy="paper_bnb",
        structured_farmer_payments=False,
        dominance_constraints=True,
        early_stop_threshold=0.0,
        hist_set_method="instance_farmers",
        pay_unmatched=True,
        seed=optimizer_seed,
        stabilize_final_solution=True
    )
    summary_dominance_pay = optimizer.solve(
        dominance_pay_options,
        epsilons=epsilons,
    )

    return {
        "schema_version": 1,
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
        # epsilons and het_costs are recorded by the summary itself
        # (summary.params); farmer quantities by its instance_snapshot.
        "summary_vanilla_no_pay": summary_vanilla_no_pay.return_dict(),
        "summary_vanilla_pay": summary_vanilla_pay.return_dict(),
        "summary_structured_no_pay": summary_structured_no_pay.return_dict(),
        "summary_structured_pay": summary_structured_pay.return_dict(),
        "summary_dominance_no_pay": summary_dominance_no_pay.return_dict(),
        "summary_dominance_pay": summary_dominance_pay.return_dict()
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
    results_path = Path("results") / "exp_2" / f"job_{job_id}"
    results_path.mkdir(parents=True, exist_ok=True)

    with graph_path.open("rb") as file:
        graph = pickle.load(file)

    solver_threads = utils.get_solver_threads()

    experiment_metadata = {
        "experiment": "exp_2",
        "base_seed": BASE_SEED,
        "job_id": job_id,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "stable_platform_matchings_version": utils.package_version(
            "stable-platform-matchings"
        ),
        "gurobipy_version": utils.package_version("gurobipy"),
        "solver_threads": solver_threads,
        "constants": {
            "min_quantity": MIN_QUANTITY,
            "max_quantity": MAX_QUANTITY,
            "max_perturb": MAX_PERTURB,
            "het_cost_mean": HET_COST_MEAN,
            "het_cost_sd": HET_COST_SD,
            "vrp_time_limit_seconds": (
                VRP_TIME_LIMIT_SECONDS
            ),
        },
    }

    save_path = results_path / f"job_{job_id}.json.gz"

    job_payload = utils.JobPayload(
        schema_version=1,
        job_id=job_id,
        experiment_metadata=experiment_metadata,
    )

    for run_index in range(N_RUNS):
        run_payload = run_one(
            job_id=job_id,
            run_index=run_index,
            instance_paths=instance_paths,
            graph=graph,
            solver_threads=solver_threads
        )

        # format results for correctness
        job_payload.add_run(run_payload)
        job_payload.save(save_path)

        print(f"Saved run {run_index} ({job_payload.n_runs}/{N_RUNS}) to {save_path}")


if __name__ == "__main__":
    main()