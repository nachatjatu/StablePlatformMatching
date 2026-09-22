from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import math

from stable_platform_matchings import Optimizer, InstanceGenerator
from stable_platform_matchings.optimization.options import OptimizerParams, SolverOptions
import stable_platform_matchings.experiments.utils as utils

BASE_SEED = 20260806
VRP_TIME_LIMIT_SECONDS = 900
DATA_DIR = Path("data")

FARMERS_FULL_CSV_PATH = DATA_DIR / Path("farmers.csv")
FARMERS_14_CSV_PATH = DATA_DIR / Path("farmers_14.csv")
INTERMEDIARIES_CSV_PATH = DATA_DIR / Path("intermediaries.csv")
GRAPH_PKL_PATH = DATA_DIR / Path("graph_0-14960_00_new.pickle")
ALPHA_JSON_PATH = DATA_DIR / Path("precomputed_alpha.json")
SIGMAS_JSON_PATH = DATA_DIR / Path("precomputed_sigmas.json")

N_HIST_SETS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BASE_EPSILON = 4
BETA = 0.05

N_CYCLES = 11
N_INTS = 12
CYCLE_LENGTH = 14

def run_one(
    *,
    job_id: int,
    generator: InstanceGenerator,
    n_hist_sets: int,
    solver_threads: int,
    base_epsilon: float,
    hist_set_method: str
) -> dict[str, Any]:

    # set random seeding
    print("Setting random seeding...")
    seed_sequence = np.random.SeedSequence(
        [BASE_SEED, job_id, n_hist_sets]
    )
    sampling_seed_sequence, optimizer_seed_sequence = (
        seed_sequence.spawn(2)
    )
    optimizer_seed = int(
        optimizer_seed_sequence.generate_state(1, dtype=np.uint32)[0]
    )

    # load instance generator
    print("Generating instance...")
    instance = generator.gen_instance(
        instance_id=str(job_id),
        day=N_CYCLES * CYCLE_LENGTH - 1,
        n_hist_sets=n_hist_sets,
    )

    # sample inputs
    print("Sampling inputs...")
    naive_epsilons = {
        intermediary.id: float(base_epsilon)
        for intermediary in instance.intermediaries
    }

    epsilon_1 = base_epsilon / np.log(1 / BETA)
    epsilon_n = lambda N: epsilon_1 * np.log(1 / BETA) / N

    scaled_root_epsilons = {
        intermediary.id: float(base_epsilon) / math.sqrt(n_hist_sets)
        for intermediary in instance.intermediaries
    }

    scaled_epsilons = {
        intermediary.id: float(epsilon_n(n_hist_sets))
        for intermediary in instance.intermediaries
    }

    het_costs = {
        intermediary.id: float(2.0 * instance.dist_to_mill[intermediary.id])
        for intermediary in instance.intermediaries
    }

    # initialize optimizer
    print("Initializing optimizer...")
    params = OptimizerParams(
        het_costs=het_costs,
        backend="gurobi",
        vrp_mode="approximate",
        vrp_time_limit_seconds=VRP_TIME_LIMIT_SECONDS,
        threads=solver_threads,
    )
    optimizer = Optimizer(
        instance=instance,
        params=params,
    )


    # solve with naive epsilon
    options = SolverOptions(
        strategy="heuristic_accelerated",
        structured_farmer_payments=False,
        dominance_constraints=False,
        early_stop_threshold=0.0,
        hist_set_method=hist_set_method,
        pay_unmatched=False,
        seed=optimizer_seed,
        stabilize_final_solution=True
    )

    # naive
    print("Solving naive...")
    summary_naive = optimizer.solve(options, epsilons=naive_epsilons)

    # scaled
    print("Solving scaled root...")
    summary_scaled_root = optimizer.solve(
        options,
        epsilons=scaled_root_epsilons,
    )

    print("Solving scaled...")
    summary_scaled = optimizer.solve(options, epsilons=scaled_epsilons)

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
            "hist_set_method": hist_set_method
        },
        # epsilons and het_costs are recorded by the summary itself
        # (summary.params); farmer quantities by its instance_snapshot.
        "summary_naive": summary_naive.return_dict(),
        "summary_scaled_root": summary_scaled_root.return_dict(),
        "summary_scaled": summary_scaled.return_dict(),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python experiment.py JOB_ID")

    job_id = int(sys.argv[1])

    # get data paths
    results_path = Path("results") / "exp_6" / f"job_{job_id}"
    results_path.mkdir(parents=True, exist_ok=True)

    # set seeds
    job_seed_sequence = np.random.SeedSequence([BASE_SEED, job_id])
    intermediary_seed_sequence, calendar_seed_sequence = job_seed_sequence.spawn(2)
    intermediary_seed = int(
        intermediary_seed_sequence.generate_state(1, dtype=np.uint32)[0]
    )
    calendar_seed = int(
        calendar_seed_sequence.generate_state(1, dtype=np.uint32)[0]
    )

    print("Loading instance generator...")
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

    solver_threads = utils.get_solver_threads()

    experiment_metadata = {
        "experiment": "exp_6",
        "base_seed": BASE_SEED,
        "job_id": job_id,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "stable_platform_matchings_version": utils.package_version(
            "stable-platform-matchings"
        ),
        "gurobipy_version": utils.package_version("gurobipy"),
        "solver_threads": solver_threads,
        "sampled_inputs": {
            "intermediary_seed": intermediary_seed,
            "calendar_seed": calendar_seed,
        },
        "constants": {
            "vrp_time_limit_seconds": VRP_TIME_LIMIT_SECONDS,
            "base_epsilon": BASE_EPSILON,
            "beta": BETA,
            "n_hist_sets_values": N_HIST_SETS,
            "n_cycles": N_CYCLES,
            "n_intermediaries": N_INTS,
            "cycle_length": CYCLE_LENGTH,
        },
    }

    save_path = results_path / f"job_{job_id}.json.gz"

    job_payload = utils.JobPayload(
        schema_version=1,
        job_id=job_id,
        experiment_metadata=experiment_metadata,
    )

    for n_hist_sets in N_HIST_SETS:
        run_payload = run_one(
            job_id=job_id,
            generator=generator,
            n_hist_sets=n_hist_sets,
            solver_threads=solver_threads,
            base_epsilon=BASE_EPSILON,
            hist_set_method="original"
        )

        # format results for correctness
        job_payload.add_run(run_payload)
        job_payload.save(save_path)

        print(
            f"Saved n_hist_sets {n_hist_sets} "
            f"({job_payload.n_runs}/{len(N_HIST_SETS)}) "
            f"to {save_path}"
        )

if __name__ == "__main__":
    main()