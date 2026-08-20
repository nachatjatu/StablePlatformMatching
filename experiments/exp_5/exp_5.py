from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

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

EPSILONS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
N_HIST_SETS = [1, 3, 5, 7, 9]
N_CYCLES = 11
N_INTS = 12
CYCLE_LENGTH = 14

def run_one(
    *,
    job_id: int,
    generator: InstanceGenerator,
    n_hist_sets: int,
    solver_threads: int,
    epsilon: float,
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
    epsilons = {
        intermediary.id: float(epsilon)
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

    # solve
    print("Solving...")
    options = SolverOptions(
        strategy="heuristic_optimized",
        structured_farmer_payments=False,
        dominance_constraints=False,
        early_stop=False,
        hist_set_method=hist_set_method,
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
            "hist_set_method": hist_set_method
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

    # get data paths
    results_path = Path("results") / "exp_5" / f"job_{job_id}"
    results_path.mkdir(parents=True, exist_ok=True)

    # set seeds
    job_seed_sequence = np.random.SeedSequence([BASE_SEED, job_id])
    intermediary_seed_sequence, calendar_seed_sequence, n_hist_sets_seed_sequence = job_seed_sequence.spawn(3)
    intermediary_seed = int(
        intermediary_seed_sequence.generate_state(1, dtype=np.uint32)[0]
    )
    calendar_seed = int(
        calendar_seed_sequence.generate_state(1, dtype=np.uint32)[0]
    )
    n_hist_sets_seed = int(
        n_hist_sets_seed_sequence.generate_state(1, dtype=np.uint32)[0]
    )

    n_hist_sets_rng = np.random.default_rng(seed=n_hist_sets_seed)

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
        "experiment": "exp_5",
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
            "epsilons": EPSILONS,
            "n_hist_sets_seed": n_hist_sets_seed,
            "intermediary_seed": intermediary_seed,
            "calendar_seed": calendar_seed,
        },
        "constants": {
            "vrp_time_limit_seconds": VRP_TIME_LIMIT_SECONDS,
            "epsilons": EPSILONS,
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
        "n_runs": 0,
        "runs": [],
    }

    n_hist_sets = int(n_hist_sets_rng.choice(N_HIST_SETS))

    for epsilon in EPSILONS:
        run_payload = run_one(
            job_id=job_id,
            generator=generator,
            n_hist_sets=n_hist_sets,
            solver_threads=solver_threads,
            epsilon=epsilon,
            hist_set_method="original"
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

        print(
            f"Saved n_hist_sets {n_hist_sets} "
            f"({job_payload['n_runs']}/{len(EPSILONS)}) "
            f"to {save_path}"
        )

if __name__ == "__main__":
    main()