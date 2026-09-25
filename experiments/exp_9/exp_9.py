from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from stable_platform_matchings import InstanceGenerator, Optimizer
from stable_platform_matchings.domain.instance import Instance
from stable_platform_matchings.optimization.options import OptimizerParams, SolverOptions
import stable_platform_matchings.experiments.utils as utils

BASE_SEED = 20260924
VRP_TIME_LIMIT_SECONDS = 900
DATA_DIR = Path("data")

FARMERS_FULL_CSV_PATH = DATA_DIR / "farmers.csv"
FARMERS_14_CSV_PATH = DATA_DIR / "farmers_14.csv"
INTERMEDIARIES_CSV_PATH = DATA_DIR / "intermediaries.csv"
GRAPH_PKL_PATH = DATA_DIR / "graph_0-14960_00_new.pickle"
ALPHA_JSON_PATH = DATA_DIR / "precomputed_alpha.json"
SIGMAS_JSON_PATH = DATA_DIR / "precomputed_sigmas.json"

N_INTS = 12
CYCLE_LENGTH = 14
# instance_farmers uses each intermediary's farmers on the sampled day, so recorded
# historical sets are unused. The generator's day-matched historical sets miss ~35% of
# existing relationships (pickup offsets are redrawn every cycle), which made `union`
# status-quo quantities too small for the cap to matter.
HIST_SET_METHOD = "instance_farmers"
N_HIST_SETS = 1
# sample days need N_HIST_SETS full cycles of history behind them
N_CYCLES = N_HIST_SETS + 1
N_SAMPLES = 10
# each sample is a different day of the final cycle, i.e. a different subset of farmers
SAMPLE_DAYS = list(range(N_CYCLES * CYCLE_LENGTH - N_SAMPLES, N_CYCLES * CYCLE_LENGTH))

# same input distributions as exp_8, so results are comparable
MIN_EPSILON = 0.0
MAX_EPSILON = 6.0
HET_COST_MEAN = 0.0
HET_COST_SD = 100_000.0

STRATEGIES = [
    "paper_bnb",
    "npm_capped",
    "npm_uncapped",
]


def sample_epsilons(instance: Instance, rng: np.random.Generator) -> dict[str, float]:
    return {
        intermediary.id: float(rng.uniform(MIN_EPSILON, MAX_EPSILON))
        for intermediary in instance.intermediaries
    }


def sample_het_costs(instance: Instance, rng: np.random.Generator) -> dict[str, float]:
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
    sample_index: int,
    generator: InstanceGenerator,
    solver_threads: int,
) -> dict[str, Any]:
    seed_sequence = np.random.SeedSequence([BASE_SEED, job_id, sample_index])
    sampling_seed_sequence, optimizer_seed_sequence = seed_sequence.spawn(2)
    rng = np.random.default_rng(sampling_seed_sequence)
    optimizer_seed = int(optimizer_seed_sequence.generate_state(1, dtype=np.uint32)[0])

    day = SAMPLE_DAYS[sample_index]
    print(f"Generating instance for day {day}...")
    instance = generator.gen_instance(
        instance_id=f"{job_id}_{sample_index}",
        day=day,
        n_hist_sets=N_HIST_SETS,
    )

    print("Sampling inputs...")
    epsilons = sample_epsilons(instance, rng)
    het_costs = sample_het_costs(instance, rng)

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

    summaries = {}
    for strategy in STRATEGIES:
        print(f"Solving with {strategy}...")
        options = SolverOptions(
            strategy=strategy,
            structured_farmer_payments=False,
            dominance_constraints=False,
            early_stop_threshold=1e-4,
            hist_set_method=HIST_SET_METHOD,
            pay_unmatched=False,
            seed=optimizer_seed,
            stabilize_final_solution=True,
        )
        summaries[strategy] = optimizer.solve(options, epsilons=epsilons).return_dict()

    return {
        "schema_version": 1,
        "metadata": {
            "sample_index": sample_index,
            "day": day,
            "n_farmers": len(instance.farmers),
            "optimizer_seed": optimizer_seed,
            "seed_sequence_state": seed_sequence.state,
            "sampling_seed_sequence_state": sampling_seed_sequence.state,
            "optimizer_seed_sequence_state": optimizer_seed_sequence.state,
        },
        # epsilons and het_costs are recorded by each summary (summary.params);
        # farmer quantities and hist sets by its instance_snapshot.
        "summary_exact": summaries["lagrangian_bnp"],
        "summary_heuristic": summaries["paper_bnb"],
        "summary_npm_capped": summaries["npm_capped"],
        "summary_npm_uncapped": summaries["npm_uncapped"],
    }


def build_generator(job_id: int) -> tuple[InstanceGenerator, int, int]:
    job_seed_sequence = np.random.SeedSequence([BASE_SEED, job_id])
    intermediary_seed_sequence, calendar_seed_sequence = job_seed_sequence.spawn(2)
    intermediary_seed = int(intermediary_seed_sequence.generate_state(1, dtype=np.uint32)[0])
    calendar_seed = int(calendar_seed_sequence.generate_state(1, dtype=np.uint32)[0])

    print("Loading instance generator...")
    generator = InstanceGenerator(
        farmers_full_csv_path=FARMERS_FULL_CSV_PATH,
        farmers_14_csv_path=FARMERS_14_CSV_PATH,
        intermediaries_csv_path=INTERMEDIARIES_CSV_PATH,
        graph_pkl_path=GRAPH_PKL_PATH,
        alpha_json_path=ALPHA_JSON_PATH,
        sigmas_json_path=SIGMAS_JSON_PATH,
    )
    generator.gen_intermediaries(n_intermediaries=N_INTS, seed=intermediary_seed)
    generator.gen_calendar(n_cycles=N_CYCLES, cycle_length=CYCLE_LENGTH, seed=calendar_seed)
    return generator, intermediary_seed, calendar_seed


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python experiment.py JOB_ID")

    job_id = int(sys.argv[1])

    results_path = Path("results") / "exp_9" / f"job_{job_id}"
    results_path.mkdir(parents=True, exist_ok=True)

    generator, intermediary_seed, calendar_seed = build_generator(job_id)
    solver_threads = utils.get_solver_threads()

    experiment_metadata = {
        "experiment": "exp_9",
        "base_seed": BASE_SEED,
        "job_id": job_id,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "stable_platform_matchings_version": utils.package_version("stable-platform-matchings"),
        "gurobipy_version": utils.package_version("gurobipy"),
        "solver_threads": solver_threads,
        "sampled_inputs": {
            "intermediary_seed": intermediary_seed,
            "calendar_seed": calendar_seed,
        },
        "constants": {
            "vrp_time_limit_seconds": VRP_TIME_LIMIT_SECONDS,
            "n_intermediaries": N_INTS,
            "n_cycles": N_CYCLES,
            "cycle_length": CYCLE_LENGTH,
            "n_hist_sets": N_HIST_SETS,
            "hist_set_method": HIST_SET_METHOD,
            "sample_days": SAMPLE_DAYS,
            "min_epsilon": MIN_EPSILON,
            "max_epsilon": MAX_EPSILON,
            "het_cost_mean": HET_COST_MEAN,
            "het_cost_sd": HET_COST_SD,
            "strategies": STRATEGIES,
        },
    }

    save_path = results_path / f"job_{job_id}.json.gz"
    job_payload = utils.JobPayload(
        schema_version=1,
        job_id=job_id,
        experiment_metadata=experiment_metadata,
    )

    for sample_index in range(N_SAMPLES):
        run_payload = run_one(
            job_id=job_id,
            sample_index=sample_index,
            generator=generator,
            solver_threads=solver_threads,
        )
        job_payload.add_run(run_payload)
        job_payload.save(save_path)
        print(f"Saved sample {sample_index} ({job_payload.n_runs}/{N_SAMPLES}) to {save_path}")


if __name__ == "__main__":
    main()
