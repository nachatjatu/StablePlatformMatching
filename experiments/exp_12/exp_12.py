from __future__ import annotations

import gzip
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stable_platform_matchings import InstanceGenerator, Optimizer
from stable_platform_matchings.domain.entities import Farmer, Intermediary, Mill
from stable_platform_matchings.domain.instance import Instance
from stable_platform_matchings.generation.instance_generator import DEFAULT_MILL, FALLBACK_SIGMA
from stable_platform_matchings.optimization.options import OptimizerParams, SolverOptions
import stable_platform_matchings.experiments.utils as utils


BASE_SEED = 20261012
VRP_TIME_LIMIT_SECONDS = 900
DATA_DIR = Path("data")
INSTANCES_CSV = Path("experiments/exp_12/compact_window_instances.csv")

FARMERS_FULL_CSV_PATH = DATA_DIR / "farmers.csv"
FARMERS_14_CSV_PATH = DATA_DIR / "farmers_14.csv"
INTERMEDIARIES_CSV_PATH = DATA_DIR / "intermediaries.csv"
GRAPH_PKL_PATH = DATA_DIR / "graph_0-14960_00_new.pickle"
ALPHA_JSON_PATH = DATA_DIR / "precomputed_alpha.json"
SIGMAS_JSON_PATH = DATA_DIR / "precomputed_sigmas.json"

SIGMA_ELL = Instance.TRUCK_FIXED_COST

COARSE_STEP = 0.5
PREDICTED_STEP = 0.1
PREDICTED_RADIUS = 2
JUMP_TOL = 0.025
MAX_BISECTION_ROUNDS = 6

HIST_SET_METHOD = "instance_farmers"
SOLVER_OPTIONS = dict(
    strategy="paper_bnb",
    structured_farmer_payments=False,
    dominance_constraints=False,
    early_stop_threshold=1e-4,
    hist_set_method=HIST_SET_METHOD,
    pay_unmatched=False,
    stabilize_final_solution=True,
)


def load_windows() -> pd.DataFrame:
    """
    Load the instance parameters and derive Prop 3's thresholds.

    Each intermediary's status quo is a whole number of farmers of `farmer_quantity` each
    (1 for every row of the compact CSV).
    """
    windows = pd.read_csv(INSTANCES_CSV).set_index("instance").sort_index()

    # status quo quantities must be whole numbers of farmers
    for typ, label in [("h", "high"), ("l", "low")]:
        windows[f"farmers_per_{label}"] = (windows[f"n_{typ}"] / windows["farmer_quantity"]).round().astype(int)
        if not np.allclose(windows[f"farmers_per_{label}"] * windows["farmer_quantity"], windows[f"n_{typ}"]):
            raise ValueError(f"n_{typ} is not a whole number of farmers of farmer_quantity")

    windows["F"] = windows["T_l"] * windows["n_l"] + windows["T_h"] * windows["n_h"]
    expected = windows["T_l"] * windows["farmers_per_low"] + windows["T_h"] * windows["farmers_per_high"]
    if not (windows["n_farmers"] == expected).all():
        raise ValueError("n_farmers does not equal T_l n_l + T_h n_h in farmers")

    windows["a"] = windows["n_l"] + windows["eps_l"]
    windows["L"] = windows["K"] - windows["n_h"]
    windows["M"] = windows["T_l"] * windows["n_l"] / windows["T_h"]
    b = windows["M"] + windows["a"]
    disc = b**2 - 4 * windows["M"] * windows["rho"] * windows["a"]
    windows["eps_h_minus"] = np.where(disc >= 0, (b - np.sqrt(disc.clip(lower=0))) / 2, np.nan)
    windows["eps_h_plus"] = np.where(disc >= 0, (b + np.sqrt(disc.clip(lower=0))) / 2, np.nan)

    # saturated-regime threshold (EC4), with sigma_l = 1 and sigma_h = rho
    e_l = windows["sigma_l"] / windows["a"]
    s_h, s_l = windows["sigma_h"], windows["sigma_l"]
    K, n_h, n_l, T_h, T_l, F = (windows[c] for c in ["K", "n_h", "n_l", "T_h", "T_l", "F"])
    eps_l_s = K * (F - T_h * K) * s_l / (F * s_h - T_h * K * s_l) - n_l
    cand_1 = windows["L"] + n_h * (s_h - s_l) / (K * e_l - s_l)
    cand_2 = T_l * n_l * s_h / ((F - T_h * K) * e_l + T_h * s_l)
    windows["eps_h_bar"] = np.where(windows["eps_l"] < eps_l_s, np.minimum(cand_1, cand_2), np.inf)

    # the analytic thresholds must reproduce the CSV's integer regime predictions:
    # regime_unsat covers integers up to L, regime_sat (when given) the integers above L
    for instance_id, row in windows.iterrows():
        start, end = (int(x) for x in row["eps_h_sweep"].split("-"))
        integers = range(start, end + 1)
        unsat = "".join(
            "I" if row["eps_h_minus"] < e < row["eps_h_plus"] else "E"
            for e in integers if e <= row["L"]
        )
        sat = "".join("I" if e > row["eps_h_bar"] else "E" for e in integers if e > row["L"])
        expected_sat = "" if pd.isna(row["regime_sat"]) else row["regime_sat"]
        if unsat != row["regime_unsat"] or sat != expected_sat:
            raise ValueError(
                f"instance {instance_id}: analytic thresholds give {unsat}|{sat}, "
                f"CSV has {row['regime_unsat']}|{expected_sat}"
            )

    return windows


def build_instance(
    generator: InstanceGenerator,
    instance_id: int,
    row: pd.Series,
) -> tuple[Instance, set[str], dict[str, float]]:
    """
    Build a two-type instance with generator geography and forced quantities.

    Returns the instance, the high-type IDs, and the heterogeneous costs.
    """
    T_l, T_h = int(row["T_l"]), int(row["T_h"])
    K = float(row["K"])
    farmer_quantity = float(row["farmer_quantity"])
    farmers_per = {"h": int(row["farmers_per_high"]), "l": int(row["farmers_per_low"])}

    seed_sequence = np.random.SeedSequence([BASE_SEED, instance_id])
    intermediary_seed_sequence, farmer_seed_sequence = seed_sequence.spawn(2)

    # generate T_l + T_h total intermediaries
    generator.gen_intermediaries(
        n_intermediaries=T_l + T_h,
        seed=int(intermediary_seed_sequence.generate_state(1, dtype=np.uint32)[0]),
    )
    rng = np.random.default_rng(farmer_seed_sequence)

    intermediary_ids = list(generator.intermediaries)
    high_ids = set(rng.choice(intermediary_ids, size=T_h, replace=False).tolist())

    farmers, intermediaries = [], []
    for intermediary_id in intermediary_ids:
        data = generator.intermediaries[intermediary_id]

        # force status quo quantities: farmers_per_* farmers of farmer_quantity tons each
        quantities = [farmer_quantity] * farmers_per["h" if intermediary_id in high_ids else "l"]

        farmer_xys = generator.gen_farmer_xys(
            intermediary_xy=data["xy"],
            intermediary_type=data["type"],
            n_farmers=len(quantities),
            rng=rng,
            sigma=generator.sigmas.get(data["type"], FALLBACK_SIGMA),
        )

        farmer_ids = []
        for k, (xy, quantity) in enumerate(zip(farmer_xys, quantities, strict=True)):
            lon, lat = generator.xy_to_ll.transform(xy[0], xy[1])
            farmer_id = f"{intermediary_id}_f{k}"
            farmers.append(
                Farmer(
                    id=farmer_id,
                    location=(lat, lon),
                    quantity=quantity,
                    intermediary_id=intermediary_id,
                )
            )
            farmer_ids.append(farmer_id)

        intermediaries.append(
            Intermediary(
                id=intermediary_id,
                capacity=K,
                location=data["ll"],
                hist_sets=[farmer_ids],
            )
        )

    instance = Instance(
        instance_id=f"window_{instance_id}",
        farmers=farmers,
        intermediaries=intermediaries,
        mill=Mill(id=DEFAULT_MILL["mill_id"], location=DEFAULT_MILL["location"]),
    )
    instance.truck_capacity_tons = K
    instance.truck_fixed_cost = SIGMA_ELL
    instance.set_graph(generator.graph)

    if len(instance.farmers) != row["n_farmers"]:
        raise ValueError(f"built {len(instance.farmers)} farmers, CSV has {row['n_farmers']}")
    if len(instance.intermediaries) != row["n_intermediaries"]:
        raise ValueError(
            f"built {len(instance.intermediaries)} intermediaries, "
            f"CSV has {row['n_intermediaries']}"
        )

    # use het_costs to modify high-type intermediary fixed cost
    het_costs = {
        intermediary_id: (float(row["rho"]) - 1.0) * SIGMA_ELL if intermediary_id in high_ids else 0.0
        for intermediary_id in intermediary_ids
    }
    return instance, high_ids, het_costs


def coarse_grid(row: pd.Series) -> list[float]:
    upper = float(row["L"]) + 1
    return list(np.arange(0, upper + COARSE_STEP / 2, COARSE_STEP))


def predicted_grid(row: pd.Series) -> list[float]:
    points = []
    # eps_h_bar is the saturated-regime threshold (EC4); it lies above L when finite
    for x in [row["eps_h_minus"], row["eps_h_plus"], row["L"], row["eps_h_bar"]]:
        if pd.notna(x) and np.isfinite(x) and x <= row["L"] + 1:
            points.extend(x + PREDICTED_STEP * np.arange(-PREDICTED_RADIUS, PREDICTED_RADIUS + 1))
    return points


def matching_signature(run: dict[str, Any]) -> tuple[tuple[str, ...], int]:
    """Which high types are matched, and how many low types: a 'jump' is a change in this."""
    matched = set(run["summary"]["platform_solve_result"]["primary_result"]["selected_intermediaries"])
    high_ids = set(run["high_ids"])
    return tuple(sorted(matched & high_ids)), len(matched - high_ids)


def bisection_points(runs: list[dict[str, Any]]) -> list[float]:
    """Midpoints of every bracket wider than JUMP_TOL whose endpoints' matchings differ."""
    solved = sorted(runs, key=lambda run: run["eps_h"])
    return [
        (left["eps_h"] + right["eps_h"]) / 2
        for left, right in zip(solved, solved[1:])
        if right["eps_h"] - left["eps_h"] > JUMP_TOL
        and matching_signature(left) != matching_signature(right)
    ]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python experiment.py JOB_ID")

    job_id = int(sys.argv[1])

    windows = load_windows()
    if not 0 <= job_id < len(windows):
        raise SystemExit(f"JOB_ID must be in [0, {len(windows) - 1}], got {job_id}")

    # one job per CSV row, in instance order
    instance_id = int(windows.index[job_id])
    row = windows.loc[instance_id]

    results_path = Path("results") / "exp_12" / f"job_{job_id}"
    results_path.mkdir(parents=True, exist_ok=True)
    save_path = results_path / f"job_{job_id}.json.gz"

    print("Loading instance generator...")
    generator = InstanceGenerator(
        farmers_full_csv_path=FARMERS_FULL_CSV_PATH,
        farmers_14_csv_path=FARMERS_14_CSV_PATH,
        intermediaries_csv_path=INTERMEDIARIES_CSV_PATH,
        graph_pkl_path=GRAPH_PKL_PATH,
        alpha_json_path=ALPHA_JSON_PATH,
        sigmas_json_path=SIGMAS_JSON_PATH,
    )

    print(f"Building instance {instance_id}...")
    instance, high_ids, het_costs = build_instance(generator, instance_id, row)

    solver_threads = utils.get_solver_threads()
    optimizer_seed = int(
        np.random.SeedSequence([BASE_SEED, instance_id, 1]).generate_state(1, dtype=np.uint32)[0]
    )

    # one Optimizer reused across ε_h: solve() resets solver state but keeps cached routing costs
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

    experiment_metadata = {
        "experiment": "exp_12",
        "base_seed": BASE_SEED,
        "job_id": job_id,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "stable_platform_matchings_version": utils.package_version("stable-platform-matchings"),
        "gurobipy_version": utils.package_version("gurobipy"),
        "solver_threads": solver_threads,
        "constants": {
            "vrp_time_limit_seconds": VRP_TIME_LIMIT_SECONDS,
            "sigma_ell": SIGMA_ELL,
            "coarse_step": COARSE_STEP,
            "predicted_step": PREDICTED_STEP,
            "predicted_radius": PREDICTED_RADIUS,
            "jump_tol": JUMP_TOL,
            "max_bisection_rounds": MAX_BISECTION_ROUNDS,
            "solver_options": SOLVER_OPTIONS,
        },
    }

    # CSV row plus the derived Prop 3 quantities, recorded once per job
    window = utils.encode_and_check(
        {
            key: (value.item() if isinstance(value, np.generic) else value)
            for key, value in row.to_dict().items()
        },
        label="window",
    )

    job_payload = utils.JobPayload(
        schema_version=1,
        job_id=job_id,
        instance_id=instance_id,
        instances_csv=str(INSTANCES_CSV),
        window=window,
        high_ids=sorted(high_ids),
        optimizer_seed=optimizer_seed,
        experiment_metadata=experiment_metadata,
    )

    # resume a timed-out or preempted job: keep runs already on disk
    runs: list[dict[str, Any]] = []
    if save_path.exists():
        with gzip.open(save_path, "rt", encoding="utf-8") as f:
            previous = json.load(f)
        for run in previous["runs"]:
            job_payload.add_run(run)
            runs.append(run)
        print(f"Resuming from {len(runs)} saved runs")

    def solve_points(eps_hs: list[float], phase: str) -> None:
        solved = {round(run["eps_h"], 6) for run in runs}
        todo = sorted({round(float(x), 6) for x in eps_hs if x >= 0} - solved)
        for eps_h in todo:
            epsilons = {
                intermediary.id: eps_h if intermediary.id in high_ids else float(row["eps_l"])
                for intermediary in instance.intermediaries
            }
            start = time.time()
            summary = optimizer.solve(
                SolverOptions(**SOLVER_OPTIONS, seed=optimizer_seed),
                epsilons=epsilons,
            )
            run = {
                "eps_h": eps_h,
                "phase": phase,
                "high_ids": sorted(high_ids),
                "wall_time": time.time() - start,
                # epsilons and het_costs are recorded by the summary itself (summary.params)
                "summary": summary.return_dict(),
            }
            job_payload.add_run(run)
            job_payload.save(save_path)
            runs.append(run)

            high_matched, n_low = matching_signature(run)
            print(
                f"[{phase}] instance {instance_id}, eps_h={eps_h:.4f}: "
                f"{len(high_matched)}/{len(high_ids)} high, {n_low} low matched "
                f"({run['wall_time']:.0f}s, {job_payload.n_runs} runs saved)"
            )

    solve_points(coarse_grid(row), "coarse")
    solve_points(predicted_grid(row), "predicted")

    for round_index in range(MAX_BISECTION_ROUNDS):
        midpoints = bisection_points(runs)
        if not midpoints:
            break
        print(f"Bisection round {round_index}: {len(midpoints)} brackets")
        solve_points(midpoints, f"bisection_{round_index}")

    print(f"Done: {job_payload.n_runs} runs saved to {save_path}")


if __name__ == "__main__":
    main()
