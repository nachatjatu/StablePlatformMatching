"""
exp_14: does keeping the high types matched get more expensive than pricing them out?

Three instances where the road network departs from Prop 3 past saturation:

  #5, #29 (exp_10 road / exp_11 star)  a second inefficient regime just past L that ends
                                      on the road network but persists with linear costs
  #11     (exp_12 road / exp_13 star)  the first window's exit, at eps_h+ = 6.6 with linear
                                      costs but only at 7.70 on the road network (past L = 7)

Each switch should be a race between two profits:

  inefficient  Pi_I(eps_h): the best platform profit with all high types matched
  efficient    Pi_E(eps_h): the best platform profit with no high type matched

The saved sweeps only give Pi_I where the solver chose it, and Pi_E as `forced_lower_bound`,
the profit of one (tied) min-cost set, i.e. a lower bound. This experiment solves both
alternatives exactly on a common eps_h grid, in both arms, by restricting the search: the
high types forced matched (Pi_I) or forced unmatched (Pi_E).

Each solve also records every intermediary's eta and the dual of its stability constraint,
so the envelope slope dPi/d eps_h = sum over high types of stability_dual * eta can be
checked against finite differences of Pi.

Instances, seeds, and solver options are exactly those of the source experiments. Run from
the repo root; each (instance, arm, side) saves to its own file, so several can run in
parallel, and a rerun resumes from the saved runs:

    python experiments/exp_14/exp_14.py                                 # everything
    python experiments/exp_14/exp_14.py --instance 29 --arm road --side efficient

Solve times (4 threads): #5 about 2 s, #29 about 2-3 min, #11 about 3 min (star) and
17 min (road), so #11's grid is coarse.

Results go to results/exp_14/instance_{id}_{arm}_{side}.json.gz.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from stable_platform_matchings import InstanceGenerator, Optimizer
from stable_platform_matchings.optimization.options import OptimizerParams, SolverOptions
import stable_platform_matchings.experiments.utils as utils

REPO = Path(__file__).resolve().parents[2]
for exp in ["exp_10", "exp_11", "exp_12", "exp_13"]:
    sys.path.insert(0, str(REPO / "experiments" / exp))

import exp_10  # noqa: E402
import exp_11  # noqa: E402
import exp_12  # noqa: E402
import exp_13  # noqa: E402

ARMS = ["road", "star"]
SIDES = ["inefficient", "efficient"]
RESULTS_DIR = Path("results") / "exp_14"


def grid(start: float, stop: float, step: float) -> list[float]:
    return np.round(np.arange(start, stop + step / 2, step), 6).tolist()


@dataclass(frozen=True)
class Case:
    road: ModuleType   # experiment that built the road-network instance
    star: ModuleType   # experiment that built its linear-cost (star) counterpart
    eps_h_grid: list[float]


CASES = {
    # from L = 5 to past the observed switch back (5.41)
    5: Case(road=exp_10, star=exp_11, eps_h_grid=grid(5.0, 5.6, 0.025)),
    # from L = 6 to past the observed switch back (6.46)
    29: Case(road=exp_10, star=exp_11, eps_h_grid=grid(6.0, 6.6, 0.05)),
    # from below eps_h+ = 6.6 to past the road exit (7.70) and L = 7
    11: Case(road=exp_12, star=exp_13, eps_h_grid=grid(6.4, 8.0, 0.2)),
}


def build(generator: InstanceGenerator, instance_id: int, arm: str):
    """Rebuild an instance exactly as its source experiment's job did."""
    module = getattr(CASES[instance_id], arm)
    row = module.load_windows().loc[instance_id]
    if arm == "road":
        instance, high_ids, het_costs = module.build_instance(generator, instance_id, row)
    else:
        instance, high_ids, het_costs, _ = module.build_instance(generator, instance_id, row, "star")
    return instance, high_ids, het_costs, row


def save_path(instance_id: int, arm: str, side: str) -> Path:
    return RESULTS_DIR / f"instance_{instance_id}_{arm}_{side}.json.gz"


def run_cell(generator: InstanceGenerator, instance_id: int, arm: str, side: str) -> None:
    case = CASES[instance_id]
    source = getattr(case, arm)
    path = save_path(instance_id, arm, side)

    solver_threads = utils.get_solver_threads()
    optimizer_seed = int(
        np.random.SeedSequence([source.BASE_SEED, instance_id, 1]).generate_state(1, dtype=np.uint32)[0]
    )
    job_payload = utils.JobPayload(
        schema_version=2,
        instance_id=instance_id,
        arm=arm,
        side=side,
        source_experiment=source.__name__,
        optimizer_seed=optimizer_seed,
        eps_h_grid=case.eps_h_grid,
        experiment_metadata={
            "experiment": "exp_14",
            "base_seed": source.BASE_SEED,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "stable_platform_matchings_version": utils.package_version("stable-platform-matchings"),
            "gurobipy_version": utils.package_version("gurobipy"),
            "solver_threads": solver_threads,
            "constants": {
                "vrp_time_limit_seconds": source.VRP_TIME_LIMIT_SECONDS,
                "sigma_ell": source.SIGMA_ELL,
                "solver_options": source.SOLVER_OPTIONS,
            },
        },
    )

    # resume: keep runs already on disk
    runs: list[dict[str, Any]] = []
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for run in json.load(f)["runs"]:
                job_payload.add_run(run)
                runs.append(run)
    solved = {round(run["eps_h"], 6) for run in runs}
    todo = [eps_h for eps_h in case.eps_h_grid if round(eps_h, 6) not in solved]
    print(f"#{instance_id} {arm} {side}: {len(runs)} saved, {len(todo)} to solve ({solver_threads} threads)")
    if not todo:
        return

    instance, high_ids, het_costs, row = build(generator, instance_id, arm)
    optimizer = Optimizer(
        instance=instance,
        params=OptimizerParams(
            het_costs=het_costs,
            backend="gurobi",
            vrp_mode="approximate",
            vrp_time_limit_seconds=source.VRP_TIME_LIMIT_SECONDS,
            threads=solver_threads,
            verbose=False,
        ),
    )
    restriction = {"forced_match": high_ids} if side == "inefficient" else {"forced_unmatch": high_ids}

    for eps_h in todo:
        epsilons = {
            t.id: eps_h if t.id in high_ids else float(row["eps_l"]) for t in instance.intermediaries
        }
        start = time.time()
        # also record the high types' worst-case deviation route mix (from the cut duals)
        summary = optimizer.solve(
            SolverOptions(**source.SOLVER_OPTIONS, seed=optimizer_seed, record_deviation_routes=tuple(sorted(high_ids))),
            epsilons=epsilons,
            **restriction,
        )
        run = {
            "arm": arm,
            "side": side,
            "eps_h": eps_h,
            "high_ids": sorted(high_ids),
            "forced_match": sorted(restriction.get("forced_match", [])),
            "forced_unmatch": sorted(restriction.get("forced_unmatch", [])),
            "wall_time": time.time() - start,
            "summary": summary.return_dict(),
        }
        job_payload.add_run(run)
        job_payload.save(path)
        runs.append(run)

        primary = summary.platform_solve_result.primary_result
        slope = sum(primary.stability_duals[h] * primary.eta[h] for h in high_ids)
        print(
            f"[#{instance_id} {arm} {side}] eps_h={eps_h:.3f}: profit {primary.platform_profit:>12,.0f}, "
            f"envelope slope {slope:>10,.0f}/ton ({run['wall_time']:.0f}s, {job_payload.n_runs} saved)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instance", type=int, nargs="+", choices=sorted(CASES), default=sorted(CASES))
    parser.add_argument("--arm", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--side", nargs="+", choices=SIDES, default=SIDES)
    args = parser.parse_args()

    # the experiment scripts use paths relative to the repo root
    os.chdir(REPO)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading instance generator...")
    generator = InstanceGenerator(
        farmers_full_csv_path=exp_10.FARMERS_FULL_CSV_PATH,
        farmers_14_csv_path=exp_10.FARMERS_14_CSV_PATH,
        intermediaries_csv_path=exp_10.INTERMEDIARIES_CSV_PATH,
        graph_pkl_path=exp_10.GRAPH_PKL_PATH,
        alpha_json_path=exp_10.ALPHA_JSON_PATH,
        sigmas_json_path=exp_10.SIGMAS_JSON_PATH,
    )
    for instance_id in args.instance:
        for arm in args.arm:
            for side in args.side:
                run_cell(generator, instance_id, arm, side)
    print("Done")


if __name__ == "__main__":
    main()
