"""
exp_11: exp_10's sweep with collection costs that respect the stylized model (§3.2).

The stylized model has linear collection costs c_t(s) = σ_t + cᵀs, so a farmer costs the
same to collect whoever collects them and whatever else is on the truck. Routing on the
real road tree is not linear: a route serving several farmers can cost less than the sum
of their single trips. Each job runs one instance under one of two linear-cost arms:

  zero  truck_cost_per_m = 0, so c_f = 0 (exactly the CSV's c_f).
  star  a star road graph with the mill at the hub and one private spur per entity. No
        two farmers share an edge, so a tour of a set S costs Σ_{f∈S} c_f, with c_f the
        farmer's real round-trip cost on the exp_10 road tree.

Prop 3's windows do not depend on c_f, so both arms should reproduce them exactly.
Instances, seeds, grid, and refinement are otherwise identical to exp_10.
"""

from __future__ import annotations

import gzip
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer

from stable_platform_matchings import InstanceGenerator, Optimizer
from stable_platform_matchings.domain.entities import Farmer, Intermediary, Mill
from stable_platform_matchings.domain.instance import Instance
from stable_platform_matchings.generation.instance_generator import DEFAULT_MILL, FALLBACK_SIGMA
from stable_platform_matchings.graphs.road_graphs import RoadGraph
from stable_platform_matchings.optimization.options import OptimizerParams, SolverOptions
import stable_platform_matchings.experiments.utils as utils


# same seed as exp_10, so each instance has the same geography, high types, and farmers
BASE_SEED = 20260930
VRP_TIME_LIMIT_SECONDS = 900
DATA_DIR = Path("data")
INSTANCES_CSV = Path("experiments/exp_11/small_window_instances.csv")

ROUTING_MODES = ["zero", "star"]
# star graph geometry only serves to snap entities to their own spur; spur lengths carry costs
STAR_RING_RADIUS_M = 1000.0
INTERMEDIARY_SPUR_LENGTH_M = 1.0
LINEARITY_CHECK_SETS = 200
LINEARITY_TOL = 1e-6

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
    """Load the instance parameters and derive Prop 3's window and saturation point."""
    windows = pd.read_csv(INSTANCES_CSV).set_index("instance").sort_index()

    windows["a"] = windows["n_l"] + windows["eps_l"]
    windows["L"] = windows["K"] - windows["n_h"]
    windows["M"] = windows["T_l"] * windows["n_l"] / windows["T_h"]
    b = windows["M"] + windows["a"]
    disc = b**2 - 4 * windows["M"] * windows["rho"] * windows["a"]
    windows["eps_h_minus"] = np.where(disc >= 0, (b - np.sqrt(disc.clip(lower=0))) / 2, np.nan)
    windows["eps_h_plus"] = np.where(disc >= 0, (b + np.sqrt(disc.clip(lower=0))) / 2, np.nan)

    # the analytic window must reproduce the CSV's integer regime predictions
    for instance_id, row in windows.iterrows():
        start, end = (int(x) for x in row["eps_h_sweep"].split("-"))
        analytic = "".join(
            "I" if row["eps_h_minus"] < e < row["eps_h_plus"] else "E"
            for e in range(start, end + 1)
        )
        if analytic != row["regime_by_eps_h"]:
            raise ValueError(
                f"instance {instance_id}: analytic window gives {analytic}, "
                f"CSV has {row['regime_by_eps_h']}"
            )

    return windows


def make_instance(
    generator: InstanceGenerator,
    instance_id: int,
    row: pd.Series,
) -> tuple[Instance, set[str], dict[str, float]]:
    """
    Build exp_10's two-type instance (generator geography, forced quantities), without a graph.

    Deterministic in (BASE_SEED, instance_id), so repeated calls give identical instances.
    Returns the instance, the high-type IDs, and the heterogeneous costs.
    """
    T_l, T_h = int(row["T_l"]), int(row["T_h"])
    n_l, n_h, K = float(row["n_l"]), float(row["n_h"]), float(row["K"])
    farmer_quantity = float(row["farmer_quantity"])

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

        # force status quo quantities
        n_t = n_h if intermediary_id in high_ids else n_l
        quantities = [farmer_quantity] * int(round(n_t / farmer_quantity))

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


def build_star_graph(instance: Instance, spur_lengths: dict[str, float], crs: Any) -> RoadGraph:
    """
    Star road graph: the mill at the hub and one paved spur per farmer and intermediary.

    Moves every entity onto its own node on a ring around the mill, so snapping in
    Instance.set_graph is one-to-one. Only the spur lengths enter costs; the ring geometry
    only keeps entities apart.
    """
    to_xy = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    mill_lat, mill_lon = instance.mill.location
    mill_x, mill_y = to_xy.transform(mill_lon, mill_lat)

    graph = nx.MultiDiGraph(crs=crs)
    graph.add_node(0, x=mill_x, y=mill_y)
    spoke_entities = list(instance.farmers) + list(instance.intermediaries)
    for node, entity in enumerate(spoke_entities, start=1):
        theta = 2 * np.pi * node / len(spoke_entities)
        x = mill_x + STAR_RING_RADIUS_M * np.cos(theta)
        y = mill_y + STAR_RING_RADIUS_M * np.sin(theta)
        graph.add_node(node, x=x, y=y)
        # "trunk" is paved, so the edge weight equals its length (no dirt factor)
        graph.add_edge(0, node, length=float(spur_lengths[entity.id]), highway="trunk", oneway=False)
        lon, lat = to_ll.transform(x, y)
        entity.location = (lat, lon)

    return RoadGraph(graph)


def build_instance(
    generator: InstanceGenerator,
    instance_id: int,
    row: pd.Series,
    routing_mode: str,
) -> tuple[Instance, set[str], dict[str, float], dict[str, Any]]:
    """
    Build the exp_10 instance under a linear-cost routing arm and verify its costs.

    Returns the instance, the high-type IDs, the heterogeneous costs, and a summary of the
    per-farmer collection costs c_f.
    """
    instance, high_ids, het_costs = make_instance(generator, instance_id, row)

    if routing_mode == "zero":
        instance.truck_cost_per_m = 0.0
        instance.set_graph(generator.graph)
    elif routing_mode == "star":
        # each farmer's spur is its one-way distance on the exp_10 road tree, so a lone
        # farmer costs the same round trip as in exp_10
        reference, reference_high_ids, _ = make_instance(generator, instance_id, row)
        if reference_high_ids != high_ids:
            raise RuntimeError("reference instance does not reproduce the high types")
        reference.set_graph(generator.graph)
        spur_lengths = {
            farmer.id: reference.dist_to_mill[farmer.id] / reference.truck_cost_per_m
            for farmer in reference.farmers
        }
        spur_lengths |= {t.id: INTERMEDIARY_SPUR_LENGTH_M for t in reference.intermediaries}
        instance.set_graph(build_star_graph(instance, spur_lengths, generator.graph.graph.graph["crs"]))

        farmer_nodes = [instance.entity_id_to_graph_node[f.id] for f in instance.farmers]
        other_nodes = {instance.entity_id_to_graph_node[e.id] for e in [*instance.intermediaries, instance.mill]}
        if len(set(farmer_nodes)) != len(farmer_nodes) or set(farmer_nodes) & other_nodes:
            raise RuntimeError("star graph: some farmers do not have a spur of their own")
    else:
        raise ValueError(f"unknown routing mode {routing_mode!r}")

    # verify that collection costs are additive: cost(S) = Σ_{f∈S} c_f. Route.cost includes
    # the truck fixed cost once per route, so it is netted out of every route
    def travel_cost(farmer_ids: list[str]) -> float:
        return instance.calculate_tree_path(farmer_ids).cost - instance.truck_fixed_cost

    c = {f.id: travel_cost([f.id]) for f in instance.farmers}
    rng = np.random.default_rng([BASE_SEED, instance_id, 2])
    farmer_ids = [f.id for f in instance.farmers]
    max_violation = 0.0
    for _ in range(LINEARITY_CHECK_SETS):
        size = int(rng.integers(2, min(int(row["K"]), len(farmer_ids)) + 1))
        subset = list(rng.choice(farmer_ids, size=size, replace=False))
        violation = abs(travel_cost(subset) - sum(c[f] for f in subset))
        max_violation = max(max_violation, violation)
    if max_violation > LINEARITY_TOL * max(1.0, max(c.values())):
        raise RuntimeError(f"{routing_mode}: collection costs are not additive (max violation {max_violation})")

    # Assumption 1(f): e_ℓ + max_f c_f < p (unit farmer quantities)
    e_ell = SIGMA_ELL / (float(row["n_l"]) + float(row["eps_l"]))
    fruit_value = instance.fruit_price_per_ton * float(row["farmer_quantity"])
    if e_ell + max(c.values()) >= fruit_value:
        raise RuntimeError(f"{routing_mode}: Assumption 1(f) fails (e_l + max c_f >= p)")

    cost_info = {
        "routing_mode": routing_mode,
        "truck_cost_per_m": instance.truck_cost_per_m,
        "c_f_min": min(c.values()),
        "c_f_mean": float(np.mean(list(c.values()))),
        "c_f_max": max(c.values()),
        "linearity_max_violation": max_violation,
        "assumption_1f_slack": fruit_value - e_ell - max(c.values()),
        "c_f": c,
    }
    return instance, high_ids, het_costs, cost_info


def coarse_grid(row: pd.Series) -> list[float]:
    upper = float(row["L"]) + 1
    return list(np.arange(0, upper + COARSE_STEP / 2, COARSE_STEP))


def predicted_grid(row: pd.Series) -> list[float]:
    points = []
    for x in [row["eps_h_minus"], row["eps_h_plus"], row["L"]]:
        if pd.notna(x):
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
    n_jobs = len(ROUTING_MODES) * len(windows)
    if not 0 <= job_id < n_jobs:
        raise SystemExit(f"JOB_ID must be in [0, {n_jobs - 1}], got {job_id}")

    # one job per (routing mode, CSV row): the first len(windows) jobs are the first mode,
    # each in instance order, so row_index matches exp_10's job_id
    routing_mode = ROUTING_MODES[job_id // len(windows)]
    row_index = job_id % len(windows)
    instance_id = int(windows.index[row_index])
    row = windows.loc[instance_id]

    results_path = Path("results") / "exp_11" / routing_mode / f"job_{job_id}"
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

    print(f"Building instance {instance_id} ({routing_mode} routing)...")
    instance, high_ids, het_costs, cost_info = build_instance(generator, instance_id, row, routing_mode)
    print(
        f"c_f in [{cost_info['c_f_min']:,.0f}, {cost_info['c_f_max']:,.0f}], "
        f"additivity violation {cost_info['linearity_max_violation']:.2e}, "
        f"Assumption 1(f) slack {cost_info['assumption_1f_slack']:,.0f}"
    )

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
        "experiment": "exp_11",
        "base_seed": BASE_SEED,
        "job_id": job_id,
        "routing_mode": routing_mode,
        "row_index": row_index,
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
        routing_mode=routing_mode,
        collection_costs=utils.encode_and_check(cost_info, label="collection_costs"),
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
