"""
Worst-case deviations of the matched high types past saturation, for exp_11_results.ipynb.

Rebuilds the double-switchback instances (#5, #29) exactly as their exp_10 (road) and
exp_11 (star, linear c_f) jobs did, and, for every saved solve above L where a high type is matched,
evaluates that high type's worst-case deviation at the saved farmer payments with
`Optimizer.worst_case_deviation`. Its profit should equal the saved intermediary profit, its `eta` is that
profit's slope in eps_h, and its routes show the swap that makes extra budget valuable.

Run as a script (`python deviation_analysis.py`, about 20 seconds) so the instance generator,
which loads the full road network, stays out of the notebook kernel. Saves to
results/exp_11/deviations.json.
"""
from __future__ import annotations

import contextlib
import gzip
import json
import sys
from pathlib import Path
from typing import Any

from stable_platform_matchings import InstanceGenerator, Optimizer
from stable_platform_matchings.optimization.options import OptimizerParams

REPO = Path(__file__).resolve().parent.parent
for exp in ["exp_10", "exp_11"]:
    sys.path.insert(0, str(REPO / "experiments" / exp))

INSTANCES = (5, 29)
SOURCES = {"road": REPO / "results" / "exp_10", "star": REPO / "results" / "exp_11" / "star"}
OUTPUT = REPO / "results" / "exp_11" / "deviations.json"
HIST_SET_METHOD = "instance_farmers"   # as in exp_10 / exp_11
SWAP_CASES = [("road", 5, "", 5.35), ("road", 29, "competent", 6.35)]


def rebuild_instance(generator, arm: str, instance_id: int):
    """Rebuild an exp_10 (road) or exp_11 (zero / star) instance exactly as its job did."""
    import exp_10
    import exp_11

    # the experiment scripts use paths relative to the repo root
    with contextlib.chdir(REPO):
        if arm == "road":
            windows = exp_10.load_windows()
            return exp_10.build_instance(generator, instance_id, windows.loc[instance_id])
        windows = exp_11.load_windows()
        instance, high_ids, het_costs, _ = exp_11.build_instance(generator, instance_id, windows.loc[instance_id], arm)
        return instance, high_ids, het_costs


def describe_route(route, high_id, payments, het_cost, fixed_cost) -> dict[str, Any]:
    farmers = sorted(f.id for f in route.farmers)
    return {
        "farmers": farmers,
        "intermediary": sum(f.quantity for f in route.farmers if f.intermediary_id == high_id),
        "out": sum(f.quantity for f in route.farmers if f.intermediary_id != high_id),
        "route_cost": route.cost - fixed_cost,
        "value": route.value - het_cost - sum(payments[f] for f in farmers),
    }


def swap_details(instance, result, payments, het_cost) -> dict[str, Any]:
    """The two routes the worst case mixes, and the farmers swapped between them."""
    fixed, price = instance.truck_fixed_cost, instance.fruit_price_per_ton
    h = result.intermediary_id
    routes = sorted((describe_route(route, h, payments, het_cost, fixed) for _, route, _ in result.routes),
                    key=lambda r: r["out"])
    A, B = routes[0], routes[-1]
    intermediary = {f.id: f.intermediary_id for f in instance.farmers}

    def farmer(f):
        return {"id": f, "intermediary": intermediary[f] == h, "payment": payments[f], "margin": price - payments[f],
                "standalone_trip": instance.calculate_tree_path([f]).cost - fixed}

    return {
        "high_type": h, "eps_h": result.epsilon, "eta": result.eta,
        "A": {k: A[k] for k in ["intermediary", "out", "route_cost", "value"]},
        "B": {k: B[k] for k in ["intermediary", "out", "route_cost", "value"]},
        "dropped": [farmer(f) for f in sorted(set(A["farmers"]) - set(B["farmers"]))],
        "added": [farmer(f) for f in sorted(set(B["farmers"]) - set(A["farmers"]))],
    }


def reconstruct() -> dict[str, Any]:
    data = REPO / "data"
    generator = InstanceGenerator(
        data / "farmers.csv", data / "farmers_14.csv", data / "intermediaries.csv",
        data / "graph_0-14960_00_new.pickle", data / "precomputed_alpha.json", data / "precomputed_sigmas.json",
    )
    rows, swaps = [], []
    for arm, root in SOURCES.items():
        # open solution files
        for file in sorted(root.rglob("*.json.gz")):
            with gzip.open(file, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            # get only instances we care about
            i = payload["instance_id"]
            if i not in INSTANCES:
                continue

            # rebuild those instances
            instance, high_ids, het_costs = rebuild_instance(generator, arm, i)
            if high_ids != set(payload["high_ids"]):
                # sanity check
                raise RuntimeError(f"rebuilt {arm} #{i} has different high types")

            # initialize optimizer
            optimizer = Optimizer(instance=instance, params=OptimizerParams(
                het_costs=het_costs, verbose=False, vrp_mode="approximate", vrp_time_limit_seconds=900))
            L = payload["window"]["K"] - payload["window"]["n_h"]

            solved = {}
            for run in payload["runs"]:
                # filter for eps_h near the threshold
                if run["eps_h"] <= L:
                    continue
                primary = run["summary"]["platform_solve_result"]["primary_result"]
                payments = primary["farmer_payments"]
                # for each matched high-type, compute their worst-case deviation
                for h in sorted(high_ids & set(primary["selected_intermediaries"])):
                    result = optimizer.worst_case_deviation(h, payments, run["eps_h"], HIST_SET_METHOD)
                    solved[(h, run["eps_h"])] = (result, payments)
                    rows.append({
                        "instance": i, "arm": arm, "high_type": h, "eps_h": run["eps_h"],
                        "saved_profit": primary["intermediary_profits"][h],
                        "reconstructed_deviation": result.profit, "eta": result.eta,
                        "worst_case_mix": [
                            {"weight": p, **{k: v for k, v in describe_route(
                                route, h, payments, het_costs[h], instance.truck_fixed_cost).items() if k != "farmers"}}
                            for _, route, p in result.routes
                        ],
                    })
            for case_arm, case_i, prefix, eps_target in SWAP_CASES:
                if (case_arm, case_i) == (arm, i):
                    h, e = min((k for k in solved if k[0].startswith(prefix)), key=lambda k: abs(k[1] - eps_target))
                    result, payments = solved[(h, e)]
                    swaps.append({"instance": i, "arm": arm, **swap_details(instance, result, payments, het_costs[h])})
            print(f"{arm} #{i}: {sum(1 for r in rows if (r['arm'], r['instance']) == (arm, i))} matched high-type points above L")
    return {"deviations": rows, "swaps": swaps}


if __name__ == "__main__":
    result = reconstruct()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=1))
    print(f"saved {len(result['deviations'])} deviations and {len(result['swaps'])} swaps to {OUTPUT}")
