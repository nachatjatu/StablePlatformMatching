"""
What do the high types do with extra ambiguity budget? Worst-case deviations for exp_14.

For every exp_14 solve (instance, arm, side, eps_h) and every high type, evaluates the high
type's worst-case deviation at the solve's saved farmer payments with
`Optimizer.worst_case_deviation`: its profit, eta (the value of one more ton of budget), and
the routes it mixes. For every farmer on those routes it also records the economics that
decide whether swapping farmers in or out is worth anything:

  margin            fruit value minus the farmer's payment (what the deviator keeps before costs)
  standalone_trip   the farmer's own round trip from the mill (its c_f; with linear costs, what
                    adding the farmer costs any intermediary)
  insertion_cost    what adding the farmer to *this* route costs: route cost minus the cost of the
                    same route without the farmer
  net               margin - insertion_cost: the farmer's contribution to this deviation

With linear (star) costs, insertion_cost = standalone_trip, and the platform's payments absorb it
(payment = price - c_f - v_f), so every farmer the deviation could take nets the same v_f and a
swap is worth nothing. On a road network insertion_cost can be far below standalone_trip.

Payments are fixed at their saved values, so no platform solves are needed. Run from the repo
root (a few minutes):

    python experiments/exp_14/exp_14_deviations.py               # #5 and #29
    python experiments/exp_14/exp_14_deviations.py --instance 29

Saves to results/exp_14/deviations_{id}.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "exp_14"))

import exp_14  # noqa: E402
from stable_platform_matchings import InstanceGenerator, Optimizer  # noqa: E402
from stable_platform_matchings.optimization.options import OptimizerParams  # noqa: E402


def route_cost(instance, farmer_ids: list[str]) -> float:
    """Travel cost of collecting these farmers in one route, net of the truck fixed cost."""
    if not farmer_ids:
        return 0.0
    return instance.calculate_tree_path(sorted(farmer_ids)).cost - instance.truck_fixed_cost


def describe(instance, high_id: str, payments: dict[str, float], het_cost: float, route, weight: float):
    price = instance.fruit_price_per_ton
    owner = {f.id: f.intermediary_id for f in instance.farmers}
    quantity = {f.id: f.quantity for f in instance.farmers}
    ids = sorted(f.id for f in route.farmers)
    cost = route_cost(instance, ids)
    farmers = []
    for f in ids:
        margin = quantity[f] * price - payments[f]
        insertion = cost - route_cost(instance, [g for g in ids if g != f])
        farmers.append({
            "id": f, "own": owner[f] == high_id, "quantity": quantity[f], "payment": payments[f],
            "margin": margin, "standalone_trip": route_cost(instance, [f]),
            "insertion_cost": insertion, "net": margin - insertion,
        })
    return {
        "weight": weight,
        "own": sum(r["quantity"] for r in farmers if r["own"]),
        "out": sum(r["quantity"] for r in farmers if not r["own"]),
        "route_cost": cost,
        "value": route.value - het_cost - sum(payments[f] for f in ids),
        "farmers": farmers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instance", type=int, nargs="+", choices=sorted(exp_14.CASES), default=[5, 29])
    args = parser.parse_args()
    os.chdir(REPO)

    e = exp_14.exp_10
    generator = InstanceGenerator(
        e.FARMERS_FULL_CSV_PATH, e.FARMERS_14_CSV_PATH, e.INTERMEDIARIES_CSV_PATH,
        e.GRAPH_PKL_PATH, e.ALPHA_JSON_PATH, e.SIGMAS_JSON_PATH,
    )
    for instance_id in args.instance:
        start = time.time()
        rows: list[dict[str, Any]] = []
        for arm in exp_14.ARMS:
            instance, high_ids, het_costs, _ = exp_14.build(generator, instance_id, arm)
            optimizer = Optimizer(instance=instance, params=OptimizerParams(
                het_costs=het_costs, verbose=False, vrp_mode="approximate",
                vrp_time_limit_seconds=exp_14.CASES[instance_id].road.VRP_TIME_LIMIT_SECONDS))
            for side in exp_14.SIDES:
                path = exp_14.save_path(instance_id, arm, side)
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    payload = json.load(f)
                for run in sorted(payload["runs"], key=lambda r: r["eps_h"]):
                    primary = run["summary"]["platform_solve_result"]["primary_result"]
                    payments = primary["farmer_payments"]
                    for h in sorted(high_ids):
                        result = optimizer.worst_case_deviation(h, payments, run["eps_h"], exp_14.exp_10.HIST_SET_METHOD)
                        rows.append({
                            "instance": instance_id, "arm": arm, "side": side, "eps_h": run["eps_h"], "high_type": h,
                            "matched": h in primary["selected_intermediaries"],
                            "saved_profit": primary["intermediary_profits"][h],
                            "deviation_profit": result.profit, "eta": result.eta,
                            "saved_eta": primary["eta"][h], "stability_dual": primary["stability_duals"][h],
                            "routes": [describe(instance, h, payments, het_costs[h], route, w) for _, route, w in result.routes],
                        })
            print(f"#{instance_id} {arm}: done ({time.time() - start:.0f}s)")
        out = Path("results") / "exp_14" / f"deviations_{instance_id}.json"
        out.write_text(json.dumps({"instance_id": instance_id, "rows": rows}, indent=1))
        print(f"saved {len(rows)} deviations to {out}")


if __name__ == "__main__":
    main()
