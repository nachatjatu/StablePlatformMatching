"""
Swap values for the matched high types' worst-case deviations in exp_14.

For every exp_14 inefficient-side solve (instance, arm, eps_h) and matched high type, take the
base route S of its worst-case deviation (from exp_14_deviations.py: the full-truck route with the
most of its own farmers) and, at the solve's saved payments, evaluate every one-for-one swap of an
own farmer f on S for an outside farmer g not on S:

  swap value(f -> g) = (margin_g - margin_f) - [c(S - f + g) - c(S)]

with margin = fruit value - payment and c the route cost on the instance's graph (net of the truck
fixed cost). Whole routes are compared, so no per-farmer cost split is involved. A positive swap
value is what one more ton of budget can buy; with linear (star) costs every swap value is 0.

Run from the repo root after exp_14_deviations.py (seconds):

    python experiments/exp_14/exp_14_swaps.py

Saves to results/exp_14/swaps_{id}.json.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "exp_14"))

import exp_14  # noqa: E402
from stable_platform_matchings import InstanceGenerator  # noqa: E402

INSTANCES = (5, 29)


def main() -> None:
    os.chdir(REPO)
    e = exp_14.exp_10
    generator = InstanceGenerator(
        e.FARMERS_FULL_CSV_PATH, e.FARMERS_14_CSV_PATH, e.INTERMEDIARIES_CSV_PATH,
        e.GRAPH_PKL_PATH, e.ALPHA_JSON_PATH, e.SIGMAS_JSON_PATH,
    )
    for instance_id in INSTANCES:
        devs = json.loads((Path("results") / "exp_14" / f"deviations_{instance_id}.json").read_text())["rows"]
        rows = []
        for arm in exp_14.ARMS:
            instance, high_ids, het_costs, _ = exp_14.build(generator, instance_id, arm)
            fixed, price = instance.truck_fixed_cost, instance.fruit_price_per_ton
            owner = {f.id: f.intermediary_id for f in instance.farmers}
            quantity = {f.id: f.quantity for f in instance.farmers}
            cache: dict[frozenset[str], float] = {}

            def cost(ids) -> float:
                key = frozenset(ids)
                if key not in cache:
                    cache[key] = instance.calculate_tree_path(sorted(key)).cost - fixed
                return cache[key]

            with gzip.open(exp_14.save_path(instance_id, arm, "inefficient"), "rt", encoding="utf-8") as fh:
                solves = {round(r["eps_h"], 6): r for r in json.load(fh)["runs"]}
            for d in devs:
                if d["arm"] != arm or d["side"] != "inefficient" or not d["matched"]:
                    continue
                h = d["high_type"]
                payments = solves[round(d["eps_h"], 6)]["summary"]["platform_solve_result"]["primary_result"]["farmer_payments"]
                margin = {f: quantity[f] * price - payments[f] for f in payments}
                base = max(d["routes"], key=lambda rt: (rt["own"], -rt["out"], rt["weight"]))
                S = {f["id"] for f in base["farmers"]}
                own_on_route = sorted(f for f in S if owner[f] == h)
                outside_off_route = sorted(f for f in owner if owner[f] != h and f not in S)
                c_S = cost(S)
                swaps = []
                for f in own_on_route:
                    for g in outside_off_route:
                        if quantity[g] != quantity[f]:
                            continue                      # keep the truck exactly full
                        delta_cost = cost(S - {f} | {g}) - c_S
                        swaps.append({
                            "drop": f, "add": g, "margin_gain": margin[g] - margin[f],
                            "route_cost_change": delta_cost,
                            "swap_value": margin[g] - margin[f] - delta_cost,
                            # the same swap with linear costs at these payments: c_f is the standalone trip
                            "linear_cost_change": cost({g}) - cost({f}),
                        })
                # full routes for each own farmer's slot: the other farmers on S plus one candidate for that slot
                # (the own farmer itself, or an outside farmer not on S), with the whole route's cost and value
                slots = []
                for f in own_on_route:
                    common = S - {f}
                    for g in [f] + [g for g in outside_off_route if quantity[g] == quantity[f]]:
                        ids = common | {g}
                        route_cost = cost(ids)
                        slots.append({
                            "slot_of": f, "candidate": g, "own": owner[g] == h, "payment": payments[g],
                            "margin": margin[g], "shared_farmers": sorted(common), "route_cost": route_cost,
                            "route_value": sum(margin[x] for x in ids) - route_cost - fixed - het_costs[h],
                        })
                rows.append({
                    "instance": instance_id, "arm": arm, "eps_h": d["eps_h"], "high_type": h, "eta": d["eta"],
                    "base_route": sorted(S), "base_own": base["own"], "base_out": base["out"], "base_cost": c_S,
                    "swaps": swaps, "slots": slots,
                })
            print(f"#{instance_id} {arm}: {sum(len(r['swaps']) for r in rows if r['arm'] == arm)} swaps evaluated")
        out = Path("results") / "exp_14" / f"swaps_{instance_id}.json"
        out.write_text(json.dumps({"instance_id": instance_id, "rows": rows}, indent=1))
        print(f"saved {len(rows)} base routes to {out}")


if __name__ == "__main__":
    main()
