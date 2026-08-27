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

BASE_SEED = 20260806
MIN_QUANTITY = 0.5
MAX_QUANTITY = 2.8
MIN_HET_COST = 200_000.0
MAX_HET_COST = 1_200_000.0
HIGH_PROB = 0.28
VRP_TIME_LIMIT_SECONDS = 900
EPSILON = 2
FRUIT_THRESH = 3.5

def sample_quantities(
    instance: Instance,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Perturb, round down, and clip each farmer's quantity."""
    quantities = {}

    for farmer in instance.farmers:
        random_quantity = rng.uniform(MIN_QUANTITY, MAX_QUANTITY)
        rounded_down = np.floor(random_quantity * 10.0) / 10.0
        clipped = np.clip(
            rounded_down,
            MIN_QUANTITY,
            MAX_QUANTITY,
        )

        quantities[farmer.id] = float(clipped)

    return quantities



def sample_het_costs(
    instance: Instance,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample heterogeneous costs for every intermediary."""
    return {
        intermediary.id: float(
            rng.uniform(MIN_HET_COST, MAX_HET_COST)
        )
        for intermediary in instance.intermediaries
    }

def set_relationships(
    instance: Instance,
    rng: np.random.Generator
):
    hist_matching = {
        intermediary.id: set() 
        for intermediary in instance.intermediaries
    }
    farmer_to_intermediary = {}
    farmer_quants = {
        farmer.id: farmer.quantity 
        for farmer in instance.farmers
    }
    unmatched_farmers = set(farmer_quants.keys())

    high_int = {
        intermediary.id for intermediary in instance.intermediaries 
        if rng.random() < HIGH_PROB
    }
    low_int = {
        intermediary.id for intermediary in instance.intermediaries 
        if intermediary.id not in high_int
    }

    intermediary_ids = [intermediary.id for intermediary in instance.intermediaries]

    while len(unmatched_farmers) > 0:
        sampled_farmer = str(rng.choice(sorted(unmatched_farmers)))
        sampled_int = str(rng.choice(sorted(intermediary_ids)))
        
        sum_fruit = sum(
            farmer_quants[f_id] for f_id in hist_matching[sampled_int]
        ) + farmer_quants[sampled_farmer]

        if (sampled_int in high_int) or (sampled_int in low_int and sum_fruit <= FRUIT_THRESH):
            hist_matching[sampled_int].add(sampled_farmer)
            farmer_to_intermediary[sampled_farmer] = sampled_int
            unmatched_farmers.remove(sampled_farmer)


    for farmer in instance.farmers:
        farmer.intermediary_id = farmer_to_intermediary[farmer.id]

    for intermediary in instance.intermediaries:
        intermediary.hist_sets = [frozenset(hist_matching[intermediary.id])]

    return hist_matching


def sample_locs(
    instance: Instance,
    rng: np.random.Generator,
    G
):  
    avail_locations = {
        str(node):(data['lat'], data['lon']) 
        for node, data in G.nodes(data=True) 
        if 'lat' in data and 'lon' in data
    }
    farmer_locations = {}
    for farmer in instance.farmers:
        node = rng.choice(list(avail_locations.keys()))
        farmer_locations[farmer.id] = node
        farmer.location = avail_locations[node]


def run_one(
    *,
    job_id: int,
    instance_path: Path,
    graph: Any,
    solver_threads: int,
) -> dict[str, Any]:
    """
    Run one reproducible experiment and return the InstanceSummary together
    with the information required to reproduce the sampled inputs.
    """

    print("Building instance...")
    seed_sequence = np.random.SeedSequence(
        [BASE_SEED, job_id]
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

    initial_instance = Instance.from_yaml(instance_path)

    quantities = sample_quantities(
        instance=initial_instance,
        rng=rng,
    )
    print("Loading instance...")
    instance = Instance.from_yaml(
        instance_path,
        force_quantities=quantities,
    )

    print("Sampling locations...")
    sample_locs(instance, rng, graph)

    print("Setting graph...")
    instance.set_graph(RoadGraph(graph))

    print("Setting relationships...")
    set_relationships(instance, rng)

    epsilons = {
        intermediary.id: float(EPSILON)
        for intermediary in instance.intermediaries
    }

    het_costs = sample_het_costs(
        instance=instance,
        rng=rng,
    )

    params = OptimizerParams(
        het_costs=het_costs,
        epsilons=epsilons,
        backend="gurobi",
        vrp_mode="approximate",
        vrp_time_limit_seconds=VRP_TIME_LIMIT_SECONDS,
        threads=solver_threads
    )

    print("Initializing optimizer...")
    optimizer = Optimizer(
        instance=instance,
        params=params,
    )

    options = SolverOptions(
        strategy="heuristic_accelerated",
        structured_farmer_payments=False,
        dominance_constraints=False,
        early_stop_threshold=float("inf"),
        hist_set_method="instance_farmers",
        pay_unmatched=False,
        seed=optimizer_seed,
        stabilize_final_solution=False
    )

    summary = optimizer.solve(options)

    return {
        "schema_version": 1,
        "metadata": {
            "optimizer_seed": optimizer_seed,
            "seed_sequence_state": seed_sequence.state,
            "sampling_seed_sequence_state": (
                sampling_seed_sequence.state
            ),
            "optimizer_seed_sequence_state": (
                optimizer_seed_sequence.state
            ),
            "instance_file": instance_path.name,
        },
        "sampled_inputs": {
            "quantities": quantities,
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
        (path for path in instances_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".yaml", ".yml"}
        and path.name.startswith("aggregate")),
        key=lambda x: int(x.name.split("_")[2].split(".")[0])
    )

    if not instance_paths:
        raise FileNotFoundError(f"No YAML instance files found in {instances_path}")

    # make results path
    results_path = Path("results") / "exp_4_heuristic" / f"job_{job_id}"
    results_path.mkdir(parents=True, exist_ok=True)

    with graph_path.open("rb") as file:
        graph = pickle.load(file)

    solver_threads = utils.get_solver_threads()

    experiment_metadata = {
        "experiment": "exp_4_heuristic",
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
            "min_het_cost": MIN_HET_COST,
            "max_het_cost": MAX_HET_COST,
            "vrp_time_limit_seconds": (
                VRP_TIME_LIMIT_SECONDS
            ),
        },
    }

    instance_path = instance_paths[job_id%14]

    save_path = results_path / f"job_{job_id}.json.gz"

    job_payload: dict[str, Any] = {
        "schema_version": 1,
        "job_id": job_id,
        "experiment_metadata": experiment_metadata,
        "completed": False
    }

    # pre-save
    utils.save_json_gz_atomic(
        payload=job_payload,
        save_path=save_path,
    )

    run_payload = run_one(
        job_id=job_id,
        instance_path=instance_path,
        graph=graph,
        solver_threads=solver_threads
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
    job_payload["run"] = safe_run_payload
    job_payload["completed"] = True
    utils.save_json_gz_atomic(
        payload=job_payload,
        save_path=save_path,
    )

    print(
        f"Saved run to {save_path}"
    )


if __name__ == "__main__":
    main()