import gurobipy as gp

from ...domain.instance import Instance
from ...domain.matching import Matching
from ...domain.route import Route


class GurobiTSPSolver:
    """Gurobi-based solver for the prize-collecting TSP defined on the tree.

    The solver takes a prize for each farmer and maximizes prize minus the
    twice-traversed edge costs and fixed truck cost.  The returned route is
    constructed using :meth:`Instance.calculate_tree_path`.

    Attributes:
        instance (Instance): current platform instance.
    """

    OBJ_TOL = 1.0
    SEL_TOL = 0.5

    def __init__(self, instance: Instance) -> None:
        """Initialize with a platform :class:`Instance`.
        Args:
            instance (Instance): current platform instance.
        """
        self.instance = instance

    def solve(
        self, 
        prizes: dict[str, float], 
        time_limit_seconds: int, 
        threads: int
    ) -> tuple[list[Route], list[float]]:
        """Solve the prize-collecting TSP.

        Args:
            prizes (dict[str, float]): prize value associated with each farmer, indexed by ID.
                `prizes[f]` is the prize associated with farmer `f`.
            time_limit_seconds (int): time limit for Gurobi, in seconds.
            threads (int): how many threads to use for Gurobi.

        Raises:
            RuntimeError: graph tree not initialized.
            ValueError: prizes length mismatch.
            RuntimeError: optimization found no solution.
            ValueError: objective mismatch.

        Returns:
            Route: best route found by the model.
            float: objective value corresponding to the route.
        """
        if self.instance.tree is None:
            raise RuntimeError("graph tree not initialized.")
        if len(prizes) != len(self.instance.farmers):
            raise ValueError(
                f"Prizes length mismatch: {len(prizes)} != {len(self.instance.farmers)}"
            )

        # set Gurobi params
        model = gp.Model("TSP")
        model.setParam("Threads", threads)
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", time_limit_seconds)

        # add binary variables for each farmer (to use or not)
        farmer_ids = [farmer.id for farmer in self.instance.farmers]
        farmer_vars = model.addVars(farmer_ids, vtype=gp.GRB.BINARY, name="visit")

        # add a continuous variable for the used truck
        used = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name="truck_used")

        # add continuous variables for each edge in the tree
        edge_vars = model.addVars(
            self.instance.tree_edges,  # type: ignore[call-overload]
            vtype=gp.GRB.CONTINUOUS,
            lb=0.0,
            name="edge",
        )

        # make sure that all edges of a node are transversed
        for farmer in self.instance.farmers:
            for edge in self.instance.root_edges[farmer.id]:
                model.addConstr(edge_vars[edge] >= farmer_vars[farmer.id])

        # make sure that if any farmer is picked up, then used is equal to one
        for farmer in self.instance.farmers:
            model.addConstr(used >= farmer_vars[farmer.id], "farmer_used")

        # make sure that at least one farmer is picked up
        model.addConstr(
            gp.quicksum(farmer_vars[farmer.id] for farmer in self.instance.farmers) >= 1,
            "at_least_one_pickup",
        )

        # add capacity constraint
        model.addConstr(
            gp.quicksum(
                farmer_vars[farmer.id] * farmer.quantity for farmer in self.instance.farmers
            )
            <= self.instance.truck_capacity_tons,
            "capacity",
        )

        # objective: maximize profit = total prize minus fixed and travel costs
        total_prize = gp.quicksum(
            prizes[farmer.id] * farmer_vars[farmer.id] for farmer in self.instance.farmers
        )

        edge_costs = gp.quicksum(
            edge_vars[edge]
            * self.instance.tree[edge[0]][edge[1]]["weight"]
            * self.instance.truck_cost_per_m
            for edge in self.instance.tree_edges
        )

        model.setObjective(
            total_prize - 2 * edge_costs - used * self.instance.truck_fixed_cost, gp.GRB.MAXIMIZE
        )

        model.optimize()

        self._require_solution(model, "TSP")

        objective = model.ObjVal

        # extract the matching
        selected_farmers = [
            farmer.id for farmer in self.instance.farmers
            if farmer_vars[farmer.id].X > GurobiTSPSolver.SEL_TOL
        ]
        route = self.instance.calculate_tree_path(selected_farmers)

        # verify that the objectives correspond
        total_prize = sum(prizes[farmer_id] for farmer_id in selected_farmers)
        alt_objective = total_prize - route.cost
        # check if the objective matches
        if abs(objective - alt_objective) > GurobiTSPSolver.OBJ_TOL:
            raise ValueError(f"Objective mismatch: {objective} != {alt_objective}")

        return [route], [objective]

    @staticmethod
    def _require_solution(model: gp.Model, context: str) -> None:
        if model.SolCount == 0:
            raise RuntimeError(f"{context} produced no feasible solution; status={model.Status}.")


class GurobiVRPSolver:
    """Gurobi-based solver for the vehicle routing problem (VRP).

    This formulation selects a set of farmers for each truck (int)
    respecting capacity and attempts to minimize total routing and fixed costs.
    It also handles lower- and upper-bounds on the number of vehicles.

    Attributes:
        instance (Instance): the current platform instance.
    """

    OBJ_TOL = 1.0
    SEL_TOL = 0.5

    def __init__(self, instance: Instance) -> None:
        """
        Initialize a VRP solver for the given platform instance.

        Args:
            instance (Instance): the current platform instance.
        """
        self.instance = instance

    def solve(
        self, 
        n_vehicles_lower_bound: int, 
        n_vehicles_upper_bound: int, 
        threads: int, 
        time_limit_seconds: int | float = 4 * 60 * 60
    ) -> Matching:
        """
        Solve the VRP with a bound on number of vehicles.

        Args:
            n_vehicles_lower_bound (int): minimum number of trucks to use.
            n_vehicles_upper_bound (int): maximum number of trucks available.
            time_limit_seconds (int, optional): time limit for VRP solver.
                defaults to 4 hours.

        Raises:
            RuntimeError: graph tree not initialized.
            RuntimeError: VRP optimization produced no solution.
            ValueError: objective mismatch.

        Returns:
            Matching: a Matching object describing the chosen routes.
        """
        if self.instance.tree is None:
            raise RuntimeError("graph tree not initialized.")

        # Gurobi params
        model = gp.Model("VRP")
        model.setParam("Threads", threads)
        model.setParam("TimeLimit", time_limit_seconds)

        # add binary variables for each farmer and intermediary
        farmer_ids = [farmer.id for farmer in self.instance.farmers]
        intermediary_ids = list(range(n_vehicles_upper_bound))

        # add binary variables for each truck and each farmer
        matching_vars = model.addVars(
            intermediary_ids, farmer_ids, vtype=gp.GRB.BINARY, name="visit"
        )

        # add continuous variables for each edge in the tree and each intermediary
        edge_vars = model.addVars(
            intermediary_ids,
            list(range(len(self.instance.tree_edges))),
            vtype=gp.GRB.CONTINUOUS,
            lb=0.0,
            ub=1.0,
            name="edge",
        )
        edge_to_index = {edge: index for index, edge in enumerate(self.instance.tree_edges)}

        # add continuous variables for each used intermediary
        used = model.addVars(
            intermediary_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=1.0, name="int_used"
        )

        # make sure that all edges of a node are transversed
        for intermediary_id in intermediary_ids:
            for farmer in self.instance.farmers:
                for edge in self.instance.root_edges[farmer.id]:
                    model.addConstr(
                        edge_vars[intermediary_id, edge_to_index[edge]]
                        >= matching_vars[intermediary_id, farmer.id]
                    )

        # make sure that if a truck picks up a farmer, then it is used
        for intermediary_id in intermediary_ids:
            model.addConstrs(
                (
                    used[intermediary_id] >= matching_vars[intermediary_id, farmer_id]
                    for farmer_id in farmer_ids
                ),
                f"used_lower_{intermediary_id}",
            )

        # make sure that if a truck is used, then it picks up at least one farmer
        for intermediary_id in intermediary_ids:
            model.addConstr(
                gp.quicksum(
                    matching_vars[intermediary_id, farmer_id]
                    for farmer_id in self.instance.farmer_by_id
                )
                >= used[intermediary_id],
                f"used_upper_{intermediary_id}",
            )

        # make sure that at least n_vehicles_lower_bound trucks are used
        model.addConstr(
            gp.quicksum(used[intermediary_id] for intermediary_id in intermediary_ids)
            >= n_vehicles_lower_bound,
            "num_vehicles",
        )

        # make sure that each farmer is picked up by exactly one truck
        model.addConstrs(
            (
                gp.quicksum(
                    matching_vars[intermediary_id, farmer_id]
                    for intermediary_id in intermediary_ids
                )
                == 1
                for farmer_id in self.instance.farmer_by_id
            ),
            "one_truck",
        )

        # add capacity constraint for each truck
        for intermediary_id in intermediary_ids:
            model.addConstr(
                gp.quicksum(
                    matching_vars[intermediary_id, farmer.id] * farmer.quantity
                    for farmer in self.instance.farmers
                )
                <= self.instance.truck_capacity_tons,
                "capacity",
            )

        # add an ordering of used trucks to break symmetry
        for intermediary_id in intermediary_ids[1:]:
            model.addConstr(
                used[intermediary_id] <= used[intermediary_id - 1], f"order_{intermediary_id}"
            )

        # objective: minimize the total cost
        edge_costs = gp.quicksum(
            2
            * edge_vars[intermediary_id, edge_index]
            * self.instance.tree[edge[0]][edge[1]]["weight"]
            * self.instance.truck_cost_per_m
            for intermediary_id in intermediary_ids
            for edge_index, edge in enumerate(self.instance.tree_edges)
        )

        truck_costs = gp.quicksum(
            used[intermediary_id] * self.instance.truck_fixed_cost
            for intermediary_id in intermediary_ids
        )

        model.setObjective(edge_costs + truck_costs, gp.GRB.MINIMIZE)

        model.optimize()

        self._require_solution(model, "VRP")

        total_cost = model.ObjVal

        # extract the matching
        routes = []
        for intermediary_id in intermediary_ids:
            selected_farmers = [
                farmer.id
                for farmer in self.instance.farmers
                if matching_vars[intermediary_id, farmer.id].X > GurobiVRPSolver.SEL_TOL
            ]

            if len(selected_farmers) > 0:
                route = self.instance.calculate_tree_path(selected_farmers)
                routes.append(route)

        matching = Matching(self.instance, routes)
        alt_cost = matching.cost

        # check if the objective match
        print("Gap", total_cost, alt_cost)
        if abs(total_cost - alt_cost) > GurobiVRPSolver.OBJ_TOL:
            raise ValueError(f"Objective mismatch: {total_cost} != {alt_cost}")

        return matching

    @staticmethod
    def _require_solution(model: gp.Model, context: str) -> None:
        if model.SolCount == 0:
            raise RuntimeError(f"{context} produced no feasible solution; status={model.Status}.")