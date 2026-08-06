import numpy as np
import time
from collections.abc import Mapping
from dataclasses import fields
from itertools import combinations
from pprint import pformat

import gurobipy as gp

from ..domain.instance import Instance
from ..reporting.containers import OptimizationResult, PlatformOutcome, BranchPrimalResult, BranchDualResult, InstanceSummary
from ..reporting.printer import Printer
from .branch import Branch
from .options import OptimizerParams, SolverOptions
from .solvers.dynamic_solvers import DynamicTSPSolver
from .solvers.lp_solvers import GurobiVRPSolver
from .solvers.runtime import require_solution
from .solvers.search_strategies import solve_exact, solve_heuristic

IntermediaryId = str
FarmerId = str
IntermediarySelection = frozenset[IntermediaryId]
RouteKey = frozenset[FarmerId]


class Optimizer:
    """Main optimization engine implementing branch-and-price for the
    platform model.

    The optimizer maintains a catalogue of intermediary sets, solves primal and dual
    LPs, and handles branching decisions.  It exposes ``solve`` methods for
    heuristic and exact modes.
    """

    BRANCH_PRUNE_TOL = 1.0
    PRIMARY_OBJECTIVE_TOL = 1.0
    GLOBAL_LB_UPDATE_TOL = 1e-9

    RANDOM_BRANCH_TOL = 1.0
    CUT_TOL = 1.0
    COLUMN_TOL = 1.0
    INT_TOL = 1e-9

    N_MATCHINGS = 20
    PAVED_THRESHOLD = 105000
    DIRT_THRESHOLD = 46000

    def __init__(
        self,
        instance: Instance,
        params: OptimizerParams,
    ) -> None:
        """
        Summary

        Args:
            instance (Instance): _description_
            parameters (dict[str, object]): _description_
            base_matchings (dict[int, Matching], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """

        params.validate(instance)

        self.params = params
        self.output = Printer(width=params.print_width, enabled=params.verbose)

        self._print_optimizer_params()

        self.instance = instance

        self.n_farmers = len(self.instance.farmers)
        self.n_intermediaries = len(self.instance.intermediaries)

        self.farmer_ids = [farmer.id for farmer in self.instance.farmers]
        self.intermediary_ids = [intermediary.id for intermediary in self.instance.intermediaries]

        self.intermediary_id_to_capacity = {
            intermediary.id: intermediary.capacity for intermediary in self.instance.intermediaries
        }

        self.vrp_solver = GurobiVRPSolver(self.instance)
        self.tsp_solver = DynamicTSPSolver(self.instance)

        self._original_hist_sets = {
            intermediary.id: (
                tuple(frozenset(hist_set) for hist_set in intermediary.hist_sets) or (frozenset(),)
            )
            for intermediary in self.instance.intermediaries
        }
        self._configure_hist_sets(aggregate=False)

        self.routing_cost_by_truck_count = self._initialize_routing_cost_by_truck_count()

        self.min_trucks = min(self.routing_cost_by_truck_count)
        self.max_trucks = max(self.routing_cost_by_truck_count)

        self.route_by_farmer_ids_set = {}

        self.time_usage = {
            "tsp": 0.0,
            "adding_constraints_start": 0.0,
            "adding_constraints_found": 0.0,
            "solving": 0.0,
            "callback": 0.0,
            "total": 0.0,
        }

        self.best_lb_result: BranchPrimalResult | None = None
        self.best_lb_set: frozenset[str] | None = None

        initial_set_to_cost = self._initialize_intermediary_set_to_cost()

        self._initial_intermediary_set_to_cost = initial_set_to_cost.copy()
        self.intermediary_set_to_cost = initial_set_to_cost.copy()

        self.dominance_relations: list[tuple[str, str]] = []
        self.best_lb, self.best_ub = -float("inf"), float("inf")
        self.rng = None

        self.total_oracle_calls = 0
        self.oracle_calls = []

    def solve(self, options: SolverOptions) -> InstanceSummary:
        """
        Public entrypoint to run optimization with a given strategy.

        Args:

        Raises:
            ValueError: unknown solver strategy.
            ValueError: optimization fails to find a feasible solution.

        Returns:
            InstanceSummary: summary containing solve time, various solutions,
                and # of oracle calls
        """

        self.options = options

        self._print_solver_options()

        if self.options.structured_farmer_payments:
            self._verify_farmer_distances()

        self._configure_hist_sets(self.options.aggregate)
        self.dominance_relations = (
            self._calc_dominance() if self.options.dominance_constraints else []
        )

        self.rng = np.random.default_rng(options.seed)

        self.intermediary_set_to_cost = self._initial_intermediary_set_to_cost.copy()
        self.route_by_farmer_ids_set = {}

        # reset solver state
        self.best_lb, self.best_ub = -float("inf"), float("inf")
        self.best_lb_summary = None
        self.best_lb_set = None
        self.total_oracle_calls = 0
        self.oracle_calls = []

        self.time_usage = {
            "tsp": 0.0,
            "adding_constraints_start": 0.0,
            "adding_constraints_found": 0.0,
            "solving": 0.0,
            "callback": 0.0,
            "total": 0.0,
        }

        # construct instance summary
        self.instance_summary = InstanceSummary(
            instance=self.instance, 
            params=self.params, 
            strategy=self.options.strategy
        )

        self.output.section(f"Strategy: {self.options.strategy}")
        self.output.metric("Farmers", self.n_farmers, precision=0)
        self.output.metric("Intermediaries", self.n_intermediaries, precision=0)

        # solve using specified search strategy
        if self.options.strategy == "exact":
            best_lb_result = solve_exact(self)
        elif self.options.strategy == "heuristic_unoptimized":
            best_lb_result = solve_heuristic(self, optimize=False)
        else:
            best_lb_result = solve_heuristic(self, optimize=True)

        # raise error if optimization fails to find a solution
        if best_lb_result is None:
            raise RuntimeError("Optimization finished without finding a feasible incumbent.")

        # record solution and print details
        self.instance_summary.platform_solve_result = best_lb_result
        self.instance_summary.total_time = time.time() - self.instance_summary.start_time
        self.instance_summary.total_oracle_calls = self.total_oracle_calls

        self.output.section("Solver Complete", fill="=")
        self.output.metric("Total time (seconds)", self.instance_summary.total_time)
        self.output.metric("Oracle calls", self.total_oracle_calls, precision=0)
        self.output.metric("Best lower bound", self.best_lb)
        self.output.metric("Best upper bound", self.best_ub)

        return self.instance_summary

    def initialize_branch(self, branch: Branch) -> bool:
        """
        Initializes a given branch, returning True if successful and False otherwise.

        Args:
            branch (Branch): branch to be initialized.

        Returns:
            bool: indicator of whether initialization was successful. False if branch restrictions
                admit no feasible intermediary set and branch should be pruned.
        """
        branch.print_(self.output)

        # compute min cost set of intermediaries
        zero_prizes = {intermediary_id: 0.0 for intermediary_id in self.intermediary_ids}

        result = self._get_best_intermediary_set(zero_prizes, branch)

        if result is None:
            self.output.status(
                "Branch restrictions admit no feasible intermediary set; pruning branch."
            )
            return False

        min_cost_set, _, min_cost = result

        # assign min cost results to branch, adding to intermediary_set_to_cost if needed
        branch.min_cost_set = min_cost_set
        branch.min_cost = min_cost

        if min_cost_set not in self.intermediary_set_to_cost:
            self.intermediary_set_to_cost[min_cost_set] = min_cost

        return True

    def solve_primal_for_branch(self, branch: Branch, sol_type: str) -> BranchPrimalResult:
        """
        Solve the primal LP under given branch and solution type in-place.

        `sol_type` may be `exact`, `forced_lower_bound` or
        `forced_upper_bound' and controls additional constraints.
        The LP is re-optimized to find optimal solutions that maximize either
        farmer or intermediary welfare.

        Args:
            branch (Branch): branch whose primal LP is to be solved.
            sol_type (str): controls which constraints are imposed.

        Returns:
            PrimalSolution: summarizes the solution including profits and cut information.
        """

        # first get all valid intermediary selections
        # consistent with branch forced match/unmatch restrictions
        valid_intermediary_sets = [
            intermediary_set
            for intermediary_set in self.intermediary_set_to_cost
            if self._is_valid_intermediary_set(branch, intermediary_set)
        ]
        n_valid_intermediary_sets = len(valid_intermediary_sets)

        # create a new model
        model = gp.Model("Primal")
        model.setParam("Threads", self.params.threads)
        model.setParam("OutputFlag", 0)

        # create payment variables for each farmer and intermediary
        farmer_payment_vars = model.addVars(
            self.farmer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="farmer_payment"
        )

        intermediary_profit_vars = model.addVars(
            self.intermediary_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="intermediary_profit"
        )

        # create variables to indicate if intermediary set is used
        intermediary_set_prob_vars = model.addVars(
            list(range(n_valid_intermediary_sets)),
            vtype=gp.GRB.CONTINUOUS,
            lb=0.0,
            ub=1.0,
            name="intermediary_set_probs",
        )

        # create variables for stability constraints
        eta = model.addVars(self.intermediary_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="eta")
        kappa = model.addVars(
            [
                (intermediary.id, hist_set_index)
                for intermediary in self.instance.intermediaries
                for hist_set_index in range(len(self.active_hist_sets[intermediary.id]))
            ],
            vtype=gp.GRB.CONTINUOUS,
            lb=-float("inf"),
            name="kappa",
        )

        # constraint 1: sum of set selection probabilities equal one
        model.addConstr(
            gp.quicksum(intermediary_set_prob_vars[k] for k in range(n_valid_intermediary_sets))
            == 1,
            "set_probs_sum",
        )

        # constraints 2, 3: forced matched/unmatched intermediaries must be matched/unmatched
        intermediary_matched = {}
        for intermediary_id in self.intermediary_ids:
            intermediary_matched[intermediary_id] = gp.quicksum(
                intermediary_set_prob_vars[k]
                for k in range(n_valid_intermediary_sets)
                if intermediary_id in valid_intermediary_sets[k]
            )

        for intermediary_id in branch.forced_match:
            model.addConstr(
                intermediary_matched[intermediary_id] == 1,
                f"intermediary_matched_{intermediary_id}",
            )

        for intermediary_id in branch.forced_unmatch:
            model.addConstr(
                intermediary_matched[intermediary_id] == 0,
                f"intermediary_unmatched_{intermediary_id}",
            )

        # constraint 4: pay intermediaries at most their max. capacity * fruit price = max. value
        if sol_type in ["exact", "forced_lower_bound"]:
            if not self.options.pay_unmatched:
                for intermediary in self.instance.intermediaries:
                    model.addConstr(
                        intermediary_profit_vars[intermediary.id]
                        <= intermediary_matched[intermediary.id]
                        * intermediary.capacity
                        * self.instance.fruit_price_per_ton
                    )
            else:
                for intermediary in self.instance.intermediaries:
                    model.addConstr(
                        intermediary_profit_vars[intermediary.id]
                        <= intermediary.capacity * self.instance.fruit_price_per_ton
                    )

        # constraint 5: if applicable, do not pay unmatched intermediaries
        if not self.options.pay_unmatched:
            model.addConstrs(
                intermediary_profit_vars[intermediary_id] <= 0
                for intermediary_id in branch.forced_unmatch
            )

        # constraint 6: if applicable, force to use min cost set
        if sol_type in ["forced_upper_bound", "forced_lower_bound"]:
            index_min_cost_set = valid_intermediary_sets.index(branch.min_cost_set)
            model.addConstr(intermediary_set_prob_vars[index_min_cost_set] == 1, "forced_selection")

        # constraint 7: constrain kappa to be the max deviation
        #   opportunity from historical set (for subset)
        for route in self.route_by_farmer_ids_set:
            self._add_stability_cuts_for_route(
                model, eta, kappa, farmer_payment_vars, self.route_by_farmer_ids_set[route]
            )

        # constraint 8: payment must be >= avg worst-case deviation
        #   opportunity from hist set + robust premium
        epsilons = self.params.epsilons
        for intermediary in self.instance.intermediaries:
            # note that constraint 7 makes kappa equal to max deviation
            #   opportunity from historical set
            n_hist_sets = len(self.active_hist_sets[intermediary.id])
            avg_deviation_payoff = (
                1
                / n_hist_sets
                * gp.quicksum(
                    kappa[intermediary.id, hist_set_index] for hist_set_index in range(n_hist_sets)
                )
            )
            model.addConstr(
                intermediary_profit_vars[intermediary.id]
                >= avg_deviation_payoff + eta[intermediary.id] * epsilons[intermediary.id],
                f"stability_{intermediary.id}",
            )

        # impose optional constraints for farmer structured payments and intermediary domination
        if self.options.structured_farmer_payments:
            fixed_payment = model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=-float("inf"), name="fixed_payment"
            )
            payment_per_quantity = model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0.0, name="payment_per_quantity"
            )
            paved_distance_penalty = model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0.0, name="paved_distance_penalty"
            )
            dirt_distance_penalty = model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0.0, name="dirt_distance_penalty"
            )

            for farmer in self.instance.farmers:
                base_payments = fixed_payment + payment_per_quantity * farmer.quantity
                paved_penalty = (
                    paved_distance_penalty
                    * (farmer.paved_to_mill > self.PAVED_THRESHOLD)
                    * farmer.quantity
                )
                dirt_penalty = (
                    dirt_distance_penalty
                    * (farmer.dirt_to_mill > self.DIRT_THRESHOLD)
                    * farmer.quantity
                )
                # net payment should equal fixed payment plus per-quantity payment minus penalties
                model.addConstr(
                    farmer_payment_vars[farmer.id] == base_payments - paved_penalty - dirt_penalty,
                    f"structured_farmer_payment_{farmer.id}",
                )
        else:
            fixed_payment = None
            payment_per_quantity = None
            paved_distance_penalty = None
            dirt_distance_penalty = None

        if self.options.dominance_constraints:
            self.output.collection("Applied dominance relations", self.dominance_relations)

            if not self.dominance_relations:
                self.output.warning(
                    "Domination is enabled, but no dominance relations were generated."
                )
            for intermediary_id_1, intermediary_id_2 in self.dominance_relations:
                model.addConstr(
                    intermediary_profit_vars[intermediary_id_1]
                    >= intermediary_profit_vars[intermediary_id_2],
                    f"domination_constraint_{intermediary_id_1}_{intermediary_id_2}",
                )

        # set objective
        total_fruit_value = sum(
            farmer.quantity * self.instance.fruit_price_per_ton for farmer in self.instance.farmers
        )
        total_farmer_payments = gp.quicksum(
            farmer_payment_vars[farmer.id] for farmer in self.instance.farmers
        )
        total_intermediary_profits = gp.quicksum(
            intermediary_profit_vars[intermediary.id]
            for intermediary in self.instance.intermediaries
        )
        expected_intermediary_costs = gp.quicksum(
            intermediary_set_prob_vars[k]
            * self.intermediary_set_to_cost[valid_intermediary_sets[k]]
            for k in range(n_valid_intermediary_sets)
        )
        platform_profit_expr = (
            total_fruit_value
            - total_farmer_payments
            - total_intermediary_profits
            - expected_intermediary_costs
        )
        model.setObjective(platform_profit_expr, gp.GRB.MAXIMIZE)
        model.update()

        # optimize with a subset of rows (fewer constraints)
        time_optimization_start = time.time()
        model.optimize()
        initial_solve_time = time.time() - time_optimization_start
        self.time_usage["solving"] += initial_solve_time

        require_solution(model, f"Initial primal solve ({sol_type})")

        def add_violated_stability_cuts() -> int:
            """
            Check whether optimizer solution violates stability constraints
            and adds cuts accordingly.

            Returns:
                int: number of new stability cuts added to prevent current violations.
            """
            start_callback_time = time.time()
            eta_val = model.getAttr("X", eta)
            kappa_val = model.getAttr("X", kappa)
            farmer_payments = model.getAttr("X", farmer_payment_vars)

            new_route_by_farmer_ids_set = {}

            new_rows = 0
            for intermediary in self.instance.intermediaries:
                for hist_set_index, hist_set in enumerate(self.active_hist_sets[intermediary.id]):
                    # construct prize for each farmer, with quantity
                    # outside historical set incurring eta penalty
                    prizes = {}
                    for farmer in self.instance.farmers:
                        farmer_payment = farmer_payments[farmer.id]

                        quantity_outside_hist_set = (
                            farmer.quantity if farmer.id not in hist_set else 0.0
                        )

                        prizes[farmer.id] = (
                            farmer.quantity * self.instance.fruit_price_per_ton
                            - farmer_payment
                            - eta_val[intermediary.id] * quantity_outside_hist_set
                        )

                    # solve the prize-collecting TSP
                    start_tsp_time = time.time()
                    candidate_routes, candidate_objs = self.tsp_solver.solve(prizes)
                    if not candidate_routes:
                        self.output.warning(
                            f"TSP returned no feasible routes for {intermediary.id}; "
                            f"hist_set_index={hist_set_index}"
                        )
                    self.time_usage["tsp"] += time.time() - start_tsp_time

                    # loop through routes and add cuts to stabilize violations
                    for route, obj in zip(candidate_routes, candidate_objs, strict=True):
                        if not route.farmers:
                            continue

                        violation = (
                            obj
                            - kappa_val[intermediary.id, hist_set_index]
                            - self.params.het_costs[intermediary.id]
                        )

                        farmer_ids_set = frozenset([farmer.id for farmer in route.farmers])

                        # add cuts if unstable and farmer set not encountered before
                        if (
                            violation > Optimizer.CUT_TOL
                            and farmer_ids_set not in new_route_by_farmer_ids_set
                        ):
                            start_time = time.time()
                            self._add_stability_cuts_for_route(
                                model, eta, kappa, farmer_payment_vars, route
                            )
                            new_rows += 1
                            self.time_usage["adding_constraints_found"] += time.time() - start_time

                            new_route_by_farmer_ids_set[farmer_ids_set] = route

            self.route_by_farmer_ids_set.update(new_route_by_farmer_ids_set)

            self.time_usage["callback"] += time.time() - start_callback_time

            return new_rows

        # repeatedly add new stability cuts (rows) in response to
        #   violations until solution is stable
        n_added_rows = 0
        while True:
            new_rows = add_violated_stability_cuts()

            time_optimization_start = time.time()
            model.optimize()
            self.time_usage["solving"] += time.time() - time_optimization_start

            require_solution(model, f"Primal row-generation solve ({sol_type})")

            if new_rows == 0:
                break

            n_added_rows += new_rows

        def _extract_result(
            model, 
            platform_profit, 
            payment_per_quantity, 
            paved_distance_penalty, 
            dirt_distance_penalty
        ):
            intermediary_profits = {
                intermediary.id: intermediary_profit_vars[intermediary.id].X
                for intermediary in self.instance.intermediaries
            }
            farmer_payments = {
                farmer.id: farmer_payment_vars[farmer.id].X for farmer in self.instance.farmers
            }
            set_probabilities = model.getAttr("X", intermediary_set_prob_vars)
            expected_intermediary_cost = gp.quicksum(
                intermediary_set_prob_vars[k]
                * self.intermediary_set_to_cost[valid_intermediary_sets[k]]
                for k in range(n_valid_intermediary_sets)
            ).getValue()

            result = OptimizationResult(
                instance=self.instance,
                intermediary_set_probabilities=set_probabilities,
                farmer_payments=farmer_payments,
                intermediary_profits=intermediary_profits,
                platform_profit=platform_profit,
                expected_intermediary_cost=expected_intermediary_cost
            )

            if (
                payment_per_quantity is not None
                and paved_distance_penalty is not None
                and dirt_distance_penalty is not None
            ):
                result.payment_per_quantity = payment_per_quantity.X
                result.paved_distance_penalty = paved_distance_penalty.X
                result.dirt_distance_penalty = dirt_distance_penalty.X
            else:
                payment_per_quantity = None
                result.paved_distance_penalty = None
                result.dirt_distance_penalty = None

            return result

        # extract solution information and add to solution summary
        platform_profit = model.ObjVal
        primary_result = _extract_result(
            model=model, 
            platform_profit=platform_profit, 
            payment_per_quantity=payment_per_quantity, 
            paved_distance_penalty=paved_distance_penalty, 
            dirt_distance_penalty=dirt_distance_penalty
        )

        # re-optimization of degenerate solutions (since there can be multiple optimal solutions)
        model.addConstr(
            platform_profit_expr >= model.ObjVal - Optimizer.PRIMARY_OBJECTIVE_TOL, "optimality"
        )

        # re-optimize to get max intermediary welfare solution
        intermediary_welfare = gp.quicksum(
            intermediary_profit_vars[intermediary_id] for intermediary_id in self.intermediary_ids
        )
        model.setObjective(intermediary_welfare, gp.GRB.MAXIMIZE)
        model.optimize()

        require_solution(model, "Maximum intermediary-welfare solve")

        # extract solution information and add to solution summary
        max_intermediary_welfare = model.ObjVal
        max_intermediary_welfare_result = _extract_result(
            model=model,
            platform_profit=platform_profit,
            payment_per_quantity=payment_per_quantity,
            paved_distance_penalty=paved_distance_penalty,
            dirt_distance_penalty=dirt_distance_penalty
        )

        # re-optimize to get max farmer welfare solution
        farmer_welfare = gp.quicksum(
            farmer_payment_vars[farmer.id] for farmer in self.instance.farmers
        )
        model.setObjective(farmer_welfare, gp.GRB.MAXIMIZE)
        model.optimize()

        require_solution(model, "Maximum farmer-welfare solve")

        max_farmer_welfare = model.ObjVal
        max_farmer_welfare_result = _extract_result(
            model=model,
            platform_profit=platform_profit,
            payment_per_quantity=payment_per_quantity,
            paved_distance_penalty=paved_distance_penalty,
            dirt_distance_penalty=dirt_distance_penalty
        )
        

        self.output.subsection("Primal Solve Result")
        self.output.metric("New rows", n_added_rows, precision=0)
        self.output.metric("Platform profit", platform_profit)
        self.output.metric("Max intermediary welfare", max_intermediary_welfare)
        self.output.metric("Max farmer welfare", max_farmer_welfare)


        return BranchPrimalResult(
            primary_result=primary_result,
            n_added_rows=n_added_rows,
            max_intermediary_welfare_result=max_intermediary_welfare_result,
            max_farmer_welfare_result=max_farmer_welfare_result
        )


    def solve_dual_for_branch(self, branch: Branch) -> BranchDualResult:
        """Construct and solve the dual LP for a given branch.

        The dual problem is used to generate new columns via a
        pricing subproblem.  It returns the objective value and the number of
        columns added.

        Args:
            branch (Branch): branch whose dual LP is to be solved.

        Returns:
            dict[str, float | int]: a dict containing the objective value
                and the number of columns added.
        """

        model = gp.Model("Dual")
        model.setParam("Threads", self.params.threads)
        model.setParam("OutputFlag", 0)

        if self.options.pay_unmatched:
            raise NotImplementedError(
                "Dual column generation is not implemented for pay_unmatched=True."
            )

        # create dual variables
        # dual variable corresponding to route-based stability cuts
        alpha = model.addVars(
            [
                (intermediary.id, hist_set_index, route_set_index)
                for intermediary in self.instance.intermediaries
                for hist_set_index in range(len(self.active_hist_sets[intermediary.id]))
                for route_set_index in self.route_by_farmer_ids_set
            ],
            vtype=gp.GRB.CONTINUOUS,
            lb=0.0,
            name="alpha",
        )

        # dual variable corresponding to constraint that intermediary set probabilities sum to one
        beta = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-float("inf"), name="beta")

        # dual variables corresponding to intermediary-payment upper-bound constraints
        lamb = model.addVars(self.intermediary_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="lamb")

        # dual variables corresponding to intermediary-payment stability constraints
        mu = model.addVars(self.intermediary_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="mu")

        # optional dual variables corresponding to optional primal
        #   structured farmer payment and domination constraints
        if self.options.structured_farmer_payments:
            gamma = model.addVars(
                self.farmer_ids, vtype=gp.GRB.CONTINUOUS, lb=-float("inf"), name="gamma"
            )
        else:
            gamma = None

        if self.options.dominance_constraints:
            self.output.collection("Applied dominance relations", self.dominance_relations)

            if not self.dominance_relations:
                self.output.warning(
                    "Domination is enabled, but no dominance relations were generated."
                )
            D = model.addVars(self.dominance_relations, vtype=gp.GRB.CONTINUOUS, lb=0.0, name="D")
        else:
            D = None

        # dual constraint 1: corresponds to farmer_payment_vars in primal
        if gamma is not None:
            for farmer in self.instance.farmers:
                model.addConstr(
                    gamma[farmer.id]
                    - 1
                    + gp.quicksum(
                        alpha[intermediary.id, hist_set_index, route_set_index]
                        * (farmer.id in route_set_index)
                        for intermediary in self.instance.intermediaries
                        for hist_set_index in range(len(self.active_hist_sets[intermediary.id]))
                        for route_set_index in self.route_by_farmer_ids_set
                    )
                    <= 0
                )
        else:
            for farmer in self.instance.farmers:
                model.addConstr(
                    -1
                    + gp.quicksum(
                        alpha[intermediary.id, hist_set_index, route_set_index]
                        * (farmer.id in route_set_index)
                        for intermediary in self.instance.intermediaries
                        for hist_set_index in range(len(self.active_hist_sets[intermediary.id]))
                        for route_set_index in self.route_by_farmer_ids_set
                    )
                    <= 0
                )
        # dual constraint 2: corresponds to intermediary_profit_vars in primal
        if D is not None:
            self.output.collection("Applied dominance relations", self.dominance_relations)

            if not self.dominance_relations:
                self.output.warning(
                    "Domination is enabled, but no dominance relations were generated."
                )

            for intermediary in self.instance.intermediaries:
                n_dominates, n_dominated_by = 0, 0
                for intermediary_id1, intermediary_id2 in self.dominance_relations:
                    if intermediary.id == intermediary_id1:
                        n_dominates += D[intermediary_id1, intermediary_id2]
                    elif intermediary.id == intermediary_id2:
                        n_dominated_by += D[intermediary_id1, intermediary_id2]

                model.addConstr(
                    -1 + mu[intermediary.id] - lamb[intermediary.id] + n_dominates - n_dominated_by
                    <= 0,
                    f"domination_constraint_{intermediary.id}",
                )
        else:
            for intermediary in self.instance.intermediaries:
                model.addConstr(-1 + mu[intermediary.id] - lamb[intermediary.id] <= 0)
        # dual constraint 3: corresponds to intermediary_set_prob_vars in primal
        model.addConstrs(
            -self.intermediary_set_to_cost[intermediary_set]
            + gp.quicksum(
                lamb[intermediary_id]
                * self.intermediary_id_to_capacity[intermediary_id]
                * self.instance.fruit_price_per_ton
                for intermediary_id in intermediary_set
            )
            - beta
            <= 0
            for intermediary_set in self.intermediary_set_to_cost
            if self._is_valid_intermediary_set(branch, intermediary_set)
        )
        # dual constraint 4: corresponds to eta variable in primal
        for intermediary in self.instance.intermediaries:
            model.addConstr(
                -self.params.epsilons[intermediary.id] * mu[intermediary.id]
                + gp.quicksum(
                    alpha[intermediary.id, hist_set_index, route_set_index]
                    * sum(
                        farmer.quantity
                        for farmer in self.route_by_farmer_ids_set[route_set_index].farmers
                        if farmer.id not in self.active_hist_sets[intermediary.id][hist_set_index]
                    )
                    for hist_set_index in range(len(self.active_hist_sets[intermediary.id]))
                    for route_set_index in self.route_by_farmer_ids_set
                )
                <= 0
            )
        # dual constraint 5: corresponds to kappa variable in primal
        for intermediary in self.instance.intermediaries:
            for hist_set_index in range(len(self.active_hist_sets[intermediary.id])):
                model.addConstr(
                    -1 / len(self.active_hist_sets[intermediary.id]) * mu[intermediary.id]
                    + gp.quicksum(
                        alpha[intermediary.id, hist_set_index, route_set_index]
                        for route_set_index in self.route_by_farmer_ids_set
                    )
                    == 0
                )
        # optional dual constraints corresponding to optional
        #   primal structured farmer payments variables
        if gamma is not None:
            # dual constraint corresponding to fixed_payment variable in primal
            model.addConstr(
                gp.quicksum(gamma[farmer.id] for farmer in self.instance.farmers) == 0,
                "gamma_sum_zero",
            )
            # dual constraint corresponding to payment_per_quantity variable in primal
            model.addConstr(
                gp.quicksum(-gamma[farmer.id] * farmer.quantity for farmer in self.instance.farmers)
                <= 0,
                "gamma_quantity_zero",
            )

            # dual constraint corresponding to paved_distance_penalty variable in primal
            model.addConstr(
                gp.quicksum(
                    gamma[farmer.id]
                    * (farmer.paved_to_mill > self.PAVED_THRESHOLD)
                    * farmer.quantity
                    for farmer in self.instance.farmers
                )
                <= 0,
                "gamma_dist_zero_1",
            )
            # dual constraint corresponding to dirt_distance_penalty variable in primal
            model.addConstr(
                gp.quicksum(
                    gamma[farmer.id] * (farmer.dirt_to_mill > self.DIRT_THRESHOLD) * farmer.quantity
                    for farmer in self.instance.farmers
                )
                <= 0,
                "gamma_dist_zero_2",
            )

        # set objective
        total_fruit_revenue = sum(
            self.instance.fruit_price_per_ton * farmer.quantity for farmer in self.instance.farmers
        )
        fruit_revenue_by_route = {
            route_set_idx: sum(
                self.instance.fruit_price_per_ton * farmer.quantity for farmer in route.farmers
            )
            for route_set_idx, route in self.route_by_farmer_ids_set.items()
        }
        stability_cut_constant_terms = gp.quicksum(
            alpha[intermediary.id, hist_set_index, route_set_index]
            * (
                route.cost
                + self.params.het_costs[intermediary.id]
                - fruit_revenue_by_route[route_set_index]
            )
            for intermediary in self.instance.intermediaries
            for hist_set_index in range(len(self.active_hist_sets[intermediary.id]))
            for route_set_index, route in self.route_by_farmer_ids_set.items()
        )

        objective = total_fruit_revenue + beta + stability_cut_constant_terms
        model.setObjective(objective, gp.GRB.MINIMIZE)
        model.update()

        # optimize with a subset of columns
        model.optimize()
        require_solution(model, "Initial dual solve")

        # repeatedly add new stability cuts (cols) in response to violations
        # until solution is stable
        n_added_cols = 0
        while True:
            # prize new intermediary sets
            intermediary_id_to_prize = {
                self.intermediary_ids[i]: (
                    lamb[self.intermediary_ids[i]].X
                    * self.intermediary_id_to_capacity[self.intermediary_ids[i]]
                    * self.instance.fruit_price_per_ton
                )
                for i in range(self.n_intermediaries)
            }

            # get best intermediary set, objective, and costs given prizes
            result = self._get_best_intermediary_set(intermediary_id_to_prize, branch)
            if result is None:
                raise RuntimeError(
                    "Branch became infeasible during dual column generation "
                    "after passing branch initialization."
                )
            best_intermediary_set, best_obj, best_cost = result
            best_intermediary_set = frozenset(best_intermediary_set)

            # add the most violated intermediary set constraint and re-solve.
            # stop when the pricing problem finds no intermediary set
            # whose dual constraint is violated beyond COLUMN_TOL.
            if best_obj > beta.X + Optimizer.COLUMN_TOL:
                constr = (
                    gp.quicksum(
                        lamb[intermediary_id]
                        * self.intermediary_id_to_capacity[intermediary_id]
                        * self.instance.fruit_price_per_ton
                        for intermediary_id in best_intermediary_set
                    )
                    - beta
                    - best_cost
                    <= 0
                )

                if best_intermediary_set in self.intermediary_set_to_cost:
                    raise RuntimeError(f"Intermediary set {best_intermediary_set} already exists")
                if not self._is_valid_intermediary_set(branch, best_intermediary_set):
                    raise RuntimeError(
                        f"Intermediary set {best_intermediary_set} is not valid for branch {branch}"
                    )

                self.intermediary_set_to_cost[best_intermediary_set] = best_cost

                model.addConstr(constr, f"intermediary_set_constraint_{n_added_cols}")
                n_added_cols += 1
                model.optimize()
                require_solution(model, "Dual column-generation solve")
            else:
                break

        self.output.subsection("Dual Column Generation Result")
        self.output.metric("New columns", n_added_cols, precision=0)
        self.output.metric("Objective", model.objVal)

        return BranchDualResult(objective_value=model.ObjVal, n_added_columns=n_added_cols)



    def exceeds_global_lb(self, value: float, tolerance: float) -> bool:
        return value > self.best_lb + tolerance

    def record_summary(self):
        self.instance_summary.lower_bounds.append(self.best_lb)
        self.instance_summary.upper_bounds.append(self.best_ub)
        self.instance_summary.timestamps.append(time.time() - self.instance_summary.start_time)
        self.instance_summary.oracle_calls.append(self.total_oracle_calls)

    def _add_stability_cuts_for_route(self, model, eta, kappa, farmer_payment_vars, route) -> None:
        """Add stability cuts to the primal model for a given route.

        Each intermediary's constraints are augmented to prevent profitable
        deviations along `route` given the current dual variables.  The
        optional `add_info` dictionary may include values for debugging or
        incremental updates.

        Args:
            model (gp.Model): Gurobi model to add cuts to
            eta (tupledict): dual variables for intermediary
            kappa (gp.MVar): matrix of variables for intermediaries and hist sets
            farmer_payment_vars (tupledict): farmer price variables

        Returns:
            None
        """
        for intermediary in self.instance.intermediaries:
            # add stability cuts for each historical set to prevent deviations
            for hist_set_idx, hist_set in enumerate(self.active_hist_sets[intermediary.id]):
                total_quantity_outside_hist_set = sum(
                    farmer.quantity for farmer in route.farmers if farmer.id not in hist_set
                )
                route_farmer_payments = gp.quicksum(
                    farmer_payment_vars[farmer.id]
                    for farmer in route.farmers
                    if farmer.id in self.farmer_ids
                )
                # see model for stability cut definition
                cut_lhs = (
                    route.value
                    - self.params.het_costs[intermediary.id]
                    - route_farmer_payments
                    - eta[intermediary.id] * total_quantity_outside_hist_set
                    - kappa[intermediary.id, hist_set_idx]
                )
                model.addConstr(cut_lhs <= 0)

    def _get_best_intermediary_set(
        self, intermediary_id_to_prize: dict[str, float], branch: Branch
    ) -> tuple[frozenset[str], float, float] | None:
        """Compute the best intermediary set given prize values and branch.

        Solves a simplified selection problem over ``prizes`` taking into
        account forced matches/unmatches.  Also increments the oracle counter
        if appropriate.

        Args:
            intermediary_id_to_prize (dict[str, float]): dict mapping intermediary IDs
                to their prize values. `intermediary_id_to_prize[i]` is the prize
                associated with intermediary `i`.
            branch (Branch): the current branch, which controls which intermediaries
                must/must not be matched.

        Returns:
            tuple[frozenset[str], float, float]: a tuple containing
                - the set of intermediary IDs that maximizes objective
                - the objective value corresponding to that set
                - the cost associated with that set.
        """

        known_ids = set(self.intermediary_ids)

        if branch.forced_match & branch.forced_unmatch:
            raise ValueError("An intermediary cannot be both forced matched and forced unmatched.")

        if not branch.forced_match <= known_ids:
            raise ValueError("Branch contains unknown forced-match IDs.")

        if not branch.forced_unmatch <= known_ids:
            raise ValueError("Branch contains unknown forced-unmatch IDs.")

        if branch.count_flag:
            self.total_oracle_calls += 1

        net_prizes = {
            intermediary_id: intermediary_id_to_prize[intermediary_id]
            - self.params.het_costs[intermediary_id]
            for intermediary_id in intermediary_id_to_prize.keys()
        }

        required_intermediaries = set(branch.forced_match)
        required_prizes_sum = sum(
            net_prizes[intermediary_id] for intermediary_id in required_intermediaries
        )
        len_required = len(required_intermediaries)

        for intermediary_id in branch.forced_unmatch:
            del net_prizes[intermediary_id]
        for intermediary_id in required_intermediaries:
            del net_prizes[intermediary_id]

        ordered_intermediaries = sorted(
            net_prizes.keys(), key=lambda x: net_prizes[x], reverse=True
        )
        objs = {}
        for n_trucks in self.routing_cost_by_truck_count:
            n_optional_needed = n_trucks - len_required

            if n_optional_needed < 0:
                continue

            if n_optional_needed > len(ordered_intermediaries):
                continue

            selected_optional = ordered_intermediaries[:n_optional_needed]

            prizes_sum = sum(net_prizes[intermediary_id] for intermediary_id in selected_optional)

            intermediary_set = frozenset(required_intermediaries.union(selected_optional))

            objs[intermediary_set] = (
                prizes_sum + required_prizes_sum - self.routing_cost_by_truck_count[n_trucks]
            )

        if not objs:
            return None

        max_obj = max(objs.values())
        max_intermediary_set = [
            intermediary_set for intermediary_set, obj in objs.items() if obj == max_obj
        ]
        max_cost = self.routing_cost_by_truck_count[len(max_intermediary_set[0])] + sum(
            self.params.het_costs[intermediary_id] for intermediary_id in max_intermediary_set[0]
        )

        return max_intermediary_set[0], max_obj, max_cost

    def _is_valid_intermediary_set(self, branch: Branch, intermediary_set: frozenset[str]) -> bool:
        """
        Check whether a candidate selection respects a branch's restrictions.

        Returns:
            bool: `True` if `intermediary_set` does not violate any forced match or
                unmatch assignments, `False` otherwise.
        """
        for intermediary_id in branch.forced_unmatch:
            if intermediary_id in intermediary_set:
                return False

        for intermediary_id in branch.forced_match:
            if intermediary_id not in intermediary_set:
                return False

        return True

    def _configure_hist_sets(
        self,
        aggregate: bool,
    ) -> None:
        if aggregate:
            self.active_hist_sets = {
                intermediary.id: (
                    frozenset(
                        farmer.id
                        for farmer in self.instance.farmers
                        if farmer.intermediary_id == intermediary.id
                    ),
                )
                for intermediary in self.instance.intermediaries
            }
        else:
            self.active_hist_sets = {
                intermediary_id: hist_sets
                for intermediary_id, hist_sets in self._original_hist_sets.items()
            }

    def _calc_dominance(self) -> list[tuple[str, str]]:
        """Compute pairwise dominance relations between intermediaries.

        An intermediary ``i`` is said to dominate ``j`` if ``i`` has lower
        heterogeneous costs and at least as much expected historical fruit.
        The method returns a list of tuples *(i, j)* that satisfy this
        condition.  Dominance is used to enforce ordering constraints in the
        optimization models.

        Returns:
            list[tuple[str, str]]: list of dominance relationships where each
                entry is a tuple `(int_1, int_2)` with the convention that
                `int_1` dominates `int_2`.
        """
        # precompute historical quantities
        hist_avg_quantities = {}
        for intermediary in self.instance.intermediaries:
            hist_sets = self.active_hist_sets[intermediary.id]

            hist_quantity = 0
            for hist_set in hist_sets:
                for farmer in self.instance.farmers:
                    if farmer.id in hist_set:
                        hist_quantity += farmer.quantity

            hist_avg_quantities[intermediary.id] = hist_quantity / len(hist_sets)

        # compare intermediaries
        def _dominates(i, j):
            return (
                self.params.het_costs[i.id] < self.params.het_costs[j.id]
                and hist_avg_quantities[i.id] >= hist_avg_quantities[j.id]
            )

        dominance_relations = []
        for intermediary_1, intermediary_2 in combinations(self.instance.intermediaries, 2):
            if _dominates(intermediary_1, intermediary_2):
                dominance_relations.append((intermediary_1.id, intermediary_2.id))
            if _dominates(intermediary_2, intermediary_1):
                dominance_relations.append((intermediary_2.id, intermediary_1.id))

        self.output.section("Dominance Relations")
        self.output.metric("Number of relations", len(dominance_relations))
        self.output.collection("Relations", dominance_relations)
        return dominance_relations

    def _initialize_routing_cost_by_truck_count(self) -> dict[int, float]:
        """Compute an initial set of matchings by solving simple VRPs.

        The method solves a sequence of vehicle routing problems with
        increasing minimum truck counts to populate ``self.base_matchings``.
        These provide starting points for column generation.  The results are
        also used to determine ``min_trucks`` and ``max_trucks``.

        Returns:
            dict[int, Matching]: a dict that associates the number of trucks with
                a Matching corresponding to it.
        """

        self.output.section("VRP Initialization")

        # solve for min cost matchin
        self.output.subsection("Solving for minimum cost matching")
        min_cost_matching = self.vrp_solver.solve(
            n_vehicles_lower_bound=1,
            n_vehicles_upper_bound=self.n_intermediaries,
            threads=self.params.threads,
            time_limit_seconds=self.params.vrp_time_limit_seconds
        )
        min_trucks = len(min_cost_matching.routes)

        self.output.metric("Minimum-cost objective", min_cost_matching.cost)
        self.output.metric("Trucks used", min_trucks, precision=0)

        self.output.blank()
        self.output.subsection(f"Populating routing costs using at least {min_trucks} trucks")
        self.output.metric("VRP mode", self.params.vrp_mode)
        self.output.blank()
        self.output.subsubsection(f"Number of Trucks = {min_trucks}")
        self.output.metric("Cost", min_cost_matching.cost, precision=0)
        routing_cost_by_truck_count = {min_trucks: min_cost_matching.cost}

        for n_trucks in range(min_trucks + 1, self.n_intermediaries + 1):
            self.output.blank()
            self.output.subsubsection(f"Number of Trucks = {n_trucks}")
            if self.params.vrp_mode == "exact":
                matching = self.vrp_solver.solve(
                    n_vehicles_lower_bound=n_trucks, 
                    n_vehicles_upper_bound=n_trucks, 
                    threads=self.params.threads,
                    time_limit_seconds=self.params.vrp_time_limit_seconds
                )
                cost = matching.cost

            elif self.params.vrp_mode == "approximate":
                cost = (
                    min_cost_matching.cost
                    + (n_trucks - min_trucks) * self.instance.truck_fixed_cost
                )
            else:
                raise ValueError("VRP Mode Not Recognized")
            self.output.metric("Cost", cost, precision=0)
            routing_cost_by_truck_count[n_trucks] = cost

        self.output.section("Initial Routing Costs by Truck Count")
        self.output.metric("Number of Trucks", "Cost")
        for n_trucks in routing_cost_by_truck_count:
            self.output.metric(str(n_trucks), routing_cost_by_truck_count[n_trucks])

        return routing_cost_by_truck_count

    def _initialize_intermediary_set_to_cost(self) -> dict[frozenset[str], float]:
        """Initialize the catalogue of all explored intermediary selections.

        Begins with the cheapest matching for the minimum truck count and
        records its cost. Additional sets will be added during column
        generation.

        Returns:
            dict[frozenset[str], float]: a dict that records the cost associated with each matching.
        """
        # sort intermediaries by heterogeneous costs.
        ordered_intermediaries = sorted(
            self.instance.intermediaries, key=lambda x: self.params.het_costs[x.id], reverse=False
        )
        # include first min_trucks intermediaries in min_set.
        min_set = set(
            [intermediary.id for intermediary in ordered_intermediaries[: self.min_trucks]]
        )

        return {frozenset(min_set): self._intermediary_set_cost(min_set)}

    def _intermediary_set_cost(self, intermediary_set: frozenset[str] | set[str]) -> float:
        """Return the total cost of a given set of intermediaries.

        Combines the base VRP cost for the appropriate truck count with the
        heterogeneous costs of the selected intermediaries.

        Args:
            intermediary_set (set[str]): set of selected intermediaries.

        Returns:
            float: total cost associated with the set of intermediaries.
        """
        n_trucks = len(intermediary_set)
        vrp_cost = self.routing_cost_by_truck_count[n_trucks]
        het_costs = sum(
            self.params.het_costs[intermediary_id] for intermediary_id in intermediary_set
        )
        return vrp_cost + het_costs

    def _verify_farmer_distances(self):
        for farmer in self.instance.farmers:
            if farmer.paved_to_mill is None or farmer.dirt_to_mill is None:
                raise RuntimeError("Farmer distances not calculated.")

    def _print_optimizer_params(self) -> None:
        self.output.section("Optimizer Parameters")

        for param_field in fields(self.params):
            value = getattr(self.params, param_field.name)
            label = param_field.name

            if isinstance(value, Mapping):
                formatted = pformat(
                    dict(value),
                    width=self.output.width,
                    sort_dicts=True,
                    compact=False,
                )
                self.output.subsection(label)
                self.output.message(formatted, indent=1)
            else:
                self.output.metric(label, value)

    def _print_solver_options(self) -> None:
        self.output.section("Solver Options")

        for field in fields(self.options):
            value = getattr(self.options, field.name)

            label = field.name.replace("_", " ").title()
            self.output.metric(label, value)
