from itertools import combinations

import numpy as np

from ...reporting.containers import BranchSolution
from ..branch import Branch
from .optimizer_protocol import OptimizerProtocol

def solve_npm(
    optimizer: OptimizerProtocol,
    capped
) -> None:
    def compute_network_priority(intermediary, capped):
        """NPM(t) := sigma_t / min(K, n_t + eps_t), where sigma_t is t's total fixed cost
        (the common truck fixed cost plus t's heterogeneous cost)."""
        sigma = optimizer.instance.truck_fixed_cost + optimizer.het_costs[intermediary.id]
        K = intermediary.capacity
        n = optimizer.status_quo_quantities[intermediary.id]
        eps = optimizer.epsilons[intermediary.id]
        denom = min(K, max(n + eps, float(1e-4))) if capped else max(n + eps, float(1e-4))
        return sigma / denom

    # order intermediaries in increasing order of e_t := sigma_t / min(K, n_t + eps_t),
    # breaking ties by decreasing epsilon
    network_priorities = [
        (
            intermediary.id,
            compute_network_priority(intermediary, capped),
            optimizer.epsilons[intermediary.id],
        )
        for intermediary in optimizer.instance.intermediaries
    ]
    sort_key = lambda x: (x[1], -x[2])
    optimizer.output.section("Network Priority Order")
    optimizer.output.collection(
        label="Network Priorities (sorted)",
        values=sorted(network_priorities, key=sort_key)
    )
    network_priority_order = [item[0] for item in sorted(network_priorities, key=sort_key)]

    # initialize solver
    P = set()
    pricing_cache = {}
    U_0, C_0 = float("inf"), float("inf")
    for k in range(len(network_priority_order) + 1):
        if k > 0:
            P = P.union({network_priority_order[k - 1]})

        # get forced min. cost. matching
        branch = Branch(
            forced_match = P,
            forced_unmatch = set()
        )
        if not optimizer.initialize_branch(branch):
            break

        # special: initialize U_0 at the root
        if k == 0:
            forced_ub_result = optimizer.solve_primal_for_branch(
                branch=branch,
                sol_type="forced_upper_bound",
                compute_intermediary_welfare=False,
                compute_farmer_welfare=False
            )
            U_0 = forced_ub_result.platform_profit
            C_0 = branch.min_cost

            optimizer.instance_summary.forced_upper_bound = U_0
            optimizer.best_ub = U_0

        # check stopping condition
        # note that U_0 = R - C_0 - phi_free -> -phi_free = U_0 - R + C_0
        # and thus U_k = R - C_k - phi_free = R - C_k + U_0 - R + C_0 = U_0 - (C_k - C_0).
        U_k = U_0 - (branch.min_cost - C_0)

        optimizer.output.subsection(f"Checking Upper Bound (k={k})")
        optimizer.output.metric(f"U_{k}", U_k)
        if not optimizer.exceeds_global_lb(U_k, optimizer.BRANCH_PRUNE_TOL):
            optimizer.output.status(
                "U_k cannot improve on the incumbent; stopping network-prioritized search."
            )
            break

        # if y^k is not in pricing cache, then solve and cache value & optimizer
        if branch.min_cost_set in pricing_cache:
            optimizer.output.subsection(f"Solved pricing for P_{k}")
            optimizer.output.metric("Objective", pricing_cache[branch.min_cost_set])
            optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))
            continue

        forced_lb_result = optimizer.solve_primal_for_branch(
            branch=branch,
            sol_type="forced_lower_bound",
            compute_intermediary_welfare=False,
            compute_farmer_welfare=False,
        )
        branch_profit = forced_lb_result.platform_profit
        pricing_cache[branch.min_cost_set] = branch_profit

        # the k=0 candidate is the efficient-matching heuristic, i.e. Pi^eff in (25)
        if k == 0:
            optimizer.instance_summary.forced_lower_bound = branch_profit

        optimizer.output.subsection(f"Lower Bound Candidate")
        optimizer.output.metric("Objective", forced_lb_result.platform_profit)
        optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

        # update global lower bound if forced lower bound is tighter (found a better feasible sol'n)
        if optimizer.exceeds_global_lb(
            forced_lb_result.platform_profit, optimizer.GLOBAL_LB_UPDATE_TOL
        ):
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

            optimizer.best_lb = forced_lb_result.platform_profit
            optimizer.best_lb_set = branch.min_cost_set
            optimizer.best_lb_result = forced_lb_result

            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Global Bound Update",
                status="Improved the global lower bound",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill=".",
            )

    previous_lb = optimizer.best_lb
    previous_ub = optimizer.best_ub

    optimizer.record_summary()

    print_bound_update(
        optimizer,
        title="Search Complete",
        status="",
        previous_lb=previous_lb,
        previous_ub=previous_ub,
        fill="=",
    )


# 2^20 candidate sets is already ~1M pricing solves; beyond this enumeration is hopeless
MAX_ENUMERATION_INTERMEDIARIES = 20


def solve_enumeration(
    optimizer: OptimizerProtocol
) -> None:
    """
    Conducts one exact solve by enumerating matched sets.

    Each candidate set S is evaluated by fixing the matching to exactly S and optimizing
    payments, i.e. the pricing problem the other strategies solve at a fully determined
    node. Candidates are visited in increasing order of transportation cost C(S), and the
    search stops once no remaining candidate can beat the incumbent, using the same bound
    as the network-prioritized heuristic: U(S) = U_0 - (C(S) - C_0), where U_0 is the
    root unrestricted-payment bound and C_0 the minimum cost. Because U(S) only decreases
    along the ordering, the first candidate that cannot improve on the incumbent certifies
    every later one. Unlike branch-and-price, this never relies on the LP relaxation, so it
    is unaffected by fractional master solutions.

    Args:
        optimizer (OptimizerProtocol): the optimizer object, see `optimizer.py`.

    Raises:
        ValueError: too many intermediaries to enumerate.
        RuntimeError: no feasible intermediary set exists.
    """
    if optimizer.n_intermediaries > MAX_ENUMERATION_INTERMEDIARIES:
        raise ValueError(
            f"Enumeration supports at most {MAX_ENUMERATION_INTERMEDIARIES} intermediaries, "
            f"got {optimizer.n_intermediaries}."
        )

    # compute the minimum-cost set and the unrestricted-payment bound U_0 at the root
    root_branch = Branch(set(), set())
    if not optimizer.initialize_branch(root_branch):
        raise RuntimeError("No feasible intermediary set exists.")

    forced_ub_result = optimizer.solve_primal_for_branch(
        branch=root_branch,
        sol_type="forced_upper_bound",
        compute_intermediary_welfare=False,
        compute_farmer_welfare=False
    )
    U_0 = forced_ub_result.platform_profit
    C_0 = root_branch.min_cost

    optimizer.instance_summary.forced_upper_bound = U_0
    optimizer.best_ub = U_0

    # rank every set the matching oracle would accept by transportation cost, breaking
    # ties deterministically by the sorted member IDs
    intermediary_ids = sorted(optimizer.intermediary_ids)
    candidates = []
    for size in range(1, len(intermediary_ids) + 1):
        for members in combinations(intermediary_ids, size):
            cost = optimizer.intermediary_set_cost(frozenset(members))
            if cost is not None:
                candidates.append((cost, members))
    candidates.sort()

    optimizer.output.section("Enumeration")
    optimizer.output.metric("Candidate sets", len(candidates), precision=0)
    optimizer.output.metric("Root upper bound U_0", U_0)

    n_evaluated = 0
    stop_bound = None
    for cost, members in candidates:
        # every remaining candidate has cost >= this one, hence bound <= U
        U = U_0 - (cost - C_0)
        if (
            not optimizer.exceeds_global_lb(U, optimizer.BRANCH_PRUNE_TOL)
            or relative_gap(optimizer.best_lb, U) <= optimizer.options.early_stop_threshold
        ):
            stop_bound = U
            break

        branch = Branch(
            forced_match=set(members),
            forced_unmatch=set(intermediary_ids) - set(members)
        )
        if not optimizer.initialize_branch(branch):
            continue

        result = optimizer.solve_primal_for_branch(
            branch=branch,
            sol_type="forced_lower_bound",
            compute_intermediary_welfare=False,
            compute_farmer_welfare=False
        )
        n_evaluated += 1

        # the minimum-cost set is the efficient-matching heuristic's candidate
        if branch.min_cost_set == root_branch.min_cost_set:
            optimizer.instance_summary.forced_lower_bound = result.platform_profit

        optimizer.output.subsection(f"Candidate {n_evaluated}")
        optimizer.output.metric("Upper bound U(S)", U)
        optimizer.output.metric("Objective", result.platform_profit)

        # update global lower bound if this set beats the incumbent
        if optimizer.exceeds_global_lb(result.platform_profit, optimizer.GLOBAL_LB_UPDATE_TOL):
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

            optimizer.best_lb = result.platform_profit
            optimizer.best_lb_set = branch.min_cost_set
            optimizer.best_lb_result = result

            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Global Bound Update",
                status="Improved the global lower bound",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill=".",
            )

    previous_lb = optimizer.best_lb
    previous_ub = optimizer.best_ub

    # evaluated sets cannot exceed the incumbent; unvisited sets are bounded by stop_bound
    if stop_bound is None:
        optimizer.best_ub = optimizer.best_lb
        status = "All candidate sets evaluated"
    else:
        optimizer.best_ub = min(optimizer.best_ub, max(optimizer.best_lb, stop_bound))
        status = "Remaining candidate sets cannot improve on the incumbent"

    optimizer.record_summary()

    optimizer.output.metric("Candidate sets evaluated", n_evaluated, precision=0)
    print_bound_update(
        optimizer,
        title="Search Complete",
        status=status,
        previous_lb=previous_lb,
        previous_ub=previous_ub,
        fill="=",
    )

    if optimizer.best_lb_result is None:
        raise RuntimeError("No primal solution has been found.")


def solve_paper_bnb(
    optimizer: OptimizerProtocol, 
    max_violation_branching: bool,
) -> None:
    """
    Conducts one solve using the paper's branch-and-bound algorithm (§3.2): one
    matching-oracle call per node, bounds from the relaxed and forced primal solves,
    and branching on the no-payment constraints for unmatched intermediaries.

    Args:
        optimizer (OptimizerProtocol): the optimizer object, see `optimizer.py`. 
        max_violation_branching (bool): branch on the largest no-payment violation if
            True, otherwise on a random violating intermediary.

    Raises:
        RuntimeError: no primal solution found.
    """

    # start from the root
    root_branch = Branch(set(), set())
    branches_to_evaluate = [root_branch]
    active_branches = []

    while True:
        # solve each branch to be evaluated; add to active queue as needed
        for branch in branches_to_evaluate:
            branch_solution = solve_branch_paper_bnb(
                optimizer=optimizer, 
                branch=branch, 
                max_violation_branching=max_violation_branching, 
            )
            if branch_solution.status in ["stop", "infeasible"]:
                continue
            elif branch_solution.status in ["active"]:
                active_branches.append(branch_solution)

        # terminate if no more active branches; every remaining node has been resolved
        # (pruned or closed with no candidate left), so the incumbent is proven optimal
        if not active_branches:
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

            optimizer.best_ub = optimizer.best_lb

            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Search Complete",
                status="No unresolved active branches",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill="=",
            )

            break

        optimizer.best_ub = min(
            max(branch_solution.upper_bound for branch_solution in active_branches),
            optimizer.best_ub
        )

        # terminate if optimality gap is small enough
        if (
            relative_gap(optimizer.best_lb, optimizer.best_ub)
            <= optimizer.options.early_stop_threshold
        ):
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Search Complete",
                status="Early stopping; relative gap below threshold",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill="=",
            )
            break

        # prune active branches whose upper bound cannot exceed global lower bound
        active_branches = [
            branch_solution for branch_solution in active_branches
            if optimizer.exceeds_global_lb(branch_solution.upper_bound, optimizer.BRANCH_PRUNE_TOL)
        ]

        # terminate if no more active branches; every remaining node has been resolved
        # (pruned or closed with no candidate left), so the incumbent is proven optimal
        if not active_branches:
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

            optimizer.best_ub = optimizer.best_lb

            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Search Complete",
                status="No unresolved active branches",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill="=",
            )

            break

        # choose max branch using max intermediary profit criterion
        max_branch = max(
            active_branches, key=lambda branch: branch.intermediary_profits[branch.branch_on]
        )

        # update global upper bound using max upper bound from active branches
        current_max_upper_bound = -float("inf")

        for branch in active_branches:
            if branch.upper_bound > current_max_upper_bound:
                current_max_upper_bound = branch.upper_bound

        if current_max_upper_bound < optimizer.best_ub:
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

            optimizer.best_ub = current_max_upper_bound

            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Global Bound Update",
                status="Tightened the global upper bound using active branches",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill=".",
            )

        # pop the max branch from the active branches
        parent_branch = max_branch.branch
        active_branches.remove(max_branch)

        # branch on max active branch
        branch_on = max_branch.branch_on
        branch_value = max_branch.intermediary_profits[max_branch.branch_on]

        optimizer.output.section(f"Branching on {branch_on} with value = {branch_value:.4f}")
        optimizer.output.collection(
            "Parent matched",
            sorted(parent_branch.forced_match),
        )
        optimizer.output.collection(
            "Parent unmatched",
            sorted(parent_branch.forced_unmatch),
        )
        optimizer.output.message(f"Left branch:  force {branch_on} = 1", indent=1)
        optimizer.output.message(f"Right branch: force {branch_on} = 0", indent=1)

        left_branch = Branch(
            parent_branch.forced_match | {branch_on}, parent_branch.forced_unmatch
        )
        right_branch = Branch(
            parent_branch.forced_match, parent_branch.forced_unmatch | {branch_on}
        )

        right_branch.count_flag = False

        branches_to_evaluate = [left_branch, right_branch]

    if optimizer.best_lb_result is None:
        raise RuntimeError("No primal solution has been found.")


def solve_branch_paper_bnb(
    optimizer: OptimizerProtocol, 
    branch: Branch, 
    max_violation_branching: bool, 
) -> BranchSolution:
    """
    Evaluate one branch in the solve using the branch-and-bound search strategy.

    Args:
        optimizer (OptimizerProtocol): the optimizer object, see `optimizer.py`. 
        branch (Branch): the branch to be evaluated.
        max_violation_branching (bool): branch on the largest no-payment violation if
            True, otherwise on a random violating intermediary.

    Raises:
        RuntimeError: RNG not initialized.
        RuntimeError: farmer welfare solution not present.

    Returns:
        BranchSolution: branch solve status and solution summary.
    """

    if not optimizer.rng:
        raise RuntimeError("RNG not initialized.")
    
    if not optimizer.initialize_branch(branch):
        return BranchSolution(status="infeasible", branch=branch)

    # compute lower bound LB^n by forcing solver to use min cost matching plus no-payment
    # constraints (this is a feasible solution)
    forced_lb_result = optimizer.solve_primal_for_branch(
        branch=branch, 
        sol_type="forced_lower_bound",
        compute_farmer_welfare=False,
        compute_intermediary_welfare=False
    )

    optimizer.output.subsection("Lower-Bound Candidate")
    optimizer.output.metric("Objective", forced_lb_result.platform_profit)
    optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

    # update lower bound if evaluating root
    if not branch.forced_match and not branch.forced_unmatch:
        optimizer.instance_summary.forced_lower_bound = forced_lb_result.platform_profit

    # update global lower bound if forced lower bound is tighter (found a better feasible sol'n)
    if optimizer.exceeds_global_lb(
        forced_lb_result.platform_profit, optimizer.GLOBAL_LB_UPDATE_TOL
    ):
        previous_lb = optimizer.best_lb
        previous_ub = optimizer.best_ub

        optimizer.best_lb = forced_lb_result.platform_profit
        optimizer.best_lb_set = branch.min_cost_set
        optimizer.best_lb_result = forced_lb_result

        optimizer.record_summary()

        print_bound_update(
            optimizer,
            title="Global Bound Update",
            status="Improved the global lower bound through forcing",
            previous_lb=previous_lb,
            previous_ub=previous_ub,
            fill=".",
        )

    optimizer.output.blank()

    # compute upper bound UB^n by forcing solver to use min cost matching while relaxing
    # the no-payment constraint. This makes the min cost matching automatically optimal
    # and hence this forced upper bound is a true upper bound.
    if optimizer.options.early_stop_threshold == float("inf"):
        forced_ub_result = optimizer.solve_primal_for_branch(
            branch=branch, 
            sol_type="forced_upper_bound",
            compute_farmer_welfare=False,
            compute_intermediary_welfare=False
        )
    else:
        forced_ub_result = optimizer.solve_primal_for_branch(
            branch=branch, 
            sol_type="forced_upper_bound",
            compute_farmer_welfare=True,
            compute_intermediary_welfare=False
        )

    optimizer.output.subsection("Upper-Bound Candidate")
    optimizer.output.metric("Objective", forced_ub_result.platform_profit)
    optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

    # update upper bound if evaluating root
    if not branch.forced_match and not branch.forced_unmatch:
        optimizer.instance_summary.forced_upper_bound = forced_ub_result.platform_profit

    # prune branch early if upper bound cannot beat existing integer solution.
    can_improve = optimizer.exceeds_global_lb(
        forced_ub_result.platform_profit, optimizer.BRANCH_PRUNE_TOL
    )
    optimizer.output.metric("Global lower bound", optimizer.best_lb)
    optimizer.output.metric("Improvement tolerance", optimizer.BRANCH_PRUNE_TOL)
    optimizer.output.metric("Decision", "Retain" if can_improve else "Prune")

    if not can_improve:
        optimizer.output.message(
            "Reason: branch upper bound cannot improve on the global lower bound.",
            indent=1,
        )

        if not branch.forced_match and not branch.forced_unmatch:
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

            optimizer.best_ub = min(optimizer.best_ub, forced_ub_result.platform_profit)
            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Global Bound Update",
                status="Tightened the global upper bound using root",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill=".",
            )

        return BranchSolution(status="stop", branch=branch)

    if optimizer.options.early_stop_threshold == float("inf"):
        return BranchSolution(
            status="active",
            branch=branch,
            upper_bound=forced_ub_result.platform_profit,
        )
    else:
        # check that farmer welfare solution is present
        if (
            forced_ub_result.max_farmer_welfare_result is None
            or forced_ub_result.max_farmer_welfare_result.intermediary_profits is None
        ):
            raise RuntimeError("farmer welfare solution is not present.")

        max_farmer_welfare_int_profits = (
            forced_ub_result.max_farmer_welfare_result.intermediary_profits
        )

        # guide using intermediary profits from max farmer welfare solution if using guided
        # option, otherwise sample randomly for positive profit intermediaries.
        if max_violation_branching:
            intermediary_profits = max_farmer_welfare_int_profits
        else:
            intermediary_profits = {
                intermediary_id: optimizer.rng.uniform(0, 1)
                if max_farmer_welfare_int_profits[intermediary_id] > optimizer.RANDOM_BRANCH_TOL
                else 0.0
                for intermediary_id in optimizer.intermediary_ids
            }

        branch_on = None
        max_profit = -float("inf")
        for intermediary_id in intermediary_profits:
            if (
                intermediary_id not in branch.min_cost_set
                and intermediary_id not in branch.forced_match
                and intermediary_id not in branch.forced_unmatch
            ):
                if intermediary_profits[intermediary_id] > max_profit:
                    max_profit = intermediary_profits[intermediary_id]
                    branch_on = intermediary_id

        if branch_on is None:
            optimizer.output.status("No eligible intermediary remains for branching; closing branch")
            return BranchSolution(status="stop", branch=branch)

        return BranchSolution(
            status="active",
            branch=branch,
            branch_on=branch_on,
            intermediary_profits=intermediary_profits,
            upper_bound=forced_ub_result.platform_profit,
        )


def solve_lagrangian_bnp(
    optimizer: OptimizerProtocol
) -> None:

    """
    Conducts one solve using the Lagrangian branch-and-price benchmark: at each node,
    the coupling no-payment constraints are priced via column generation over
    intermediary sets, and branching is on fractional matching probabilities.

    Args:
        optimizer (OptimizerProtocol): the optimizer object, see `optimizer.py`. 

    Raises:
        RuntimeError: no primal solution found.
    """

    # start from the root
    root_branch = Branch(set(), set())
    branches_to_evaluate = [root_branch]
    active_branches = []

    while True:
        # solve each branch to be evaluated; add to active queue as needed
        for branch in branches_to_evaluate:
            branch_solution = solve_branch_lagrangian_bnp(
                optimizer=optimizer, 
                branch=branch
            )
            if branch_solution.status in ["stop", "integral", "infeasible"]:
                continue
            elif branch_solution.status in ["fractional"]:
                active_branches.append(branch_solution)

        # terminate if no more active branches
        if not active_branches:
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

            optimizer.best_ub = optimizer.best_lb

            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Search Complete",
                status="No unresolved active branches",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill=".",
            )

            break

        optimizer.best_ub = min(
            max(branch_solution.upper_bound for branch_solution in active_branches),
            optimizer.best_ub
        )

        # terminate if optimality gap is small enough
        if (
            relative_gap(optimizer.best_lb, optimizer.best_ub) 
            <= optimizer.options.early_stop_threshold
        ):
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub
            
            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Search Complete",
                status="Early stopping; relative gap below threshold",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill="=",
            )
            break

        # prune branches whose upper bound cannot exceed global lower bound
        active_branches = [
            branch_solution for branch_solution in active_branches
            if optimizer.exceeds_global_lb(branch_solution.upper_bound, optimizer.BRANCH_PRUNE_TOL)
        ]

        # terminate if no more active branches
        if not active_branches:
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

            optimizer.best_ub = optimizer.best_lb

            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Search Complete",
                status="No unresolved active branches",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill=".",
            )

            break

        # choose max branch using max upper bound criterion
        max_branch = max(active_branches, key=lambda branch: branch.upper_bound)

        # update global upper bound using max upper bound from active branches
        current_max_upper_bound = -float("inf")

        for branch_solution in active_branches:
            if branch_solution.upper_bound > current_max_upper_bound:
                current_max_upper_bound = branch_solution.upper_bound

        if current_max_upper_bound < optimizer.best_ub:
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

            optimizer.best_ub = current_max_upper_bound

            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Global Bound Update",
                status="Tightened the global upper bound from active branches",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill=".",
            )

        # pop the max branch from the active branches
        parent_branch = max_branch.branch
        active_branches.remove(max_branch)

        # branch on max active branch
        branch_on = max_branch.branch_on
        branch_value = max_branch.branch_value

        optimizer.output.section(
            f"Branching on {branch_on} with value = {branch_value:.4f}",
        )
        optimizer.output.collection(
            "Parent matched",
            sorted(parent_branch.forced_match),
        )
        optimizer.output.collection(
            "Parent unmatched",
            sorted(parent_branch.forced_unmatch),
        )
        optimizer.output.message(f"Left branch:  force {branch_on} = 1", indent=1)
        optimizer.output.message(f"Right branch: force {branch_on} = 0", indent=1)

        left_branch = Branch(parent_branch.forced_match | {branch_on}, parent_branch.forced_unmatch)

        right_branch = Branch(
            parent_branch.forced_match, parent_branch.forced_unmatch | {branch_on}
        )

        branches_to_evaluate = [left_branch, right_branch]

    if optimizer.best_lb_result is None:
        raise RuntimeError("No primal solution has been found.")


def solve_branch_lagrangian_bnp(
    optimizer: OptimizerProtocol, 
    branch: Branch
) -> BranchSolution:
    """Perform a Lagrangian branch-and-price iteration on a given branch.

    Args:
        branch (Branch): Branching restrictions to apply (fixed matches/unmatches).

    Returns:
        BranchSolution: branch solve status and solution summary.
    """
    if not optimizer.initialize_branch(branch):
        return BranchSolution(status="infeasible", branch=branch)

    # compute lower bound LB^n by forcing solver to use min cost matching plus no-payment
    # constraints (this is a feasible solution)
    forced_lb_result = optimizer.solve_primal_for_branch(
        branch=branch, 
        sol_type="forced_lower_bound",
        compute_farmer_welfare=False,
        compute_intermediary_welfare=False
    )
    forced_lb_platform_profit = forced_lb_result.platform_profit

    optimizer.output.subsection("Lower-Bound Candidate")
    optimizer.output.metric("Objective", forced_lb_platform_profit)
    optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

    # update lower bound if evaluating root
    if not branch.forced_match and not branch.forced_unmatch:
        optimizer.instance_summary.forced_lower_bound = forced_lb_result.platform_profit

    # update global lower bound if forced lower bound is tighter
    if optimizer.exceeds_global_lb(forced_lb_platform_profit, optimizer.GLOBAL_LB_UPDATE_TOL):
        previous_lb = optimizer.best_lb
        previous_ub = optimizer.best_ub

        optimizer.best_lb = forced_lb_platform_profit
        optimizer.best_lb_set = branch.min_cost_set
        optimizer.best_lb_result = forced_lb_result

        optimizer.record_summary()

        print_bound_update(
            optimizer,
            title="Global Bound Update",
            status="Improved the global lower bound through forcing",
            previous_lb=previous_lb,
            previous_ub=previous_ub,
            fill=".",
        )

    optimizer.output.blank()

    # compute upper bound UB^n by forcing solver to use min cost matching while relaxing
    # the no-payment constraint. This makes the min cost matching automatically optimal
    # and hence this forced upper bound is a true upper bound.
    forced_ub_solution = optimizer.solve_primal_for_branch(
        branch=branch, 
        sol_type="forced_upper_bound",
        compute_farmer_welfare=False,
        compute_intermediary_welfare=False
    )

    optimizer.output.subsection("Forced Upper Bound")
    optimizer.output.metric("Objective", forced_ub_solution.platform_profit)
    optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

    # prune branch early if upper bound cannot beat existing integer solution
    can_improve = optimizer.exceeds_global_lb(
        forced_ub_solution.platform_profit, optimizer.BRANCH_PRUNE_TOL
    )
    optimizer.output.metric("Global lower bound", optimizer.best_lb)
    optimizer.output.metric("Improvement tolerance", optimizer.BRANCH_PRUNE_TOL)
    optimizer.output.metric("Decision", "Retain" if can_improve else "Prune")
    if not can_improve:
        optimizer.output.message(
            "Reason: branch upper bound cannot improve on the global lower bound.",
            indent=1,
        )
        return BranchSolution(status="stop", branch=branch)

    # otherwise, solve primal and dual restricted problems w/ generating columns and cutting rows
    optimizer.output.subsection("Exact")
    iteration = 0
    while True:
        optimizer.output.iteration(iteration, "Iteration")

        dual_result = optimizer.solve_dual_for_branch(branch)
        primal_result = optimizer.solve_primal_for_branch(
            branch=branch,
            sol_type="exact",
            compute_farmer_welfare=False,
            compute_intermediary_welfare=False
        )

        optimizer.output.metric("Dual objective", dual_result.objective_value)
        optimizer.output.metric("Primal objective", primal_result.platform_profit)
        optimizer.output.metric("Columns added", dual_result.n_added_columns, precision=0)
        optimizer.output.metric("Rows added", primal_result.primary_n_added_rows, precision=0)

        if dual_result.n_added_columns == 0 and primal_result.primary_n_added_rows == 0:
            break
        else:
            iteration += 1

    optimizer.output.subsection("Restricted Master Solution")
    optimizer.output.metric("Dual objective", dual_result.objective_value)
    optimizer.output.metric("Primal objective", primal_result.platform_profit)

    # prune branch early if primal relaxed solution cannot beat existing integer solution
    can_improve = optimizer.exceeds_global_lb(
        primal_result.platform_profit, optimizer.BRANCH_PRUNE_TOL
    )
    optimizer.output.metric("Global lower bound", optimizer.best_lb)
    optimizer.output.metric("Improvement tolerance", optimizer.BRANCH_PRUNE_TOL)
    optimizer.output.metric("Decision", "Retain" if can_improve else "Prune")

    if not can_improve:
        optimizer.output.message(
            "Reason: branch LP relaxation bound cannot improve on the global lower bound.",
            indent=1,
        )
        return BranchSolution(status="stop", branch=branch)

    # check if solution is integral by checking marginal intermediary probabilities
    solution_is_integral = True
    intermediary_probabilities = {}
    intermediary_set_probabilities = primal_result.primary_result.intermediary_set_probabilities
    for intermediary_id in optimizer.intermediary_ids:
        probability = 0.0
        for intermediary_set in intermediary_set_probabilities:
            if intermediary_id in intermediary_set:
                probability += intermediary_set_probabilities[intermediary_set]

        intermediary_probabilities[intermediary_id] = probability

    for intermediary_id in optimizer.intermediary_ids:
        if optimizer.INT_TOL < intermediary_probabilities[intermediary_id] < 1 - optimizer.INT_TOL:
            solution_is_integral = False
            break

    # branch if fractional, update bounds if integral
    if solution_is_integral:
        if optimizer.exceeds_global_lb(
            primal_result.platform_profit, optimizer.GLOBAL_LB_UPDATE_TOL
        ):
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

            optimizer.best_lb = primal_result.platform_profit
            optimizer.best_lb_set = frozenset(
                {
                    intermediary_id
                    for intermediary_id, probability in intermediary_probabilities.items()
                    if probability > 1 - optimizer.INT_TOL
                }
            )
            optimizer.best_lb_result = primal_result

            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Global Bound Update",
                status="Improved global lower bound by finding a better feasible solution",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill=".",
            )

        return BranchSolution(status="integral", branch=branch)
    else:
        fractional_probs = {
            intermediary_id: probability
            for intermediary_id, probability in intermediary_probabilities.items()
            if optimizer.INT_TOL < probability < 1 - optimizer.INT_TOL
        }

        branch_on = min(
            fractional_probs,
            key=lambda intermediary_id: abs(0.5 - fractional_probs[intermediary_id]),
        )

        return BranchSolution(
            status="fractional",
            branch=branch,
            branch_on=branch_on,
            branch_value=intermediary_probabilities[branch_on],
            upper_bound=primal_result.platform_profit,
        )


def relative_gap(lower_bound: float, upper_bound: float) -> float:
    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        return float("inf")

    denominator = max(abs(lower_bound), 1.0)
    return max(0.0, upper_bound - lower_bound) / denominator


def format_bound_transition(old: float, new: float, precision: int = 3) -> str:
    def format_value(value: float) -> str:
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{value:,.{precision}f}"

    return f"{format_value(old)} -> {format_value(new)}"


def print_bound_update(
    optimizer, *, title: str, status: str, previous_lb: float, previous_ub: float, fill: str = "-"
) -> None:
    new_lb = optimizer.best_lb
    new_ub = optimizer.best_ub

    previous_abs_gap = (
        previous_ub - previous_lb
        if np.isfinite(previous_lb) and np.isfinite(previous_ub)
        else float("inf")
    )
    new_abs_gap = new_ub - new_lb if np.isfinite(new_lb) and np.isfinite(new_ub) else float("inf")

    previous_rel_gap = relative_gap(previous_lb, previous_ub)
    new_rel_gap = relative_gap(new_lb, new_ub)

    optimizer.output.blank()
    optimizer.output.subsection(title, fill=fill)
    optimizer.output.status(status)
    optimizer.output.metric(
        "Lower bound",
        format_bound_transition(previous_lb, new_lb),
    )

    if np.isfinite(previous_lb) and np.isfinite(new_lb) and new_lb != previous_lb:
        optimizer.output.metric(
            "LB improvement",
            new_lb - previous_lb,
        )

    optimizer.output.metric(
        "Upper bound",
        format_bound_transition(previous_ub, new_ub),
    )

    if np.isfinite(previous_ub) and np.isfinite(new_ub) and new_ub != previous_ub:
        optimizer.output.metric(
            "UB reduction",
            previous_ub - new_ub,
        )

    optimizer.output.metric(
        "Absolute gap",
        format_bound_transition(previous_abs_gap, new_abs_gap),
    )

    optimizer.output.metric(
        "Relative gap (%)",
        f"{format_relative_gap(previous_rel_gap)} -> {format_relative_gap(new_rel_gap)}",
    )


def format_relative_gap(value: float) -> str:
    return f"{100 * value:.3f}%" if np.isfinite(value) else "undefined"