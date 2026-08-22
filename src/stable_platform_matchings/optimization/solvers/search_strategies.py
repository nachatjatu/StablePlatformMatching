import numpy as np

from ...reporting.containers import BranchSolution
from ..branch import Branch
from .optimizer_protocol import OptimizerProtocol


def solve_heuristic(
    optimizer: OptimizerProtocol, 
    heuristic_optimized: bool,
) -> None:
    
    root_branch = Branch(set(), set())
    branches_to_evaluate = [root_branch]
    active_branches = []

    while True:
        for branch in branches_to_evaluate:
            branch_solution = solve_branch_heuristic(
                optimizer=optimizer, 
                branch=branch, 
                heuristic_optimized=heuristic_optimized, 
            )
            if branch_solution.status in ["stop", "integral", "infeasible"]:
                continue
            elif branch_solution.status in ["heuristic"]:
                active_branches.append(branch_solution)

        if not active_branches:
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

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

        if optimizer.options.early_stop:
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub
            
            optimizer.best_ub = max(
                branch_solution.upper_bound for branch_solution in active_branches
            )
            
            optimizer.record_summary()

            print_bound_update(
                optimizer,
                title="Search Complete",
                status="Early stopping requested",
                previous_lb=previous_lb,
                previous_ub=previous_ub,
                fill="=",
            )
            break

        active_branches = [
            branch_solution
            for branch_solution in active_branches
            if optimizer.exceeds_global_lb(branch_solution.upper_bound, optimizer.BRANCH_PRUNE_TOL)
        ]

        if not active_branches:
            previous_lb = optimizer.best_lb
            previous_ub = optimizer.best_ub

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

        queue_summary = [
            {
                "matched": sorted(branch_solution.branch.forced_match),
                "unmatched": sorted(branch_solution.branch.forced_unmatch),
                "upper_bound": branch_solution.upper_bound,
                "can_improve": optimizer.exceeds_global_lb(
                    branch_solution.upper_bound, optimizer.BRANCH_PRUNE_TOL
                ),
            }
            for branch_solution in active_branches
        ]

        optimizer.output.collection("Branch summaries", queue_summary)

        # choose max branch using max profit criterion
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

        left_branch = Branch(parent_branch.forced_match | {branch_on}, parent_branch.forced_unmatch)
        right_branch = Branch(
            parent_branch.forced_match, parent_branch.forced_unmatch | {branch_on}
        )

        right_branch.count_flag = False

        branches_to_evaluate = [left_branch, right_branch]

    if optimizer.best_lb_result is None:
        raise RuntimeError("No primal solution has been found.")


def solve_branch_heuristic(
    optimizer: OptimizerProtocol, 
    branch: Branch, 
    heuristic_optimized: bool, 
) -> BranchSolution:

    if not optimizer.rng:
        raise RuntimeError("RNG not initialized.")
    
    if not optimizer.initialize_branch(branch):
        return BranchSolution(status="infeasible", branch=branch)

    # compute lower bound LB^n using forced solution
    forced_lb_result = optimizer.solve_primal_for_branch(
        branch=branch, 
        sol_type="forced_lower_bound",
        compute_farmer_welfare=False,
        compute_intermediary_welfare=False
    )

    optimizer.output.subsection("Lower-Bound Candidate")
    optimizer.output.metric("Objective", forced_lb_result.platform_profit)
    optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

    # heuristic:
    if not branch.forced_match and not branch.forced_unmatch:
        optimizer.instance_summary.forced_lower_bound = forced_lb_result.platform_profit

    # update global lower bound if forced lower bound is tighter
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

    # compute upper bound UB^n using forced solution, LP relaxation
    # (note that min cost set is optimal if you can pay unmatched)
    forced_ub_result = optimizer.solve_primal_for_branch(
        branch=branch, 
        sol_type="forced_upper_bound",
        compute_farmer_welfare=True,
        compute_intermediary_welfare=False
    )

    optimizer.output.subsection("Upper-Bound Candidate")
    optimizer.output.metric("Objective", forced_ub_result.platform_profit)
    optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

    # heuristic:
    if not branch.forced_match and not branch.forced_unmatch:
        optimizer.instance_summary.forced_upper_bound = forced_ub_result.platform_profit

    # prune branch early if forced upper bound cannot beat existing integer solution
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

    # heuristic: optional optimize flag
    if (
        forced_ub_result.max_farmer_welfare_result is None
        or forced_ub_result.max_farmer_welfare_result.intermediary_profits is None
    ):
        raise RuntimeError("updated_intermediary_profits is None.")

    max_farmer_welfare_int_profits = (
        forced_ub_result.max_farmer_welfare_result.intermediary_profits
    )

    if heuristic_optimized:
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
        status="heuristic",
        branch=branch,
        branch_on=branch_on,
        intermediary_profits=intermediary_profits,
        upper_bound=forced_ub_result.platform_profit,
    )


def solve_exact(
    optimizer: OptimizerProtocol
) -> None:
    """Perform a full exact branch-and-price solver.

    This method orchestrates branching on fractional variables and
    maintains global best bounds, returning a summary of the best
    integral solution found.
    """

    root_branch = Branch(set(), set())
    branches_to_evaluate = [root_branch]
    active_branches = []

    while True:
        for branch in branches_to_evaluate:
            branch_solution = solve_branch_exact(
                optimizer=optimizer, 
                branch=branch
            )
            if branch_solution.status in ["stop", "integral", "infeasible"]:
                continue
            elif branch_solution.status in ["fractional"]:
                active_branches.append(branch_solution)

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

        queue_summary = [
            {
                "matched": sorted(branch_solution.branch.forced_match),
                "unmatched": sorted(branch_solution.branch.forced_unmatch),
                "upper_bound": branch_solution.upper_bound,
                "can_improve": optimizer.exceeds_global_lb(
                    branch_solution.upper_bound, optimizer.BRANCH_PRUNE_TOL
                ),
            }
            for branch_solution in active_branches
        ]

        optimizer.output.collection("Branch summaries", queue_summary)

        active_branches = [
            branch_solution
            for branch_solution in active_branches
            if optimizer.exceeds_global_lb(branch_solution.upper_bound, optimizer.BRANCH_PRUNE_TOL)
        ]

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


def solve_branch_exact(
    optimizer: OptimizerProtocol, 
    branch: Branch
) -> BranchSolution:
    """Perform an exact branch-and-price iteration on a given branch.

    Parameters
    ----------
    branch : Branch
        Branching restrictions to apply (fixed matches/unmatches).

    Returns
    -------
    dict
        Information about the branch outcome including status codes,
        potential branching variable, and bound values.
    """
    if not optimizer.initialize_branch(branch):
        return BranchSolution(status="infeasible", branch=branch)

    # compute lower bound LB^n using forced solution
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

    # compute upper bound UB^n using forced solution, LP relaxation
    forced_ub_solution = optimizer.solve_primal_for_branch(
        branch=branch, 
        sol_type="forced_upper_bound",
        compute_farmer_welfare=False,
        compute_intermediary_welfare=False
    )

    optimizer.output.subsection("Forced Upper Bound")
    optimizer.output.metric("Objective", forced_ub_solution.platform_profit)
    optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

    # prune branch early if forced upper bound cannot beat existing integer solution
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
            "Reason: branch upper bound cannot improve on the global lower bound.",
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