import numpy as np

from ...reporting.containers import BranchSolution, PrimalSolution
from ..branch import Branch
from .optimizer_protocol import OptimizerProtocol


def solve_heuristic(optimizer: OptimizerProtocol, optimize: bool) -> PrimalSolution:
    """Entry point for a heuristic branch-and-price search.

    Parameters
    ----------
    optimize : bool
        If ``True`` perform profit optimization during branching, otherwise
        use random heuristics.

    Returns
    -------
    dict[str, object] | None
        Summary of the best lower-bound solution found.
    """
    root_branch = Branch(set(), set())
    branches_to_evaluate = [root_branch]
    active_branches = []

    while True:
        for branch in branches_to_evaluate:
            branch_solution = solve_branch_heuristic(optimizer, branch, optimize)
            if branch_solution.status in ["stop", "integral", "infeasible"]:
                continue
            elif branch_solution.status in ["heuristic"]:
                active_branches.append(branch_solution)

        if not active_branches:
            optimizer.best_ub = optimizer.best_lb

            optimizer.record_summary()

            optimizer.output.section("Search Complete")
            optimizer.output.status("All branches have been resolved or pruned")
            optimizer.output.metric("Best objective found", optimizer.best_lb)
            optimizer.output.status("Heuristic search complete")

            break

        if optimizer.options.get("early_stop", False):
            optimizer.best_ub = max(
                branch_solution.upper_bound for branch_solution in active_branches
            )
            optimizer.record_summary()
            optimizer.output.status("Early stopping requested")
            break

        active_branches = [
            branch_solution
            for branch_solution in active_branches
            if optimizer.branch_can_improve(branch_solution.upper_bound)
        ]

        if not active_branches:
            optimizer.best_ub = optimizer.best_lb

            optimizer.record_summary()

            optimizer.output.section("Search Complete")
            optimizer.output.status("All branches have been resolved or pruned")
            optimizer.output.metric("Best objective found", optimizer.best_lb)
            optimizer.output.status("Heuristic search complete")

            break

        optimizer.output.subsection("Active Branch Queue")
        optimizer.output.metric("Branches", len(active_branches), precision=0)
        optimizer.output.collection(
            "Upper bounds",
            [branch_solution.upper_bound for branch_solution in active_branches],
        )

        # choose max branch using max profit criterion
        max_branch = max(
            active_branches, key=lambda branch: branch.branch_profits[branch.branch_on]
        )

        # update global upper bound using max upper bound from active branches
        current_max_upper_bound = -float("inf")

        for branch in active_branches:
            if branch.upper_bound > current_max_upper_bound:
                current_max_upper_bound = branch.upper_bound

        if current_max_upper_bound < optimizer.best_ub:
            optimizer.best_ub = current_max_upper_bound

            optimizer.record_summary()

            current_gap = (optimizer.best_ub - optimizer.best_lb) / np.abs(optimizer.best_lb)

            optimizer.output.subsection("Global Bound Update")
            optimizer.output.metric("New upper bound", optimizer.best_ub)
            optimizer.output.metric("Current gap", current_gap, precision=6)
            optimizer.output.collection(
                "Upper-bound history", optimizer.instance_summary.upper_bounds
            )

        # pop the max branch from the active branches
        parent_branch = max_branch.branch
        active_branches.remove(max_branch)

        # branch on max active branch
        branch_on = max_branch.branch_on
        branch_value = max_branch.branch_profits[max_branch.branch_on]

        optimizer.output.section(
            f"Branching on {branch_on} with value = {branch_value:.4f}",
            fill="-",
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

        right_branch.count_flag = False

        branches_to_evaluate = [left_branch, right_branch]

    if optimizer.best_lb_summary is None:
        raise RuntimeError("No primal solution has been found.")

    return optimizer.best_lb_summary


def solve_exact(
    optimizer: OptimizerProtocol,
) -> PrimalSolution:
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
            branch_solution = solve_branch_exact(optimizer, branch)
            if branch_solution.status in ["stop", "integral", "infeasible"]:
                continue
            elif branch_solution.status in ["fractional"]:
                active_branches.append(branch_solution)

        if not active_branches:
            optimizer.best_ub = optimizer.best_lb

            optimizer.record_summary()

            optimizer.output.section("Search Complete")
            optimizer.output.status("All branches have been resolved or pruned")
            optimizer.output.metric("Optimal objective", optimizer.best_lb)
            optimizer.output.metric("Final gap", 0.0)

            break

        optimizer.output.subsection("Active Branch Queue")
        optimizer.output.metric("Branches", len(active_branches), precision=0)
        optimizer.output.collection(
            "Upper bounds",
            [branch_solution.upper_bound for branch_solution in active_branches],
        )

        active_branches = [
            branch_solution
            for branch_solution in active_branches
            if optimizer.branch_can_improve(branch_solution.upper_bound)
        ]

        if not active_branches:
            optimizer.best_ub = optimizer.best_lb

            optimizer.record_summary()

            optimizer.output.section("Search Complete")
            optimizer.output.status("All branches have been resolved or pruned")
            optimizer.output.metric("Optimal objective", optimizer.best_lb)
            optimizer.output.metric("Final gap", 0.0)

            break

        # choose max branch using max upper bound criterion
        max_branch = max(active_branches, key=lambda branch: branch.upper_bound)

        # update global upper bound using max upper bound from active branches
        current_max_upper_bound = -float("inf")

        for branch_solution in active_branches:
            if branch_solution.upper_bound > current_max_upper_bound:
                current_max_upper_bound = branch_solution.upper_bound

        if current_max_upper_bound < optimizer.best_ub:
            optimizer.best_ub = current_max_upper_bound

            optimizer.record_summary()

            optimizer.output.subsection("Global Bound Update")
            optimizer.output.metric("New upper bound", max_branch.upper_bound)
            optimizer.output.metric(
                "Current gap",
                (optimizer.best_ub - optimizer.best_lb) / np.abs(optimizer.best_lb),
                precision=6,
            )

        # pop the max branch from the active branches
        parent_branch = max_branch.branch
        active_branches.remove(max_branch)

        # branch on max active branch
        branch_on = max_branch.branch_on
        branch_value = max_branch.branch_value

        optimizer.output.section(
            f"Branching on {branch_on} with value = {branch_value:.4f}",
            fill="-",
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

    if optimizer.best_lb_summary is None:
        raise RuntimeError("No primal solution has been found.")

    return optimizer.best_lb_summary


def solve_branch_exact(optimizer: OptimizerProtocol, branch: Branch) -> BranchSolution:
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
    forced_lb_solution = optimizer.solve_primal_for_branch(
        branch, "forced_lower_bound"
    )  # PrimalSummary
    forced_lb_platform_profit = forced_lb_solution.platform_profit

    optimizer.output.subsection("Forced Lower Bound")
    optimizer.output.metric("Objective", forced_lb_platform_profit)
    optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

    # update global lower bound if forced lower bound is tighter
    if optimizer.improves_incumbent(forced_lb_platform_profit):
        optimizer.best_lb = forced_lb_platform_profit
        optimizer.best_lb_set = branch.min_cost_set
        optimizer.best_lb_summary = forced_lb_solution

        optimizer.record_summary()

        optimizer.output.status("Found a better lower bound through forcing")
        optimizer.output.metric("New best lower bound", forced_lb_platform_profit)
        optimizer.output.metric(
            "New gap", (optimizer.best_ub - optimizer.best_lb) / np.abs(optimizer.best_lb)
        )

    # compute upper bound UB^n using forced solution, LP relaxation
    forced_ub_solution = optimizer.solve_primal_for_branch(branch, "forced_upper_bound")

    optimizer.output.subsection("Forced Upper Bound")
    optimizer.output.metric("Objective", forced_ub_solution.platform_profit)
    optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

    # prune branch early if forced upper bound cannot beat existing integer solution
    if not optimizer.branch_can_improve(forced_ub_solution.platform_profit):
        optimizer.output.status(
            "Forced upper bound cannot improve the incumbent within tolerance; pruning branch"
        )
        return BranchSolution(status="stop", branch=branch)

    # otherwise, solve primal and dual restricted problems w/ generating columns and cutting rows
    iteration = 0
    while True:
        optimizer.output.iteration(iteration, "Exact Iteration")

        dual_solution = optimizer.solve_dual_for_branch(branch)
        primal_solution = optimizer.solve_primal_for_branch(branch, "exact")

        optimizer.output.metric("Dual objective", dual_solution.platform_profit)
        optimizer.output.metric("Primal objective", primal_solution.platform_profit)
        optimizer.output.metric("Columns added", dual_solution.n_added_cols, precision=0)
        optimizer.output.metric("Rows added", primal_solution.n_added_rows, precision=0)

        if dual_solution.n_added_cols == 0 and primal_solution.n_added_rows == 0:
            break
        else:
            iteration += 1

    optimizer.output.subsection("Restricted Master Solution")
    optimizer.output.metric("Dual objective", dual_solution.platform_profit)
    optimizer.output.metric("Primal objective", primal_solution.platform_profit)
    optimizer.output.collection("Intermediary probabilities", primal_solution.intermediary_probs)

    # prune branch early if primal relaxed solution cannot beat existing integer solution
    if not optimizer.branch_can_improve(primal_solution.platform_profit):
        optimizer.output.status(
            "Primal solution cannot improve the incumbent within tolerance; pruning branch"
        )
        return BranchSolution(status="stop", branch=branch)

    # check if solution is integral
    solution_is_integral = True
    for intermediary_id in primal_solution.intermediary_probs:
        if (
            primal_solution.intermediary_probs[intermediary_id] > optimizer.INT_TOL
            and primal_solution.intermediary_probs[intermediary_id] < 1 - optimizer.INT_TOL
        ):
            solution_is_integral = False
            break

    # branch if fractional, update bounds if integral
    if solution_is_integral:
        if optimizer.improves_incumbent(primal_solution.platform_profit):
            optimizer.best_lb = primal_solution.platform_profit
            optimizer.best_lb_set = frozenset(
                {
                    intermediary_id
                    for intermediary_id, probability in primal_solution.intermediary_probs.items()
                    if probability > 1 - optimizer.INT_TOL
                }
            )
            optimizer.best_lb_summary = primal_solution

            optimizer.record_summary()

            optimizer.output.status("Found a better integral incumbent")
            optimizer.output.metric("New best lower bound", optimizer.best_lb)

        return BranchSolution(status="integral", branch=branch)
    else:
        fractional_probs = {
            intermediary_id: probability
            for intermediary_id, probability in primal_solution.intermediary_probs.items()
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
            branch_value=primal_solution.intermediary_probs[branch_on],
            upper_bound=primal_solution.platform_profit,
        )


def solve_branch_heuristic(
    optimizer: OptimizerProtocol, branch: Branch, optimize: bool
) -> BranchSolution:
    """Heuristic version of branch evaluation used in greedy search.

    ``optimize`` toggles whether to use the updated intermediary profits.
    Returns a dict similar to :meth:`solve_branch_exact` but may propose
    heuristic branching decisions.
    """
    if not optimizer.initialize_branch(branch):
        return BranchSolution(status="infeasible", branch=branch)

    # compute lower bound LB^n using forced solution
    forced_lb_solution = optimizer.solve_primal_for_branch(branch, "forced_lower_bound")

    optimizer.output.subsection("Forced Lower Bound")
    optimizer.output.metric("Objective", forced_lb_solution.platform_profit)
    optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

    # heuristic:
    if len(branch.forced_match) == 0 and len(branch.forced_unmatch) == 0:
        optimizer.instance_summary.forced_lower_bound = forced_lb_solution.platform_profit
        optimizer.instance_summary.forced_cost = forced_lb_solution.expected_intermediary_costs

    # update global lower bound if forced lower bound is tighter
    if optimizer.improves_incumbent(forced_lb_solution.platform_profit):
        optimizer.best_lb = forced_lb_solution.platform_profit
        optimizer.best_lb_set = branch.min_cost_set
        optimizer.best_lb_summary = forced_lb_solution

        optimizer.record_summary()

        optimizer.output.status("Found a better lower bound through forcing")
        optimizer.output.metric("New best lower bound", forced_lb_solution.platform_profit)
        optimizer.output.metric(
            "New gap", (optimizer.best_ub - optimizer.best_lb) / np.abs(optimizer.best_lb)
        )

    # compute upper bound UB^n using forced solution, LP relaxation
    forced_ub_solution = optimizer.solve_primal_for_branch(branch, "forced_upper_bound")

    optimizer.output.subsection("Forced Upper Bound")
    optimizer.output.metric("Objective", forced_ub_solution.platform_profit)
    optimizer.output.collection("Minimum-cost set", sorted(branch.min_cost_set))

    # heuristic:
    if len(branch.forced_match) == 0 and len(branch.forced_unmatch) == 0:
        optimizer.instance_summary.forced_upper_bound = forced_ub_solution.platform_profit

    # prune branch early if forced upper bound cannot beat existing integer solution
    if not optimizer.branch_can_improve(forced_ub_solution.platform_profit):
        optimizer.output.status(
            "Forced upper bound cannot improve the incumbent within tolerance; pruning branch"
        )
        return BranchSolution(status="stop", branch=branch)

    # heuristic: optional optimize flag
    if forced_ub_solution.updated_intermediary_profits is None:
        raise RuntimeError("updated_intermediary_profits is None.")

    if optimize:
        intermediary_profits = forced_ub_solution.updated_intermediary_profits
    else:
        intermediary_profits = {
            intermediary_id: np.random.uniform(0, 1)
            if forced_ub_solution.updated_intermediary_profits[intermediary_id]
            > optimizer.BRANCH_SCORE_TOL
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
        branch_profits=intermediary_profits,
        upper_bound=forced_ub_solution.platform_profit,
    )
