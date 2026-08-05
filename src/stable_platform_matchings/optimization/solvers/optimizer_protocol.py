from typing import Protocol

from ..options import OptimizerParams, SolverOptions
from ...reporting.containers import DualSolution, InstanceSummary, PrimalSolution
from ...reporting.printer import Printer
from ..branch import Branch


class OptimizerProtocol(Protocol):
    best_lb: float
    best_ub: float
    best_lb_set: frozenset[str] | None
    best_lb_summary: PrimalSolution | None

    oracle_calls: int
    options: SolverOptions
    output: Printer
    instance_summary: InstanceSummary
    intermediary_ids: list[str]

    INT_TOL = 1e-9
    BRANCH_PRUNE_TOL = 1.0
    GLOBAL_LB_UPDATE_TOL = 1e-9
    RANDOM_BRANCH_TOL = 1.0

    def initialize_branch(self, branch: Branch) -> bool: ...

    def solve_primal_for_branch(
        self,
        branch: Branch,
        sol_type: str,
    ) -> PrimalSolution: ...

    def solve_dual_for_branch(
        self,
        branch: Branch,
    ) -> DualSolution: ...

    def exceeds_global_lb(self, value: float, tolerance: float) -> bool: ...

    def record_summary(self) -> None: ...
