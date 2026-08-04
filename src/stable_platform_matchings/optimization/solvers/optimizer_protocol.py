from typing import Protocol

from ...reporting.containers import DualSolution, InstanceSummary, PrimalSolution
from ...reporting.printer import Printer
from ..branch import Branch


class OptimizerProtocol(Protocol):
    best_lb: float
    best_ub: float
    best_lb_set: frozenset[str] | None
    best_lb_summary: PrimalSolution | None

    oracle_calls: int
    options: dict[str, bool]
    output: Printer
    instance_summary: InstanceSummary
    intermediary_ids: list[str]

    INT_TOL: float
    BRANCH_SCORE_TOL: float

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

    def branch_can_improve(self, upper_bound: float) -> bool: ...

    def improves_incumbent(self, candidate: float) -> bool: ...

    def record_summary(self) -> None: ...
