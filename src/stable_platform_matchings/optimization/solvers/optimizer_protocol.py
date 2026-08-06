from typing import Protocol
import numpy as np

from ...reporting.containers import InstanceSummary, BranchPrimalResult, BranchDualResult
from ...reporting.printer import Printer
from ..branch import Branch
from ..options import SolverOptions


class OptimizerProtocol(Protocol):
    best_lb: float
    best_ub: float
    best_lb_set: frozenset[str] | None
    best_lb_result: BranchPrimalResult | None

    oracle_calls: list[int]
    total_oracle_calls: int
    options: SolverOptions
    output: Printer
    instance_summary: InstanceSummary
    intermediary_ids: list[str]

    rng: np.random.Generator | None

    INT_TOL = 1e-9
    BRANCH_PRUNE_TOL = 1.0
    GLOBAL_LB_UPDATE_TOL = 1e-9
    RANDOM_BRANCH_TOL = 1.0

    def initialize_branch(self, branch: Branch) -> bool: ...

    def solve_primal_for_branch(
        self,
        branch: Branch,
        sol_type: str,
    ) -> BranchPrimalResult: ...

    def solve_dual_for_branch(
        self,
        branch: Branch,
    ) -> BranchDualResult: ...

    def exceeds_global_lb(self, value: float, tolerance: float) -> bool: ...

    def record_summary(self) -> None: ...
