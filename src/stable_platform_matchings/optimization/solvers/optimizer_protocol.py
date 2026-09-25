from typing import Mapping, Protocol
import numpy as np

from ...reporting.containers import InstanceSummary, BranchPrimalResult, BranchDualResult
from ...reporting.printer import Printer
from ..branch import Branch
from ..options import SolverOptions, OptimizerParams
from ...domain.instance import Instance


class OptimizerProtocol(Protocol):
    best_lb: float
    best_ub: float
    best_lb_set: frozenset[str] | None
    best_lb_result: BranchPrimalResult | None

    oracle_calls: list[int]
    total_oracle_calls: int
    options: SolverOptions
    instance_summary: InstanceSummary

    params: OptimizerParams
    output: Printer
    instance: Instance
    n_farmers: int
    n_intermediaries: int

    farmer_ids: list[str]
    intermediary_ids: list[str]

    het_costs: Mapping[str, float]

    # set per solve, not at construction
    epsilons: Mapping[str, float]
    status_quo_quantities: dict[str, float]


    rng: np.random.Generator | None

    INT_TOL = 1e-9
    BRANCH_PRUNE_TOL = 1.0
    GLOBAL_LB_UPDATE_TOL = 1e-9
    RANDOM_BRANCH_TOL = 1.0

    def initialize_branch(self, branch: Branch) -> bool: ...

    def intermediary_set_cost(self, intermediary_set: frozenset[str]) -> float | None: ...

    def solve_primal_for_branch(
        self,
        branch: Branch,
        sol_type: str,
        compute_intermediary_welfare: bool,
        compute_farmer_welfare: bool
    ) -> BranchPrimalResult: ...

    def solve_dual_for_branch(
        self,
        branch: Branch,
    ) -> BranchDualResult: ...

    def exceeds_global_lb(self, value: float, tolerance: float) -> bool: ...

    def record_summary(self) -> None: ...

