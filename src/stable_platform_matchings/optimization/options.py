import warnings
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Mapping
import gurobipy as gp

from ..domain.instance import Instance

Backend = Literal["gurobi"]
VRPMode = Literal["exact", "approximate"]
SolverStrategy = Literal[
    "paper_bnb", "paper_bnb_random", "lagrangian_bnp",
    "npm_capped", "npm_uncapped", "enumeration"
]
SOLVER_STRATEGIES = frozenset({
    "paper_bnb", "paper_bnb_random", "lagrangian_bnp",
    "npm_capped", "npm_uncapped", "enumeration"
})

# deprecated strategy names, kept so that older scripts keep running
DEPRECATED_STRATEGY_ALIASES = {
    "heuristic_accelerated": "paper_bnb",
    "heuristic_vanilla": "paper_bnb_random",
    "exact": "lagrangian_bnp",
    "network_prioritized_capped": "npm_capped",
    "network_prioritized_uncapped": "npm_uncapped",
}


@dataclass(frozen=True, slots=True)
class OptimizerParams:
    het_costs: Mapping[str, float]

    backend: Backend = "gurobi"
    vrp_mode: VRPMode = "approximate"
    verbose: bool = True
    print_width: int = 80
    threads: int = 1
    vrp_time_limit_seconds: int | float = gp.GRB.INFINITY

    def __post_init__(self) -> None:
        if type(self.verbose) is not bool:
            raise TypeError("verbose must be bool")

        if type(self.print_width) is not int:
            raise TypeError("print_width must be int")

        if self.print_width <= 0:
            raise ValueError("print_width must be positive")

        for intermediary_id, cost in self.het_costs.items():
            if not isinstance(cost, Real) or isinstance(cost, bool):
                raise TypeError(f"het_costs[{intermediary_id!r}] must be numeric")

    def validate(self, instance: Instance) -> None:
        # only support Gurobi for now
        if self.backend != "gurobi":
            raise ValueError(f"Unsupported backend: {self.backend!r}")
        # only exact and approximate VRP supported
        if self.vrp_mode not in {"exact", "approximate"}:
            raise ValueError(f"Unsupported VRP mode: {self.vrp_mode!r}")
        # check that print_width is positive
        if self.print_width <= 0:
            raise ValueError("print_width must be positive.")

        if type(self.threads) is not int :
                    raise TypeError(f"threads must be int, got {type(self.threads).__name__}")
        
        if (
            not isinstance(self.vrp_time_limit_seconds, Real)
            or isinstance(self.vrp_time_limit_seconds, bool)
        ):
            raise TypeError("vrp_time_limit_seconds must be numeric")

        if self.vrp_time_limit_seconds <= 0:
            raise ValueError("vrp_time_limit_seconds must be positive")

        # validate heterogenous costs
        intermediary_ids = {intermediary.id for intermediary in instance.intermediaries}
        het_cost_ids = set(self.het_costs)

        if het_cost_ids != intermediary_ids:
            missing = intermediary_ids - het_cost_ids
            extra = het_cost_ids - intermediary_ids
            raise ValueError(
                "het_costs must contain exactly the intermediary IDs; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        for intermediary_id, cost in self.het_costs.items():
            if cost + instance.truck_fixed_cost < 0:
                raise ValueError(f"Total intermediary cost is negative for {intermediary_id!r}.")


def validate_epsilons(
    instance: Instance,
    epsilons: Mapping[str, float],
) -> None:
    """
    Validate the ambiguity levels supplied to a single solve.

    Ambiguity levels are per-solve rather than per-Optimizer because nothing cached on
    the Optimizer depends on them, so one Optimizer -- including its cached VRP routing
    costs and its heterogeneous-cost-dependent matching catalogue -- can be reused
    across an ambiguity sweep.

    Args:
        instance (Instance): the platform instance being solved.
        epsilons (Mapping[str, float]): maps intermediary ID to ambiguity level.

    Raises:
        ValueError: the ID set does not match the instance, or a level is negative.
        TypeError: a level is not numeric.
    """
    intermediary_ids = {intermediary.id for intermediary in instance.intermediaries}
    epsilon_ids = set(epsilons)

    if epsilon_ids != intermediary_ids:
        missing = intermediary_ids - epsilon_ids
        extra = epsilon_ids - intermediary_ids
        raise ValueError(
            "epsilons must contain exactly the intermediary IDs; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    for intermediary_id, epsilon in epsilons.items():
        if not isinstance(epsilon, Real) or isinstance(epsilon, bool):
            raise TypeError(f"epsilon[{intermediary_id!r}] must be numeric.")
        if epsilon < 0:
            raise ValueError(f"epsilon[{intermediary_id!r}] must be nonnegative.")


@dataclass(frozen=True, slots=True)
class SolverOptions:
    seed: int = 0
    strategy: SolverStrategy = "paper_bnb"
    structured_farmer_payments: bool = False
    dominance_constraints: bool = False
    early_stop_threshold: float = float("inf")
    hist_set_method: str = "instance_farmers"
    pay_unmatched: bool = False
    stabilize_final_solution: bool = True

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise TypeError(f"seed must be int, got {type(self.seed).__name__}")
        
        if type(self.strategy) is not str:
            raise TypeError(f"strategy must be str, got {type(self.strategy).__name__}")

        if self.strategy in DEPRECATED_STRATEGY_ALIASES:
            new_strategy = DEPRECATED_STRATEGY_ALIASES[self.strategy]
            warnings.warn(
                f"strategy {self.strategy!r} is deprecated; use {new_strategy!r}",
                DeprecationWarning,
                stacklevel=3,
            )
            object.__setattr__(self, "strategy", new_strategy)

        if self.strategy not in SOLVER_STRATEGIES:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        for name in (
            "structured_farmer_payments",
            "dominance_constraints",
            "pay_unmatched",
        ):
            value = getattr(self, name)

            if type(value) is not bool:
                raise TypeError(f"{name} must be bool, got {type(value).__name__}")