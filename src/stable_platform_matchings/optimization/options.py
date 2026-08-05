from dataclasses import dataclass
from numbers import Real
from typing import Literal, Mapping

from ..domain.instance import Instance

Backend = Literal["gurobi"]
VRPMode = Literal["exact", "approximate"]
SolverStrategy = Literal["exact", "heuristic_optimized", "heuristic_unoptimized"]


@dataclass(frozen=True, slots=True)
class OptimizerParams:
    het_costs: Mapping[str, float]
    epsilons: Mapping[str, float]

    backend: Backend = "gurobi"
    vrp_mode: VRPMode = "approximate"
    verbose: bool = True
    print_width: int = 80

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

        for intermediary_id, epsilon in self.epsilons.items():
            if not isinstance(epsilon, Real) or isinstance(epsilon, bool):
                raise TypeError(f"epsilon[{intermediary_id!r}] must be numeric")

    def validate(self, instance: Instance) -> None:
        intermediary_ids = {intermediary.id for intermediary in instance.intermediaries}
        het_cost_ids = set(self.het_costs)
        epsilon_ids = set(self.epsilons)

        # check that each intermediary has well-defined heterogeneous cost
        if het_cost_ids != intermediary_ids:
            missing = intermediary_ids - het_cost_ids
            extra = het_cost_ids - intermediary_ids
            raise ValueError(
                "het_costs must contain exactly the intermediary IDs; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        # check that each intermediary has well-defined epsilon
        if epsilon_ids != intermediary_ids:
            missing = intermediary_ids - epsilon_ids
            extra = epsilon_ids - intermediary_ids
            raise ValueError(
                "epsilon must contain exactly the intermediary IDs; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        # only support Gurobi for now
        if self.backend != "gurobi":
            raise ValueError(f"Unsupported backend: {self.backend!r}")
        # only exact and approximate VRP supported
        if self.vrp_mode not in {"exact", "approximate"}:
            raise ValueError(f"Unsupported VRP mode: {self.vrp_mode!r}")
        # check that print_width is positive
        if self.print_width <= 0:
            raise ValueError("print_width must be positive.")

        # validate heterogenous costs
        for intermediary_id, cost in self.het_costs.items():
            if not isinstance(cost, int | float):
                raise TypeError(f"het_costs[{intermediary_id!r}] must be numeric.")

            if cost + instance.truck_fixed_cost < 0:
                raise ValueError(f"Total intermediary cost is negative for {intermediary_id!r}.")
        # validate epsilon
        for intermediary_id, epsilon in self.epsilons.items():
            if not isinstance(epsilon, int | float):
                raise TypeError(f"epsilon[{intermediary_id!r}] must be numeric.")
            if epsilon < 0:
                raise ValueError(f"epsilon[{intermediary_id!r}] must be nonnegative.")


@dataclass(frozen=True, slots=True)
class SolverOptions:
    strategy: SolverStrategy = "heuristic_optimized"
    structured_farmer_payments: bool = False
    dominance_constraints: bool = False
    early_stop: bool = False
    aggregate: bool = False
    pay_unmatched: bool = False

    def __post_init__(self) -> None:

        value = self.strategy

        if type(value) is not str:
            raise TypeError(f"strategy must be str, got {type(value).__name__}")

        if value not in {"exact", "heuristic_optimized", "heuristic_unoptimized"}:
            raise ValueError(f"Unsupported strategy: {value}")

        for name in (
            "structured_farmer_payments",
            "dominance_constraints",
            "early_stop",
            "aggregate",
            "pay_unmatched",
        ):
            value = getattr(self, name)

            if type(value) is not bool:
                raise TypeError(f"{name} must be bool, got {type(value).__name__}")

    def validate_for_strategy(self, strategy: str) -> None:
        if strategy == "exact" and self.pay_unmatched:
            raise ValueError(
                "The exact strategy does not support "
                "pay_unmatched=True because dual column generation "
                "is unavailable for that formulation."
            )
