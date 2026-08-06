from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from ..domain.instance import Instance
from ..domain.matching import Matching

if TYPE_CHECKING:
    from ..optimization.branch import Branch
    from ..optimization.options import OptimizerParams


IntermediarySet: TypeAlias = frozenset[str]
IntermediarySetProbabilities: TypeAlias = dict[IntermediarySet, float]

@dataclass
class BranchSolution:
    status: str
    branch: Branch
    branch_on: str | None = None
    branch_value: float | None = None
    intermediary_profits: dict[str, float] | None = None
    upper_bound: float | None = None

@dataclass
class OptimizationResult:
    """Result of one optimization pass.

    The result may be integral or fractional. A concrete matching can only be
    recovered directly when `selected_intermediaries` is not None.
    """

    instance: Instance

    intermediary_set_probabilities: IntermediarySetProbabilities

    farmer_payments: dict[str, float]
    intermediary_profits: dict[str, float]

    platform_profit: float
    expected_intermediary_cost: float

    payment_per_quantity: float | None = None
    paved_distance_penalty: float | None = None
    dirt_distance_penalty: float | None = None

    INTEGRALITY_TOL: float = field(
        default=1e-9,
        init=False,
        repr=False,
    )

    @property
    def selected_intermediaries(self) -> IntermediarySet | None:
        """Return the selected intermediary set when the result is integral."""
        positive_sets = [
            (intermediary_set, probability)
            for intermediary_set, probability
            in self.intermediary_set_probabilities.items()
            if probability > self.INTEGRALITY_TOL
        ]

        if len(positive_sets) != 1:
            return None

        intermediary_set, probability = positive_sets[0]

        if probability < 1.0 - self.INTEGRALITY_TOL:
            return None

        return intermediary_set

    @property
    def is_integral(self) -> bool:
        return self.selected_intermediaries is not None

    @property
    def farmer_welfare(self) -> float:
        return sum(self.farmer_payments.values())

    @property
    def intermediary_welfare(self) -> float:
        return sum(self.intermediary_profits.values())

    def return_dict(self) -> dict[str, object]:
        return {
            "farmer_payments": self.farmer_payments,
            "intermediary_profits": self.intermediary_profits,
            "intermediary_set_probabilities": [
                {
                    "intermediaries": sorted(intermediary_set),
                    "probability": probability,
                }
                for intermediary_set, probability
                in sorted(
                    self.intermediary_set_probabilities.items(),
                    key=lambda item: sorted(item[0]),
                )
            ],
            "platform_profit": self.platform_profit,
            "selected_intermediaries": (
                sorted(self.selected_intermediaries)
                if self.selected_intermediaries is not None
                else None
            ),
            "is_integral": self.is_integral,
            "expected_intermediary_cost": self.expected_intermediary_cost,
            "farmer_welfare": self.farmer_welfare,
            "intermediary_welfare": self.intermediary_welfare,
            "payment_per_quantity": self.payment_per_quantity,
            "paved_distance_penalty": self.paved_distance_penalty,
            "dirt_distance_penalty": self.dirt_distance_penalty,
        }


@dataclass
class BranchPrimalResult:
    """Primary platform optimization and welfare reoptimizations."""

    primary_result: OptimizationResult
    n_added_rows: int

    max_intermediary_welfare_result: (
        OptimizationResult | None
    ) = None

    max_farmer_welfare_result: (
        OptimizationResult | None
    ) = None

    @property
    def platform_profit(self) -> float:
        return self.primary_result.platform_profit

    @property
    def selected_intermediaries(self) -> IntermediarySet | None:
        return self.primary_result.selected_intermediaries

@dataclass(frozen=True)
class BranchDualResult:
    objective_value: float
    n_added_columns: int



@dataclass
class PlatformOutcome:
    """Concrete implementable platform outcome with a recovered matching."""

    optimization_result: OptimizationResult
    matching: Matching

    def __post_init__(self) -> None:
        if not self.optimization_result.is_integral:
            raise ValueError(
                "Cannot create a concrete PlatformOutcome from a "
                "fractional optimization result."
            )

    @property
    def instance(self) -> Instance:
        return self.optimization_result.instance

    @property
    def selected_intermediaries(self) -> IntermediarySet:
        selected = self.optimization_result.selected_intermediaries

        if selected is None:
            raise RuntimeError("PlatformOutcome requires an integral result.")

        return selected

    @property
    def farmer_payments(self) -> dict[str, float]:
        return self.optimization_result.farmer_payments

    @property
    def intermediary_profits(self) -> dict[str, float]:
        return self.optimization_result.intermediary_profits

    @property
    def platform_profit(self) -> float:
        return self.optimization_result.platform_profit

@dataclass
class InstanceSummary:
    """Final platform outcome and optimization-run diagnostics."""

    instance: Instance
    params: OptimizerParams
    strategy: str

    start_time: float = field(default_factory=time.time)
    platform_solve_result: BranchPrimalResult | None = None
    total_time: float | None = None
    total_oracle_calls: int | None = None

    lower_bounds: list = field(default_factory=list)
    upper_bounds: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    oracle_calls: list = field(default_factory=list)

    forced_lower_bound: float | None = None
    forced_upper_bound: float | None = None