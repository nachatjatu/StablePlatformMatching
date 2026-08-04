from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..domain.instance import Instance
from ..domain.matching import Matching

if TYPE_CHECKING:
    from ..optimization.branch import Branch


@dataclass
class PlatformSolution:
    """Represents a final platform outcome including payments and matching.

    This is a lightweight container used by higher-level scripts to record
    platform-wide results.
    """

    instance: Instance
    matching: Matching
    farmer_payments: dict[str, float]
    intermediary_profits: dict[str, float]


class Solution:
    """
    Collects information about the solved instance.

    Attributes:
        instance (Instance): the current platform instance.
        farmer_payments (dict[str, float]): a dict storing payments to farmers.
            `farmer_payments[f]` is the payment associated with farmer with id `f`.
        intermediary_profits (dict[str, float]): a dict storing intermediary profits.
            `intermediary_profits[i]` is the profit associated with intermediary `i`.
        intermediary_probs (dict[str, float]): a dict storing intermediary selection probabilities.
            `intermediary_probs[i]` is the selection probability associated with intermediary `i`.
        platform_profit (float): the optimal platform profit from optimizing the instance.
        selected_set (set[str]): a set of IDs of selected intermediaries.
        expected_intermediary_costs (float): expected total cost borne by intermediaries.
        payment_per_quantity (float): value from optimized variable
        paved_distance_penalty (float): value from optimized variable
        dirt_distance_penalty (float):  value from optimized variable
    """

    MATCH_TOL = 1e-9

    def __init__(
        self,
        instance: Instance,
        intermediary_probs: dict[str, float | int],
        farmer_payments: dict[str, float],
        intermediary_profits: dict[str, float],
        platform_profit: float,
        expected_intermediary_costs: float,
    ) -> None:
        """
        Initializes the Solution

        Args:
            instance (Instance): the current platform instance.
            intermediary_probs (dict[str, float]): a dict storing intermediary
                selection probabilities.
            farmer_payments (dict[str, float]): a dict storing payments to farmers.
                `farmer_payments[f]` is the payment associated with farmer with id `f`.
            intermediary_profits (dict[str, float]): a dict storing intermediary profits.
                `intermediary_profits[i]` is the profit associated with intermediary `i`.
            platform_profit (float): the optimal platform profit from optimizing the instance.
            expected_intermediary_costs (float): expected total cost borne by intermediaries.
        """
        self.instance: Instance = instance
        self.intermediary_probs: dict[str, float] = intermediary_probs
        self.farmer_payments: dict[str, float] = farmer_payments
        self.intermediary_profits: dict[str, float] = intermediary_profits
        self.platform_profit: float = platform_profit
        self.expected_intermediary_costs: float = expected_intermediary_costs

        self.selected_set: set | None = set()
        self.payment_per_quantity: float | None = None
        self.paved_distance_penalty: float | None = None
        self.dirt_distance_penalty: float | None = None

        # classify each intermediary as selected (near 1) or not (near 0) or neither (fractional)
        for intermediary_id in intermediary_probs:
            if intermediary_probs[intermediary_id] > 1 - Solution.MATCH_TOL:
                self.selected_set.add(intermediary_id)
            elif intermediary_probs[intermediary_id] < Solution.MATCH_TOL:
                continue
            else:
                self.selected_set = None
                break

    def farmer_welfare(self) -> float:
        return sum(self.farmer_payments[farmer.id] for farmer in self.instance.farmers)

    def intermediary_welfare(self) -> float:
        return sum(
            self.intermediary_profits[intermediary_id]
            for intermediary_id in self.intermediary_profits
        )

    def return_dict(self) -> dict[str, object]:
        data = {
            "farmer_payments": self.farmer_payments,
            "intermediary_profits": self.intermediary_profits,
            "intermediary_probs": self.intermediary_probs,
            "platform_profit": self.platform_profit,
            "selected_set": (sorted(self.selected_set) if self.selected_set is not None else None),
            "expected_intermediary_costs": self.expected_intermediary_costs,
            "farmer_welfare": self.farmer_welfare(),
            "intermediary_welfare": self.intermediary_welfare(),
            "payment_per_quantity": self.payment_per_quantity,
            "paved_distance_penalty": self.paved_distance_penalty,
            "dirt_distance_penalty": self.dirt_distance_penalty,
        }
        return data


@dataclass
class PrimalSolution:
    platform_profit: float
    n_added_rows: int
    intermediary_probs: dict[str, float]
    initial_intermediary_profits: dict[str, float]
    expected_intermediary_costs: float

    max_intermediary_welfare: float | None = None
    min_farmer_welfare: float | None = None
    max_intermediary_welfare_solution: Solution | None = None

    max_farmer_welfare: float | None = None
    min_intermediary_welfare: float | None = None
    updated_intermediary_profits: dict[str, float] | None = None
    max_farmer_welfare_solution: Solution | None = None


@dataclass
class DualSolution:
    platform_profit: float
    n_added_cols: int


@dataclass
class BranchSolution:
    status: str
    branch: Branch
    branch_on: str | None = None
    branch_value: float | None = None
    branch_profits: dict[str, float] | None = None
    upper_bound: float | None = None


@dataclass
class InstanceSummary:
    instance: Instance
    parameters: dict[str, object]
    strategy: str

    start_time: float = field(default_factory=time.time)
    total_time: float | None = None
    timestamps: list[float] = field(default_factory=list)

    upper_bounds: list[float] = field(default_factory=list)
    lower_bounds: list[float] = field(default_factory=list)

    oracle_calls: list[int] = field(default_factory=list)
    total_oracle_calls: int | None = None

    max_intermediary_welfare_solution: Solution | None = None
    max_farmer_welfare_solution: Solution | None = None

    forced_lower_bound: float | None = None
    forced_upper_bound: float | None = None
    forced_cost: float | None = None
