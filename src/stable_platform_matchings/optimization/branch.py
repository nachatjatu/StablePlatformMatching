from ..reporting.printer import Printer


class Branch:
    def __init__(
        self,
        forced_match: set[str] | frozenset[str],
        forced_unmatch: set[str] | frozenset[str],
    ) -> None:
        self.forced_match = frozenset(forced_match)
        self.forced_unmatch = frozenset(forced_unmatch)
        self.count_flag = True
        self.min_cost_set: frozenset[str] = frozenset()
        self.min_cost = float("inf")

    def print_(self, output: Printer) -> None:
        output.section("Branch Evaluation")
        output.collection(
            "Forced matched",
            sorted(self.forced_match),
        )
        output.collection(
            "Forced unmatched",
            sorted(self.forced_unmatch),
        )
