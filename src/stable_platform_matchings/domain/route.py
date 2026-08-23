from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

from .entities import Farmer

if TYPE_CHECKING:
    from .instance import Instance


class Route:
    """
    Tree-based route visiting a selected set of farmers.

    Attributes:
        farmers (list[Farmer]): list of farmers on the route.
        total_quantity (float): total farmer quantity of farmers on the route.
        instance (Instance): the current platform instance.
        cost (float): total cost of the route.
        is_feasible (bool): indicates whether the route meets truck capacity constraint.
        value (float): intermediary profit (revenue minus cost) from the route.
    """

    CAPACITY_TOL = 1e-6

    def __init__(self, farmers: list[Farmer], instance: Instance) -> None:
        """
        Initialize a route and calculate its quantity, cost, and value.

        Args:
            farmers (list[Farmer]): list of farmers on the route.
            instance (Instance): the current platform instance.
        """

        self.farmers = farmers
        self.total_quantity = sum(farmer.quantity for farmer in farmers)
        self.instance = instance
        self.cost = self.calculate_route_tree_cost()
        self.is_feasible = (
            self.total_quantity <= self.instance.truck_capacity_tons + Route.CAPACITY_TOL
        )
        self.value = self.total_quantity * self.instance.fruit_price_per_ton - self.cost

    def calculate_route_tree_cost(self) -> float:
        """
        Calculate the total cost of the route.

        Raises:
            RuntimeError: tree graph is not initialized.

        Returns:
            float: fixed + travel costs by the intermediary on the route.
        """

        if not self.farmers:
            return 0

        if self.instance.tree is None:
            raise RuntimeError("tree graph is not initialized.")

        # start with truck fixed cost
        cost = self.instance.truck_fixed_cost

        # add cost of traversing each node
        graph_nodes = (
            [self.instance.entity_id_to_graph_node[self.instance.mill.id]]
            + [self.instance.entity_id_to_graph_node[farmer.id] for farmer in self.farmers]
            + [self.instance.entity_id_to_graph_node[self.instance.mill.id]]
        )
        for node_idx in range(len(graph_nodes) - 1):
            cost += (
                nx.shortest_path_length(
                    self.instance.tree,
                    source=graph_nodes[node_idx],
                    target=graph_nodes[node_idx + 1],
                    weight="weight",
                )
                * self.instance.truck_cost_per_m
            )

        return cost

    def __repr__(self) -> str:
        """
        Returns a human-readable representation of route data.

        Returns:
            str: string representation of route data.
        """

        return f"Route(farmers={[farmer.id for farmer in self.farmers]})"
