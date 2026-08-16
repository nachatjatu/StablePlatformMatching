from __future__ import annotations

import os
import pickle
from collections import defaultdict
from collections.abc import Hashable
from typing import TypeAlias
from pathlib import Path
import math

import networkx as nx
import yaml

from ..graphs.road_graphs import RoadGraph
from .entities import Entity, Farmer, Intermediary, Mill
from .route import Route

GraphNode: TypeAlias = Hashable
TreeEdge: TypeAlias = tuple[GraphNode, GraphNode]


class Instance:
    """
    Platform instance containing farmers, intermediaries, mill, and graph data.

    Attributes:
        instance_id (str): identifier associated with instance.
        source (str | None): metadata, records instance data source.

        farmers (list[Farmer]): list of participating farmers.
        intermediaries (list[Intermediary]): list of participating intermediaries.
        mill (Mill): target mill receiving produce.
        entities (list[Entity]): list of farmers, intermediaries, and mill.

        truck_capacity_tons (float): maximum truck capacity in tons.
        truck_fixed_cost (float): fixed cost per truck.
        truck_cost_per_m (float): truck cost per meter.
        fruit_price_per_ton (float): fruit price per ton.
        lc_to_usd (float): exchange rate, amount of local currency per 1 USD.

        mill_key (str): ID associated with target mill.
        farmer_by_id (dict[str, Farmer]): dict mapping farmer ID to farmer.
            `farmer_by_id[f]` is the Farmer with ID `f`.

        dist_to_mill (dict[str, float]): dict associating an Entity ID with
            its distance to the mill.
        dirt_to_mill (dict[str, float]): same as dist_to_mill but using dirt roads.
        paved_to_mill (dict[str, float]): same as dist_to_mill but using paved roads.

        graph (RoadGraph): graph of platform road network.
        tree (nx.Graph): tree graph derived from platform road network.
        tree_order (list[GraphNode]): list containing nodes in tree order.
        tree_edges (list[TreeEdge]): unique edges in the derived tree.
        root_edges (dict[str, list[TreeEdge]]): maps each entity ID to
            the tree edges on its path to the root.
        child_to_parent (dict[GraphNode, GraphNode]): maps each non-root graph node to its
            parent graph node.
        edge_to_idx (dict[TreeEdge, int]): maps each tree edge to its
            internal index.
        edge_to_root_farmers (dict[TreeEdge, list[Farmer]]): maps each
            tree edge to farmers whose root path contains that edge.
        entity_id_to_graph_node (dict[str, GraphNode]): maps each entity ID to its associated
            OSMNX graph node ID.
        graph_node_to_entity_ids (dict[GraphNode, list[str]]): maps each OSMNX graph node
            ID to a list of associated entity IDs.
    """

    FRUIT_PRICE_PER_KG = 2513  # local currency (IDR)
    FRUIT_PRICE_PER_TON = FRUIT_PRICE_PER_KG * 1000

    TRUCK_CAPACITY_TONS = 9.0  # default value for capacity constraint
    TRUCK_FIXED_COST = 800000  # default value for truck fixed cost, local currency
    TRUCK_COST_PER_KM = 2625  # (2625.0 + 2065.0) / 2.0, local currency
    TRUCK_COST_PER_M = TRUCK_COST_PER_KM / 1000

    MILL_LOC = [-0.682643, 102.501522]

    LC_TO_USD = 14500  # amount of local currency per 1 USD (IDR)

    def __init__(
        self,
        instance_id: str,
        farmers: list[Farmer],
        intermediaries: list[Intermediary],
        mill: Mill,
    ) -> None:
        """
        Initialize a platform instance and validate required farmers.

        Args:
            instance_id (str): identifier associated with instance.
            farmers (list[Farmer]): list of farmers participating in the matching.
            intermediaries (list[Intermediary]): list of participating intermediaries.
            mill (Mill): target mill receiving produce.

        Raises:
            ValueError: if entity IDs are not unique.
        """

        # store metadata
        self.instance_id = instance_id
        self.source = None

        # store participant Entities
        self.farmers = farmers
        self.intermediaries = intermediaries
        self.mill = mill
        self.entities = farmers + intermediaries + [mill]
        self.mill_key = mill.id

        # validate entity names are unique
        entity_ids = [entity.id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Entity IDs must be unique across the instance.")

        # store instance constants
        self.truck_capacity_tons = Instance.TRUCK_CAPACITY_TONS
        self.truck_fixed_cost = Instance.TRUCK_FIXED_COST
        self.truck_cost_per_m = Instance.TRUCK_COST_PER_M
        self.fruit_price_per_ton = Instance.FRUIT_PRICE_PER_TON
        self.lc_to_usd = Instance.LC_TO_USD

        # misc.
        self.farmer_by_id = {farmer.id: farmer for farmer in farmers}

        # store distances/cost to mill
        self.dist_to_mill, self.dirt_to_mill, self.paved_to_mill = {}, {}, {}

        # store graph data
        self.graph, self.tree = None, None
        self.tree_order, self.tree_edges = [], []
        self.root_edges = {}
        self.child_to_parent = {}
        self.edge_to_idx = {}
        self.edge_to_root_farmers = {}
        self.entity_id_to_graph_node, self.graph_node_to_entity_ids = {}, {}

        self.status_quo_quantities = self._calculate_avg_hist_quantities()


    def to_snapshot(self) -> dict[str, object]:
        def finite_or_none(value: float | None) -> float | None:
            if value is None or not math.isfinite(value):
                return None
            return float(value)

        return {
            "instance_id": self.instance_id,
            "source": self.source,
            "farmers": [
                {
                    "id": farmer.id,
                    "quantity": farmer.quantity,
                    "location": list(farmer.location),
                    "intermediary_id": farmer.intermediary_id,
                    "dist_to_mill": finite_or_none(
                        getattr(farmer, "dist_to_mill", None)
                    ),
                    "dirt_to_mill": finite_or_none(
                        getattr(farmer, "dirt_to_mill", None)
                    ),
                    "paved_to_mill": finite_or_none(
                        getattr(farmer, "paved_to_mill", None)
                    ),
                }
                for farmer in self.farmers
            ],
            "intermediaries": [
                {
                    "id": intermediary.id,
                    "capacity": intermediary.capacity,
                    "location": list(intermediary.location),
                    "hist_sets": [
                        sorted(hist_set)
                        for hist_set in intermediary.hist_sets
                    ],
                    "dist_to_mill": finite_or_none(
                        getattr(intermediary, "dist_to_mill", None)
                    ),
                    "dirt_to_mill": finite_or_none(
                        getattr(intermediary, "dirt_to_mill", None)
                    ),
                    "status_quo_quantity": self.status_quo_quantities[intermediary.id]
                }
                for intermediary in self.intermediaries
            ],
            "mill": {
                "id": self.mill.id,
                "location": list(self.mill.location),
            },
            "constants": {
                "truck_capacity_tons": self.truck_capacity_tons,
                "truck_fixed_cost": self.truck_fixed_cost,
                "truck_cost_per_m": self.truck_cost_per_m,
                "fruit_price_per_ton": self.fruit_price_per_ton,
                "lc_to_usd": self.lc_to_usd,
            },
            "distances": {
                "total": {
                    key: finite_or_none(value)
                    for key, value in self.dist_to_mill.items()
                },
                "dirt": {
                    key: finite_or_none(value)
                    for key, value in self.dirt_to_mill.items()
                },
                "paved": {
                    key: finite_or_none(value)
                    for key, value in self.paved_to_mill.items()
                },
            },
        }

    @classmethod
    def from_dict(
        cls, instance_dict: dict, force_quantities: dict[str, float] | None = None
    ) -> Instance:
        """
        Construct an instance from a dictionary representation.

        Args:
            instance_dict (dict[str, object]): dict representation of instance.
            force_quantities (dict[str, float]): dict mapping farmer IDs to preset quantities.
                `force_quantities[f]` is the preset quantity associated with farmer `f`.
        """

        force_quantities = {} if force_quantities is None else force_quantities

        # construct farmers list
        farmers = []
        for farmer_data in instance_dict["farmers"]:
            farmer_id = farmer_data["farmer_id"]
            quantity = (
                farmer_data["quantity"] if not force_quantities else force_quantities[farmer_id]
            )

            farmers.append(
                Farmer(
                    id=farmer_id,
                    quantity=quantity,
                    location=farmer_data["location"],
                    intermediary_id=farmer_data["intermediary_id"],
                )
            )

        farmer_by_id = {farmer.id: farmer for farmer in farmers}

        if len(farmer_by_id) != len(farmers):
            raise ValueError("Farmer IDs must be unique.")

        # construct intermediaries list
        intermediaries = []
        for intermediary_data in instance_dict["intermediaries"]:
            hist_sets = [frozenset(route) for route in intermediary_data["routes"]]
            if not hist_sets:
                hist_sets.append(frozenset())

            intermediaries.append(
                Intermediary(
                    id=intermediary_data["intermediary_id"],
                    capacity=intermediary_data["capacity"],
                    location=intermediary_data["location"],
                    hist_sets=hist_sets,
                )
            )

        # find mill
        mill = None
        for m in instance_dict["mills"]:
            if m["location"] == cls.MILL_LOC:
                mill = Mill(m["mill_id"], tuple(m["location"]))
                break
        if mill is None:
            raise ValueError(f"No mill with location {cls.MILL_LOC!r} was found.")

        return cls(
            instance_id=instance_dict["instance_id"],
            farmers=farmers,
            intermediaries=intermediaries,
            mill=mill,
        )

    @classmethod
    def from_yaml(
        cls, yaml_filepath: Path, force_quantities: dict[str, float] | None = None
    ) -> Instance:
        """Load an instance from a YAML file."""
        with open(yaml_filepath, "r") as file:
            instance_dict = yaml.safe_load(file)

        instance = cls.from_dict(instance_dict, force_quantities)
        instance.source = os.path.splitext(yaml_filepath)[0]
        return instance

    def calculate_tree_path(self, farmer_ids: list[str]) -> Route:
        """
        Build a tree-ordered route for the selected farmers.

        Args:
            farmer_ids (list[str]): list of farmer IDs to build route for.

        Raises:
            ValueError: farmer IDs are not unique.
            ValueError: farmer IDs do not match those in initialization.
            ValueError: not every farmer is reached in the tour.

        Returns:
            Route: route object containing farmers in farmer_ids.
        """

        # validate input IDs
        if len(farmer_ids) != len(set(farmer_ids)):
            raise ValueError("farmer_ids must not contain duplicates.")
        farmer_ids_set = set(farmer_ids)
        unknown_ids = farmer_ids_set - set(self.farmer_by_id)
        if unknown_ids:
            raise ValueError(f"Unknown farmer IDs: {unknown_ids}")

        # build tour using tree order.
        tour = []
        for graph_node in self.tree_order:
            entity_ids_in_graph_node = self.graph_node_to_entity_ids.get(graph_node, [])
            for entity_id in entity_ids_in_graph_node:
                if entity_id in farmer_ids_set:
                    tour.append(self.farmer_by_id[entity_id])

        if len(tour) != len(farmer_ids):
            raise ValueError("Some farmers were not reached in the tour.")

        return Route(tour, self)

    def set_graph(self, graph: RoadGraph) -> None:
        """
        Attach a road graph and precompute tree and distance data in-place.

        Args:
            graph (RoadGraph): graph of road network for platform instance.
        """

        self.graph = graph
        self._precompute_mappings()
        self.tree, list_root_edges = self.graph.build_tree(
            list(self.entity_id_to_graph_node.keys()),
            list(self.entity_id_to_graph_node.values()),
            self.mill_key,
            plot=False,
        )

        # create list of tree edges
        self.root_edges = dict(
            zip(self.entity_id_to_graph_node.keys(), list_root_edges, strict=True)
        )

        seen = {}
        for f_id in self.root_edges:
            for edge in self.root_edges[f_id]:
                seen[edge] = None
        self.tree_edges = list(seen)

        # create tree ordering using DFS
        self.tree_order = list(nx.dfs_preorder_nodes(self.tree, source=self.mill_key))

        # cache node parents
        self.child_to_parent = nx.dfs_predecessors(self.tree, source=self.mill_key)

        # cache convenient lookup tables
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.tree_edges)}
        self.edge_to_root_farmers = {edge: [] for edge in self.tree_edges}
        for farmer in self.farmers:
            for edge in self.root_edges[farmer.id]:
                self.edge_to_root_farmers[edge].append(farmer)

        def _compute_dist_to_mill(
            entity_node: GraphNode, entity: Entity, weight: str = "weight"
        ) -> None:
            """
            Computes distance from entity to mill using various weights
            and updates internal Instance and Entity attributes in-place.

            Args:
                entity_node (GraphNode): OSM ID of entity node.
                entity (Entity): entity whose distance is to be calculated.
                weight (str, optional): the type of road to use in computation.
                    can be any of ["weight", "weight_dirt", "weight_paved"].
                    defaults to "weight".

            Raises:
                ValueError: weight type not in allowed options.
                RuntimeError: graph tree not initialized
            """

            if weight not in ["weight", "weight_dirt", "weight_paved"]:
                raise ValueError(f"Expected weight, weight_dirt, or weight_paved, got {weight}.")

            if self.tree is None:
                raise RuntimeError("graph tree not initialized.")

            dist = (
                nx.shortest_path_length(
                    self.tree,
                    source=entity_node,
                    target=self.entity_id_to_graph_node[self.mill.id],
                    weight=weight,
                )
                * self.truck_cost_per_m
            )

            if weight == "weight":
                self.dist_to_mill[entity.id] = dist
                entity.dist_to_mill = dist
            elif weight == "weight_dirt":
                self.dirt_to_mill[entity.id] = dist
                entity.dirt_to_mill = dist
            else:
                self.paved_to_mill[entity.id] = dist
                entity.paved_to_mill = dist

        self.dist_to_mill = {}
        self.dirt_to_mill = {}
        self.paved_to_mill = {}

        # calculate distances to mill for each intermediary
        for intermediary in self.intermediaries:
            intermediary_node = self.entity_id_to_graph_node[intermediary.id]
            _compute_dist_to_mill(intermediary_node, intermediary, "weight")
            _compute_dist_to_mill(intermediary_node, intermediary, "weight_dirt")

        # calculate distances to mill for each farmer
        for farmer in self.farmers:
            farmer_node = self.entity_id_to_graph_node[farmer.id]
            _compute_dist_to_mill(farmer_node, farmer, "weight")
            _compute_dist_to_mill(farmer_node, farmer, "weight_dirt")
            _compute_dist_to_mill(farmer_node, farmer, "weight_paved")

    def save_graph_data(self, filepath: str) -> None:
        """
        Save precomputed graph and distance data to a pickle file.

        Args:
            filepath (str): path where graph is to be saved.
                this path should also include the file name.
        Raises:
            RuntimeError: graph data not initialized.
        """
        if self.graph is None or self.tree is None:
            raise RuntimeError("Graph data has not been initialized. Call set_graph() first.")

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "graph": self.graph,
                    "tree": self.tree,
                    "root_edges": self.root_edges,
                    "tree_edges": self.tree_edges,
                    "tree_order": self.tree_order,
                    "entity_id_to_graph_node": self.entity_id_to_graph_node,
                    "graph_node_to_entity_ids": self.graph_node_to_entity_ids,
                    "child_to_parent": self.child_to_parent,
                    "edge_to_idx": self.edge_to_idx,
                    "dist_to_mill": self.dist_to_mill,
                    "dirt_to_mill": self.dirt_to_mill,
                    "paved_to_mill": self.paved_to_mill,
                },
                f,
            )

    def load_graph_data(self, filepath: str) -> bool:
        """
        Load cached graph and distance data if the file exists.

        Args:
            filepath (str): path where graph is to be loaded from.
                this path should also include the file name.

        Returns:
            bool: True if successful, False if file not found.
        """

        if not os.path.exists(filepath):
            return False

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.graph = data["graph"]
        self.tree = data["tree"]
        self.root_edges = data["root_edges"]
        self.tree_edges = data["tree_edges"]
        self.tree_order = data["tree_order"]

        self.entity_id_to_graph_node = data["entity_id_to_graph_node"]
        self.graph_node_to_entity_ids = data["graph_node_to_entity_ids"]
        self.child_to_parent = data["child_to_parent"]
        self.edge_to_idx = data["edge_to_idx"]

        self.edge_to_root_farmers = {edge: [] for edge in self.tree_edges}

        for farmer in self.farmers:
            for edge in self.root_edges[farmer.id]:
                self.edge_to_root_farmers[edge].append(farmer)

        self.dist_to_mill = data["dist_to_mill"]
        self.dirt_to_mill = data["dirt_to_mill"]
        self.paved_to_mill = data["paved_to_mill"]

        self._restore_distance_attributes()

        return True

    def _restore_distance_attributes(self) -> None:
        """Restore cached mill distances onto farmer and intermediary objects."""
        for intermediary in self.intermediaries:
            intermediary.dist_to_mill = self.dist_to_mill.get(intermediary.id, float("inf"))
            intermediary.dirt_to_mill = self.dirt_to_mill.get(intermediary.id, float("inf"))

        for farmer in self.farmers:
            farmer.dist_to_mill = self.dist_to_mill.get(farmer.id, float("inf"))
            farmer.dirt_to_mill = self.dirt_to_mill.get(farmer.id, float("inf"))
            farmer.paved_to_mill = self.paved_to_mill.get(farmer.id, float("inf"))

    def _precompute_mappings(self) -> None:
        """Map instance entity IDs to their nearest road-graph nodes."""
        locations = [entity.location for entity in self.entities]
        entity_ids = [entity.id for entity in self.entities]

        if self.graph is None:
            raise RuntimeError("graph not initialized.")

        # snap entity location to nearest graph node
        self.entity_id_to_graph_node = dict(
            zip(entity_ids, self.graph.get_closest_points(locations), strict=True)
        )

        graph_node_to_entity_ids = defaultdict(list)
        # map graph node to list of entity IDs associated with it
        for entity_id, graph_node in self.entity_id_to_graph_node.items():
            graph_node_to_entity_ids[graph_node].append(entity_id)
        self.graph_node_to_entity_ids = dict(graph_node_to_entity_ids)

    def _calculate_avg_hist_quantities(self) -> dict[str, float]:
        """
        Calculate average historical quantities for each intermediary.

        Returns:
            dict[str, float]: dict that maps intermediary ID to avg. quantity.
        """

        avg_hist_quantities = {}
        for intermediary in self.intermediaries:
            if not intermediary.hist_sets:
                avg_hist_quantities[intermediary.id] = 0.0
                continue

            # calculate total quantity across all historical sets
            total_hist_quantity = sum(
                self.farmer_by_id[farmer_id].quantity
                for hist_set in intermediary.hist_sets
                for farmer_id in hist_set
            )
            # average it out by the number of sets
            avg_hist_quantities[intermediary.id] = total_hist_quantity / len(intermediary.hist_sets)

        return avg_hist_quantities

    def __repr__(self) -> str:
        """
        Returns a human-readable representation of platform instance data.

        Returns:
            str: string representation of platform instance data.
        """

        return f"PlatformInstance(id={self.instance_id})"
