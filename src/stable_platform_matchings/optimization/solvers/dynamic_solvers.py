import math

import numpy as np
import numpy.typing as npt

from ...domain.instance import Instance
from ...domain.route import Route


class PrizeMatrix:
    """
    Helper class for DP computation of best farmer subsets.

    Stores DP information on a tree derived from platform graph.
    Used by TSPSolver class to find high-value farmer sets without
    solving a full integer program.

    Attributes:
        instance (Instance): the current platform instance.
        q_max (int): max. scaled quantity, generally truck capacity.
        n_farmers (int): number of farmers in the instance.
        best_prizes (npt.NDArray[np.float64]): a `float64` array of shape `(n_nodes, q_max + 1)`.
            `best_prizes[i, q]` is the max. prize obtainable at node `i` for quantity `q`.
        farmer_used (npt.NDArray[np.bool_]): a Boolean array of shape
            `(n_nodes, q_max + 1, n_farmers)`. `farmer_used[i, q, f]` indicates whether
            farmer `f` is included in the solution represented by `best_prizes[i, q]`.
        farmer_idx_to_node_idx (dict[int, int]): a dict that maps farmer indices to node indices.
            `farmer_idx_to_node_idx[f]` is the node index associated with farmer `f`.
        ordering (list[int]): node indices in the order used to merge tree states
            during the dynamic-programming computation.
        quantity_by_farmer_idx (list[int]): scaled farmer quantities indexed by farmer index.
            `quantity_by_farmer_idx[f]` is the quantity associated with farmer `f`.
        parent_idx_by_child_idx (list[int]): parent node index indexed by child node index.
            `parent_idx_by_child_idx[i]` is the index of the parent of child node `i`.
        root_node_idx (int): the index of the root node in the tree.
        edge_costs (npt.NDArray[np.float64]): a `float64` array of shape `(n_nodes, n_nodes)`
            containing edge traversal costs. `edge_costs[i, j]` is the cost of the edge
            from node `i` to node `j`.
    """

    CANDIDATE_POOL_SIZE = 15

    def __init__(
        self,
        instance: Instance,
        q_max: int,
        n_nodes: int,
        farmer_idx_to_node_idx: dict[int, int],
        ordering: list[int],
        quantity_by_farmer_idx: list[int],
        parent_idx_by_child_idx: list[int],
        root_node_idx: int,
        edge_costs: npt.NDArray[np.float64],
    ) -> None:
        """
        Initializes a prize matrix for a given tree.
        **Assumes that each farmer is assigned a unique node**.

        Args:
            instance (Instance): current platform instance.
            q_max (int): max. scaled quantity, generally truck capacity.
            n_nodes (int): number of nodes in tree graph.
            farmer_idx_to_node_idx (dict[int, int]): maps farmer index to associated node index.
            ordering (list[int]): lists order of node indices for merging tree states during DP.
                node ordering should be bottom-up, that is, children appear before parents.
            quantity_by_farmer_idx (list[int]): maps farmer index to quantity produced.
            parent_idx_by_child_idx (list[int]): maps child index to parent index.
            root_node_idx (int): index of tree root node.
            edge_costs (npt.NDArray[np.float64]): edge cost matrix.

        Raises:
            ValueError: farmers are not mapped to unique nodes.
            ValueError: quantity is not in bounds [1, q_max]
        """

        self.instance = instance
        self.q_max = q_max
        self.n_farmers = len(self.instance.farmers)
        self.best_prizes = np.zeros((n_nodes, q_max + 1))
        self.farmer_used = np.zeros((n_nodes, q_max + 1, self.n_farmers), dtype=bool)
        self.farmer_idx_to_node_idx = farmer_idx_to_node_idx
        self.ordering = ordering
        self.quantity_by_farmer_idx = quantity_by_farmer_idx
        self.parent_idx_by_child_idx = parent_idx_by_child_idx
        self.root_node_idx = root_node_idx
        self.edge_costs = edge_costs

        node_indices = list(farmer_idx_to_node_idx.values())
        if len(node_indices) != len(set(node_indices)):
            raise ValueError("PrizeMatrix requires each farmer to be assigned to a unique node.")

        for farmer_idx, quantity in enumerate(quantity_by_farmer_idx):
            if not 1 <= quantity <= q_max:
                raise ValueError(
                    f"Scaled quantity for farmer {farmer_idx} must be in "
                    f"[1, {q_max}], got {quantity}."
                )

    def reset(self, prizes: list[float]) -> None:
        """
        Resets internal arrays in-place and seeds them using a given prize vector.
        **Assumes that farmers have positive scaled quantities**.

        Args:
            prizes (list[float]): list of prizes associated with each farmer index.
                `prizes[f]` is the prize associated with farmer `f`.

        Raises:
            ValueError: a farmer has a non-positive scaled quantity.

        Returns:
            None
        """

        self.best_prizes.fill(-np.inf)
        self.best_prizes[:, 0] = 0
        self.farmer_used.fill(False)

        for farmer_idx in range(len(prizes)):
            node_idx = self.farmer_idx_to_node_idx[farmer_idx]
            farmer_quantity = self.quantity_by_farmer_idx[farmer_idx]
            self.best_prizes[node_idx, farmer_quantity] = prizes[farmer_idx]
            self.farmer_used[node_idx, farmer_quantity, farmer_idx] = True

    def merge(self, child_idx: int, parent_idx: int, merged_idx: int, cost: float) -> None:
        """
        Combines two nodes in-place during recursive DP computation.

        Args:
            child_idx (int): index of child node.
            parent_idx (int): index of parent node.
            merged_idx (int): index of resulting merged node.
            cost (float): merge cost, generally the round-trip cost from child to parent.

        Returns:
            None
        """
        merged_prizes = np.full(self.q_max + 1, -np.inf)
        merged_prizes[0] = 0
        merged_farmers = np.zeros((self.q_max + 1, self.n_farmers), dtype=bool)

        for child_quantity in range(self.q_max + 1):
            child_prize = self.best_prizes[child_idx, child_quantity]

            child_feasible = child_quantity == 0 or (child_quantity > 0 and child_prize > -np.inf)

            if not child_feasible:
                continue

            effective_cost = cost if child_quantity > 0 else 0

            for parent_quantity in range(self.q_max + 1 - child_quantity):
                parent_prize = self.best_prizes[parent_idx, parent_quantity]

                parent_feasible = parent_quantity == 0 or (
                    parent_quantity > 0 and parent_prize > -np.inf
                )

                if not parent_feasible:
                    continue

                # combine quantities and prizes
                merged_quantity = child_quantity + parent_quantity
                merged_prize = child_prize + parent_prize - effective_cost

                if merged_prizes[merged_quantity] < merged_prize:
                    merged_prizes[merged_quantity] = merged_prize
                    merged_farmers[merged_quantity] = (
                        self.farmer_used[child_idx, child_quantity]
                        | self.farmer_used[parent_idx, parent_quantity]
                    )

        # update relevant matrices with merged values
        self.best_prizes[merged_idx] = merged_prizes
        self.farmer_used[merged_idx] = merged_farmers

    def merge2(self, child_idx: int, parent_idx: int, merged_idx: int, cost: float) -> None:
        """
        Combines two nodes in-place during recursive DP computation.

        Args:
            child_idx (int): index of child node.
            parent_idx (int): index of parent node.
            merged_idx (int): index of resulting merged node.
            cost (float): merge cost, generally the round-trip cost from child to parent.

        Returns:
            None
        """
        child_prizes = self.best_prizes[child_idx]
        parent_prizes = self.best_prizes[parent_idx]

        # feasibility masking
        child_feasible = np.zeros(self.q_max + 1, dtype=bool)
        child_feasible[0] = True
        child_feasible[1:] = child_prizes[1:] > -np.inf
        parent_feasible = np.zeros(self.q_max + 1, dtype=bool)
        parent_feasible[0] = True
        parent_feasible[1:] = parent_prizes[1:] > -np.inf

        # effective_cost applies only for quantity > 0
        child_adj = child_prizes.copy()
        child_adj[1:] -= cost
        child_adj[~child_feasible] = -np.inf

        parent_adj = parent_prizes.copy()
        parent_adj[~parent_feasible] = -np.inf

        merged_prizes = np.full(self.q_max + 1, -np.inf)
        merged_prizes[0] = 0
        merged_farmers = np.zeros((self.q_max + 1, self.n_farmers), dtype=bool)

        for merged_quantity in range(self.q_max + 1):
            child_quantities = np.arange(merged_quantity + 1)
            parent_quantities = merged_quantity - child_quantities 
            candidate_values = child_adj[child_quantities] + parent_adj[parent_quantities]

            best_local_idx = np.argmax(candidate_values)
            best_value = candidate_values[best_local_idx]

            if best_value > merged_prizes[merged_quantity]:
                merged_prizes[merged_quantity] = best_value
                merged_farmers[merged_quantity] = (
                    self.farmer_used[child_idx, child_quantities[best_local_idx]]
                    | self.farmer_used[parent_idx, parent_quantities[best_local_idx]]
                )

        # update relevant matrices with merged values
        self.best_prizes[merged_idx] = merged_prizes
        self.farmer_used[merged_idx] = merged_farmers

    def solve(
        self, prizes_by_farmer_idx: list[float], threshold: float
    ) -> tuple[list[float], list[npt.NDArray[np.intp]]]:
        """
        Evaluate the prize matrix and return top routes exceeding threshold.
        **Assumes that each farmer in the instance has an associated prize**.
        Returns [], [] if no valid routes exceed threshold.

        Args:
            prizes_by_farmer_idx (list[float]): list of prizes associated with each farmer index.
                `prizes_by_farmer_idx[f]` is the prize associated with farmer `f`.
            threshold (float): minimum objective value to consider.

        Raises:
            ValueError: if the length of `prizes` does not match the number of farmers.

        Returns:
            list[float]: list of route objectives sorted in descending order
                that exceed `threshold`.
            list[npt.NDArray[np.intp]]: list of corresponding farmer indices.
        """
        if len(prizes_by_farmer_idx) != self.n_farmers:
            raise ValueError(
                f"Prizes length mismatch: {len(prizes_by_farmer_idx)} != {self.n_farmers}"
            )

        # reset internal parameters
        self.reset(prizes_by_farmer_idx)

        # perform DFS of the tree
        for child_idx in self.ordering:
            if child_idx != self.root_node_idx:
                parent_idx = self.parent_idx_by_child_idx[child_idx]
                # self.merge(
                #     child_idx=child_idx,
                #     parent_idx=parent_idx,
                #     merged_idx=parent_idx,
                #     cost=self.edge_costs[child_idx, parent_idx]
                #     + self.edge_costs[parent_idx, child_idx],
                # )
                self.merge2(
                    child_idx=child_idx,
                    parent_idx=parent_idx,
                    merged_idx=parent_idx,
                    cost=self.edge_costs[child_idx, parent_idx]
                    + self.edge_costs[parent_idx, child_idx],
                )

        # extract all prizes
        total_prizes, farmer_idx_sets = [], []
        for quantity in range(1, self.q_max + 1):
            root_exceeds = self.best_prizes[self.root_node_idx, quantity] > threshold
            root_used = np.sum(self.farmer_used[self.root_node_idx, quantity]) > 0
            if root_exceeds and root_used:
                total_prizes.append(self.best_prizes[self.root_node_idx, quantity])
                farmer_idx_sets.append(np.where(self.farmer_used[self.root_node_idx, quantity])[0])

        if not total_prizes:
            return [], []

        # sort objectives in descending order
        candidates = sorted(
            zip(total_prizes, farmer_idx_sets, strict=True), key=lambda x: x[0], reverse=True
        )[: self.CANDIDATE_POOL_SIZE]

        best_total_prizes, best_farmer_idx_sets = zip(*candidates, strict=True)

        return list(best_total_prizes), list(best_farmer_idx_sets)


class DynamicTSPSolver:
    """
    Implements a fast prize-collecting TSP formulation using dynamic programming.

    Attributes:
        instance (Instance): the current platform instance.
        prize_matrix (PrizeMatrix): prize matrix corresponding to instance tree.
    """

    MULTIPLIER = 10
    OBJ_TOL = 1e-4
    ABS_TOL = 1e-9
    REL_TOL = 0.0
    TOP_N = 5

    def __init__(self, instance: Instance) -> None:
        """
        Initializes solver and precomputes prize matrix from instance tree.

        Args:
            instance (Instance): the current platform instance.
        """

        self.instance = instance
        self.prize_matrix = self._init_prize_matrix()

    def _init_prize_matrix(self) -> PrizeMatrix:
        """
        Helper function that initializes prize matrix using self.instance.
        **Assumes that truck capacity and quantities are one decimal place**.
        **Assumes that edge costs are symmetric**.

        Raises:
            RuntimeError: tree graph not initialized.

        Returns:
            PrizeMatrix: initialized prize matrix associated with instance.
        """

        if self.instance.tree is None:
            raise RuntimeError("tree graph not initialized.")

        def _scale_quantity(quantity: float, multiplier: int = DynamicTSPSolver.MULTIPLIER) -> int:
            scaled = quantity * multiplier
            rounded = round(scaled)
            if not math.isclose(
                scaled, rounded, rel_tol=DynamicTSPSolver.REL_TOL, abs_tol=DynamicTSPSolver.ABS_TOL
            ):
                raise ValueError(f"Expected at most one decimal place, got {quantity}")
            return rounded

        q_max = _scale_quantity(self.instance.truck_capacity_tons, DynamicTSPSolver.MULTIPLIER)
        n_nodes = len(self.instance.tree.nodes())

        # initialize node-index mappings
        nodes_by_idx = list(self.instance.tree.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes_by_idx)}
        farmer_idx_to_node_idx = {
            farmer_idx: node_to_idx[farmer.id]
            for farmer_idx, farmer in enumerate(self.instance.farmers)
        }
        parent_idx_by_child_idx = [
            -1
            if node == self.instance.mill.id
            else node_to_idx[self.instance.child_to_parent[node]]
            for node in nodes_by_idx
        ]
        quantity_by_farmer_idx = [
            _scale_quantity(farmer.quantity, DynamicTSPSolver.MULTIPLIER)
            for farmer in self.instance.farmers
        ]

        # compute other tree parameters
        ordering = [node_to_idx[node] for node in reversed(self.instance.tree_order)]
        root_node_idx = node_to_idx[self.instance.mill.id]

        # populate costs matrix with edge costs
        edge_costs = np.zeros((n_nodes, n_nodes))
        for child_idx, child_node in enumerate(nodes_by_idx):
            if child_node == self.instance.mill.id:
                continue

            parent_idx = parent_idx_by_child_idx[child_idx]
            parent_node = nodes_by_idx[parent_idx]

            # we assume that edge costs are symmetric
            edge_cost = (
                self.instance.tree[child_node][parent_node]["weight"]
                * self.instance.truck_cost_per_m
            )
            edge_costs[child_idx, parent_idx] = edge_cost
            edge_costs[parent_idx, child_idx] = edge_cost

        return PrizeMatrix(
            instance=self.instance,
            q_max=q_max,
            n_nodes=n_nodes,
            farmer_idx_to_node_idx=farmer_idx_to_node_idx,
            ordering=ordering,
            quantity_by_farmer_idx=quantity_by_farmer_idx,
            parent_idx_by_child_idx=parent_idx_by_child_idx,
            root_node_idx=root_node_idx,
            edge_costs=edge_costs,
        )

    def solve(self, prizes_dict: dict[str, float]) -> tuple[list[Route], list[float]]:
        """Solve the simplified TSP problem.

        **Assumes that farmers have unique IDs**.

        Args:
            prizes_dict (dict[str, float]): dictionary that maps farmer IDs
                to their associated prize. `prizes_dict[id]` is the prize associated with
                the farmer whose ID is `id`.

        Raises:
            ValueError: if the farmer IDs in `prizes_dict` do not match those in the instance.
            ValueError: if no farmers are selected in a route.
            ValueError: if the objective does not match alternative computation (consistency check).
            RuntimeError: if pricing returned no feasible routes.

        Returns:
            list[Route]: list of TOP_N routes with highest objectives.
            list[float]: list of TOP_N objectives corresponding to best routes.
        """

        # check that farmer IDs are unique
        farmer_ids = [farmer.id for farmer in self.instance.farmers]

        if len(farmer_ids) != len(set(farmer_ids)):
            raise ValueError("Farmer IDs must be unique.")

        # check that prize IDs exactly match farmer IDs
        expected_ids = {farmer.id for farmer in self.instance.farmers}
        provided_ids = set(prizes_dict)

        if provided_ids != expected_ids:
            missing = expected_ids - provided_ids
            extra = provided_ids - expected_ids
            raise ValueError(f"Prize IDs mismatch. Missing: {missing}, Extra: {extra}")

        # convert prizes_dict into a list and solve
        prizes_by_farmer_idx = [prizes_dict[farmer.id] for farmer in self.instance.farmers]
        total_prizes, farmer_idx_sets = self.prize_matrix.solve(prizes_by_farmer_idx, -np.inf)

        # extract selected farmers using selected index sets
        selected_farmers_sets = [
            [self.instance.farmers[farmer_idx].id for farmer_idx in farmer_idx_set]
            for farmer_idx_set in farmer_idx_sets
        ]

        # extract feasible candidate routes only
        feasible_candidates = []
        for total_prize, selected_farmers in zip(total_prizes, selected_farmers_sets, strict=True):
            if not selected_farmers:
                raise ValueError("No farmers selected.")

            route = self.instance.calculate_tree_path(selected_farmers)

            # skip infeasible DP candidates, but keep searching through the larger pool
            if not route.is_feasible:
                continue

            objective = total_prize - self.instance.truck_fixed_cost

            # objective consistency check
            alt_objective = (
                sum(prizes_dict[farmer_id] for farmer_id in selected_farmers) - route.cost
            )

            if abs(objective - alt_objective) > DynamicTSPSolver.OBJ_TOL:
                raise ValueError(f"Objective mismatch: {objective} != {alt_objective}")

            feasible_candidates.append((objective, route))

        # keep top N feasible routes in decreasing order
        feasible_candidates.sort(key=lambda x: x[0], reverse=True)
        feasible_candidates = feasible_candidates[: DynamicTSPSolver.TOP_N]

        final_routes = [route for _, route in feasible_candidates]
        final_objs = [objective for objective, _ in feasible_candidates]

        if len(final_routes) == 0:
            raise RuntimeError(
                "Pricing returned no feasible routes after DP candidate generation. "
                "This can underconstrain the primal and produce inflated profit."
            )

        return final_routes, final_objs
