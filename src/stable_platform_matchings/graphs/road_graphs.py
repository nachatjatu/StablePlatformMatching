from collections.abc import Hashable
from itertools import combinations
from typing import Any, TypeAlias, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
from matplotlib.lines import Line2D
from shapely.geometry import Point

GraphNode: TypeAlias = Hashable


class RoadGraph:
    DIRT_FACTOR = 4
    MAPPING_SURFACES = {
        None: "dirt",  # assume dirt by default
        "path": "dirt",
        "primary": "paved",
        "primary_link": "paved",
        "residential": "dirt",
        "secondary": "paved",
        "secondary_link": "paved",
        "service": "dirt",
        "tertiary": "paved",
        "track": "dirt",
        "trunk": "paved",
        "trunk_link": "paved",
        "unclassified": "dirt",
        "living_street": "dirt",
    }
    LOCAL_CRS = "EPSG:4326"

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.mapping_surfaces = RoadGraph.MAPPING_SURFACES
        self.graph = graph
        self.undirected_graph = self.get_undirected_graph()

    def get_undirected_graph(self) -> nx.Graph:
        """
        Returns an undirected version of self.graph with appropriate connections between
        connected components such that the entire graph is connected.

        Returns:
            nx.Graph: undirected version of self.graph.
        """
        G = self.graph.to_undirected()

        # 1. connect the graph
        #   a. get all connected components
        components = list(nx.connected_components(G))

        if not components:
            raise ValueError("The road graph contains no nodes.")

        print("Using corrected graph algorithm")

        main_component = max(components, key=len)
        main_connector = list(G.subgraph(main_component).nodes())[0]

        for component in components:
            if component == main_component:
                continue

            component_connector = list(G.subgraph(component).nodes())[0]

            G.add_edge(
                main_connector,
                component_connector,
                highway="trunk",
                length=0.0,
            )

        # 2. add additional weights for different surface types
        for _, _, _, edge_data in G.edges(keys=True, data=True):
            # a. get the highway attribute
            highway_attr = edge_data.get("highway", [])

            # b. ensure it"s a list so we can iterate consistently
            highways = highway_attr if isinstance(highway_attr, list) else [highway_attr]

            # c. see if any part of that road is paved
            is_paved = any(self.mapping_surfaces.get(highway) == "paved" for highway in highways)

            # d. Set edge attributes depending on surface type
            edge_length = edge_data["length"]
            edge_data["surface"] = "paved" if is_paved else "dirt"
            edge_data["weight"] = edge_length if is_paved else RoadGraph.DIRT_FACTOR * edge_length
            edge_data["weight_paved"] = edge_length if is_paved else 0.0
            edge_data["weight_dirt"] = 0.0 if is_paved else self.DIRT_FACTOR * edge_length

        # 3. remove multi-edges
        seen_pairs: set[frozenset[object]] = set()

        for u, v in list(G.edges()):
            pair = frozenset((u, v))

            if pair in seen_pairs:
                continue

            edge_bundle = G.get_edge_data(u, v)

            if edge_bundle is None or len(edge_bundle) <= 1:
                continue

            min_key = min(edge_bundle, key=lambda key: edge_bundle[key]["weight"])

            for k in list(edge_bundle):
                if k != min_key:
                    G.remove_edge(u, v, key=k)

            seen_pairs.add(pair)

        # 4. construct a new undirected graph from G
        new_G = nx.Graph()
        for node_1, node_2, edge_data in G.edges(data=True):
            keys = list(G[node_1][node_2].keys())
            assert len(keys) == 1  # sanity check that multi edges removed
            new_G.add_edge(node_1, node_2, **edge_data)

        assert nx.is_connected(new_G)  # sanity check that graph is connected

        return new_G

    def get_closest_points(self, all_points: list[tuple[float, float]]) -> list[GraphNode]:
        """
        Returns the closest nodes to a list of points in lat lon coordinates.
        **Assumes points are provided in the casual form (latitude, longitude)**.

        Args:
            all_points (list[tuple[float, float]]): a list containing points to be
                snapped to the graph.

        Returns:
            list[Point]: a list of the nearest shapely Points on the graph to
                the list of points given.
        """
        # convert all_points into shapely Points
        shapely_points = [Point(longitude, latitude) for latitude, longitude in all_points]

        # project to graph CRS
        graph_crs = self.graph.graph["crs"]
        gdf = gpd.GeoDataFrame(geometry=shapely_points, crs=RoadGraph.LOCAL_CRS)
        gdf_proj = ox.projection.project_gdf(gdf, to_crs=graph_crs)

        # extract coordinates
        xs = gdf_proj.geometry.x
        ys = gdf_proj.geometry.y

        # find nearest nodes and return as list
        nearest_nodes = ox.nearest_nodes(self.graph, xs, ys)
        return nearest_nodes.tolist()

    @staticmethod
    def _prune_graph(G: nx.Graph, do_not_prune: set[GraphNode]) -> bool:
        """
        Prunes a graph, excluding nodes in a given set.

        Args:
            G (nx.Graph): graph to be pruned.
            do_not_prune (set[GraphNode]): a set of graph nodes that must not be pruned.

        Returns:
            bool: True if the graph was pruned, False otherwise.
        """
        initial_len = len(G)

        # remove unprotected degree-1 nodes (dead ends)
        dead_ends = [
            node for node, degree in G.degree() if degree == 1 and node not in do_not_prune
        ]
        G.remove_nodes_from(dead_ends)

        # repeat the same procedure for nodes of degree 2, merging them
        while True:
            target = None
            # find an unprotected degree-2 node to prune
            for node, deg in G.degree():
                if deg == 2 and node not in do_not_prune:
                    target = node
                    break

            # exit if no removable nodes of degree 2 found
            if target is None:
                break

            neighbors = list(G.neighbors(target))
            assert len(neighbors) == 2  # confirm node is degree 2

            # get connecting edges from target to neighbors
            neighbor_1, neighbor_2 = neighbors
            edge_1, edge_2 = (target, neighbor_1), (target, neighbor_2)

            # sanity check edge existence
            assert edge_1 in G.edges() and edge_2 in G.edges()

            # combine edge attributes
            edge_1_data, edge_2_data = G.get_edge_data(*edge_1), G.get_edge_data(*edge_2)

            new_weight = edge_1_data["weight"] + edge_2_data["weight"]
            new_weight_paved = edge_1_data["weight_paved"] + edge_2_data["weight_paved"]
            new_weight_dirt = edge_1_data["weight_dirt"] + edge_2_data["weight_dirt"]

            # remove node
            G.remove_node(target)

            # update edge information (if necessary)
            existing_data = G.get_edge_data(neighbor_1, neighbor_2)

            if existing_data is None or new_weight < existing_data["weight"]:
                G.add_edge(
                    neighbor_1,
                    neighbor_2,
                    weight=new_weight,
                    weight_dirt=new_weight_dirt,
                    weight_paved=new_weight_paved,
                )

        final_len = len(G)
        return initial_len != final_len  # check if graph was pruned

    @staticmethod
    def iteratively_prune(G: nx.Graph, not_to_touch: set) -> None:
        while True:
            if not RoadGraph._prune_graph(G, not_to_touch):
                break

    def build_tree(
        self, all_ids: list[object], all_stops: list[object], root: str, plot=True
    ) -> tuple[nx.Graph, list]:
        """
        Calculates an approximation to the steiner tree, returns the tree and
        edges that connect the root to all stops.

        A
        """
        all_stops_set = set(all_stops)

        # copy and iteratively prune
        G_pruned = self.undirected_graph.copy()
        self.iteratively_prune(G_pruned, all_stops_set)  # prunes in-place

        # create Graph and add direct edges between stops
        complete_graph = nx.Graph()
        for stop_1, stop_2 in combinations(all_stops, 2):
            weight = nx.shortest_path_length(G_pruned, stop_1, stop_2, weight="weight")
            complete_graph.add_edge(stop_1, stop_2, weight=weight)

        # extract minimum spanning tree
        T_complete = nx.minimum_spanning_tree(complete_graph, weight="weight")

        # add shortest path edges between stops
        edges_to_add = set()
        for node_1, node_2 in T_complete.edges():
            path = nx.shortest_path(self.undirected_graph, node_1, node_2, weight="weight")
            # add edges to connect stops
            for s in range(len(path) - 1):
                edge_fwd, edge_rev = (path[s], path[s + 1]), (path[s + 1], path[s])
                if edge_fwd not in edges_to_add and edge_rev not in edges_to_add:
                    edges_to_add.add(edge_fwd)

        # extract minimum spanning tree
        T = self.undirected_graph.edge_subgraph(list(edges_to_add))
        T = nx.minimum_spanning_tree(T, weight="weight")

        # create a subgraph of the original graph containing only
        # the edges in T that are in between the stops
        edges_to_add_subgraph = set()
        for stop_1, stop_2 in combinations(all_stops, 2):
            path = nx.shortest_path(T, stop_1, stop_2, weight="weight")
            for s in range(len(path) - 1):
                edge_fwd, edge_rev = (path[s], path[s + 1]), (path[s + 1], path[s])
                if edge_fwd not in edges_to_add_subgraph and edge_rev not in edges_to_add_subgraph:
                    edges_to_add_subgraph.add(edge_fwd)

        self.iteratively_prune(T, all_stops_set)

        # sanity check
        for stop in all_stops:
            assert stop in T.nodes()

        # add a node per each id in all_ids and connect it to the stop corresponding to the id
        for i in range(len(all_ids)):
            T.add_node(all_ids[i])
            T.add_edge(all_ids[i], all_stops[i], weight=0)

        # calculate edges that connect the root to all stops
        root_edges = []
        for node_id in all_ids:
            path = nx.shortest_path(T, root, node_id, weight="weight")
            path_edges = list(zip(path[:-1], path[1:], strict=True)) if len(path) > 1 else []
            root_edges.append(path_edges)

        # create subgraph
        # edges_subset = []
        # for node_1, node_2 in edges_to_add_subgraph:
        #     edges_subset.extend(self.graph.edges([node_1, node_2], keys=True))
        # # get all multi-edges with their keys

        # subgraph = self.graph.edge_subgraph(edges_subset)

        edges_subset = []

        for u, v in edges_to_add_subgraph:
            forward_edges = self.graph.get_edge_data(u, v, default={})
            reverse_edges = self.graph.get_edge_data(v, u, default={})

            edges_subset.extend((u, v, key) for key in forward_edges)
            edges_subset.extend((v, u, key) for key in reverse_edges)

        subgraph = self.graph.edge_subgraph(edges_subset).copy()

        # set the surface attribute for each edge in the subgraph,
        # remember that this is a multi-graph
        for u, v, k in subgraph.edges(keys=True):
            subgraph[u][v][k]["surface"] = self.undirected_graph[u][v]["surface"]

        # plot (optional)
        if plot:
            self.plot_graph(subgraph)

        # assign to subgraph attribute
        self.subgraph = subgraph

        return T, root_edges

    def plot_graph(
        self,
        subgraph: nx.MultiDiGraph,
        figsize: tuple[int, int] = (6, 6),
        options: dict[str, Any] | None = None,
    ) -> None:
        """
        Plots the provided subgraph.

        Args:
            subgraph (nx.MultiDiGraph): the subgraph to be plotted
            options (dict[str, object] | None, optional): a dict containing
                plotting options such as figure size and save path. defaults to None.
        """

        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
            }
        )

        if options is None:
            options = {}
        figsize = options.get("figsize", (6, 6))  # default figsize
        save_path = options.get("save_path", None)  # default to not saving

        # set up plot and axis
        _, ax = plt.subplots(figsize=figsize)

        # plot the main graph
        ox.plot_graph(
            self.graph,
            ax=ax,
            node_color="lightgrey",
            edge_color="lightgrey",
            show=False,
            close=False,
        )

        # plot the edge subgraph with colors depending on the surface attribute
        # black for paved, red for unpaved
        edge_colors = [
            "#000000" if subgraph[u][v][key]["surface"] == "paved" else "#ff0000"
            for u, v, key in subgraph.edges(keys=True)
        ]

        # extract node colors based on the edges connected to them
        node_colors = {}
        for u, v, key in subgraph.edges(keys=True):
            edge_color = "#000000" if subgraph[u][v][key]["surface"] == "paved" else "#ff0000"
            if u not in node_colors or node_colors[u] == "#ff0000":
                node_colors[u] = edge_color
            if v not in node_colors or node_colors[v] == "#ff0000":
                node_colors[v] = edge_color

        # apply node colors to the plot
        ox.plot_graph(
            subgraph,
            ax=ax,
            edge_color=edge_colors,  # color edges based on surface attribute
            node_size=2.5,
            edge_linewidth=2.5,
            node_color=[node_colors[node] for node in subgraph.nodes()],
            show=False,
            close=False,
        )

        # add legend for edge colors
        paved_patch = Line2D([0], [0], color="#000000", lw=2, label="Paved")
        unpaved_patch = Line2D([0], [0], color="#ff0000", lw=2, label="Unpaved")
        plt.legend(handles=[paved_patch, unpaved_patch], loc="upper right")

        if options:
            if "mill_node" in options and "farmer_nodes" in options:
                mill_node = options["mill_node"]
                # draw a circle at the mill_node
                mill_node_coords = subgraph.nodes[mill_node]["x"], subgraph.nodes[mill_node]["y"]
                ax.scatter(*mill_node_coords, c="blue", s=100, label="Mill Node", zorder=3)
                farmer_nodes = cast(list, options["farmer_nodes"])
                sizes = cast(list, options.get("residual", [1] * len(farmer_nodes)))
                # draw circles for farmer nodes
                for farmer_node, size in zip(farmer_nodes, sizes, strict=True):
                    farmer_node_coords = (
                        subgraph.nodes[farmer_node]["x"],
                        subgraph.nodes[farmer_node]["y"],
                    )
                    ax.scatter(
                        *farmer_node_coords,
                        c="green",
                        s=size * 100,  # scale size for better visibility
                        label="Farmer Node",
                        zorder=3,
                    )

        # tighten the layout
        plt.tight_layout()

        # save to PDF if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()
