import networkx as nx

from stable_platform_matchings.graphs.road_graphs import RoadGraph


def add_weighted_edge(
    graph: nx.Graph,
    node_1: int,
    node_2: int,
    weight: float,
    *,
    weight_paved: float = 0.0,
    weight_dirt: float = 0.0,
) -> None:
    graph.add_edge(
        node_1,
        node_2,
        weight=weight,
        weight_paved=weight_paved,
        weight_dirt=weight_dirt,
    )


def test_prune_graph_contracts_degree_two_node() -> None:
    graph = nx.Graph()

    add_weighted_edge(
        graph,
        1,
        2,
        weight=3.0,
        weight_paved=3.0,
    )
    add_weighted_edge(
        graph,
        2,
        3,
        weight=5.0,
        weight_dirt=5.0,
    )

    was_pruned = RoadGraph._prune_graph(
        graph,
        do_not_prune={1, 3},
    )

    assert 2 not in graph
    assert graph.has_edge(1, 3)

    edge_data = graph.get_edge_data(1, 3)
    assert edge_data is not None
    assert edge_data["weight"] == 8.0
    assert edge_data["weight_paved"] == 3.0
    assert edge_data["weight_dirt"] == 5.0

    assert was_pruned is True


def test_prune_graph_keeps_cheaper_existing_edge() -> None:
    graph = nx.Graph()

    add_weighted_edge(graph, 1, 2, weight=4.0)
    add_weighted_edge(graph, 2, 3, weight=5.0)
    add_weighted_edge(graph, 1, 3, weight=2.0)

    RoadGraph._prune_graph(
        graph,
        do_not_prune={1, 3},
    )

    edge_data = graph.get_edge_data(1, 3)
    assert edge_data is not None
    assert edge_data["weight"] == 2.0
