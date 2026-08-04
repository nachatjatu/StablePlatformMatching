def test_major_modules_import() -> None:
    from stable_platform_matchings.domain.entities import Farmer
    from stable_platform_matchings.domain.instance import Instance
    from stable_platform_matchings.generation.instance_generator import (
        InstanceGenerator,
    )
    from stable_platform_matchings.graphs.road_graphs import RoadGraph
    from stable_platform_matchings.optimization.optimizer import Optimizer
    from stable_platform_matchings.reporting.containers import Solution

    assert Farmer is not None
    assert Instance is not None
    assert InstanceGenerator is not None
    assert RoadGraph is not None
    assert Optimizer is not None
    assert Solution is not None
