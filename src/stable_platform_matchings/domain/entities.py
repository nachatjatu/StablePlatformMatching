from dataclasses import dataclass, field

TRUCK_CAPACITY_TONS = 9.0

@dataclass
class Entity:
    """
    Base class for participants on the platform, with ID and location.

    Attributes:
        id (str): unique entity identifier.
        location (tuple[float, float]): latitude and longitude of entity.
        dist_to_mill (float): shortest path distance from entity to mill.
        dirt_to_mill (float): shortest path distance from entity to mill using dirt road.
        paved_to_mill (float): shortest path distance from entity to mill using paved road.
    """

    id: str
    location: tuple[float, float]
    dist_to_mill: float = field(init=False)
    dirt_to_mill: float = field(init=False)
    paved_to_mill: float = field(init=False)


@dataclass
class Farmer(Entity):
    """
    Subclass of Entity class representing a farmer producing some fruit quantity.

    Attributes:
        quantity (float): fruit quantity produced.
        intermediary_id (str): ID of intermediary associated with the farmer.
    """

    quantity: float = 0.0
    intermediary_id: str = field(default_factory=str)


@dataclass
class Intermediary(Entity):
    """
    Subclass of Entity class representing an intermediary collecting fruit
    and transporting it to a mill with some capacity.

    Attributes:
        capacity (float): intermediary truck capacity.
        hist_sets (list[frozenset[str]]): list of the intermediary's historical sets, where
            each historical set is a set of IDs of farmers picked up from.
    """
    capacity: float = TRUCK_CAPACITY_TONS
    hist_sets: list[frozenset[str]] = field(default_factory=list[frozenset[str]])


@dataclass
class Mill(Entity):
    """Subclass of Entity class representing a mill receiving fruit from routes."""
    pass
