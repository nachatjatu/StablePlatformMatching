from dataclasses import dataclass, field

TRUCK_CAPACITY_TONS = 9.0


@dataclass
class Entity:
    """
    Base entity with an identifier and a geographical location.

    Attributes:
        id (str): unique entity identifier.
        location (tuple[float, float]): latitude and longitude of entity.
        dist_to_mill (float | None, optional): shortest path distance from entity to mill.
            defaults to None.
        dirt_to_mill (float | None, optional): shortest path distance from entity to mill
            using dirt road. defaults to None.
        paved_to_mill (float | None, optional): shortest path distance from entity to mill
            using paved road. defaults to None.
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
        quantity (float, optional): fruit quantity produced. defaults to `0.0`
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
    """

    capacity: float = TRUCK_CAPACITY_TONS
    hist_sets: list[frozenset[str]] = field(default_factory=list[frozenset[str]])


@dataclass
class Mill(Entity):
    """Subclass of Entity class representing a mill receiving fruit from routes."""

    pass
