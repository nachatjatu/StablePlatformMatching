from .instance import Instance
from .route import Route


class Matching:
    """
    Collection of routes covering all required farmers exactly once.

    Attributes:
        instance (Instance): the current platform instance.
        routes (list[Route]): list of intermediary routes in the matching.
        cost (float): sum of route costs in the matching.
    """

    def __init__(self, instance: Instance, routes: list[Route]) -> None:
        """
        Initializes and validates a matching.

        Args:
            instance (Instance): the current platform instance.
            routes (list[Route]): list of intermediary routes in the matching.
        """

        self.instance = instance
        self.routes = routes
        self._verify_routes()
        self.cost = sum(route.cost for route in self.routes)

    def _verify_routes(self) -> None:
        """
        Validates routes in the matching.

        Raises:
            ValueError: route is empty.
            ValueError: duplicate farmers across routes.
            ValueError: not all farmers covered.
        """

        farmers_in_routes = []
        for route in self.routes:
            if not route.farmers:
                raise ValueError("Route cannot be empty.")

            if not route.is_feasible:
                raise ValueError(
                    f"Route exceeds truck capacity: "
                    f"{route.total_quantity} > {self.instance.truck_capacity_tons}"
                )

            for farmer in route.farmers:
                farmers_in_routes.append(farmer.id)

        instance_farmer_ids_set = set(farmer.id for farmer in self.instance.farmers)
        route_farmer_ids_set = set(farmers_in_routes)

        if len(farmers_in_routes) != len(route_farmer_ids_set):
            raise ValueError("Duplicate farmers found across routes.")

        if route_farmer_ids_set != instance_farmer_ids_set:
            missing = instance_farmer_ids_set - route_farmer_ids_set
            extra = route_farmer_ids_set - instance_farmer_ids_set
            raise ValueError(f"Coverage mismatch. Missing: {missing}, Extra: {extra}")
