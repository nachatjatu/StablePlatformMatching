import gurobipy as gp


def require_solution(model: gp.Model, context: str) -> None:
    if model.SolCount == 0:
        raise RuntimeError(f"{context} produced no feasible solution; status={model.Status}.")
