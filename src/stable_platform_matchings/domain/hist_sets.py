from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .instance import Instance

HistSetMethod = Literal["original", "union", "instance_farmers", "all"]

HIST_SET_METHODS: tuple[HistSetMethod, ...] = (
    "original",
    "union",
    "instance_farmers",
    "all",
)


def original_hist_sets(
    instance: Instance,
) -> dict[str, tuple[frozenset[str], ...]]:
    """
    Historical collection schedules exactly as recorded on the instance.

    An intermediary with no recorded schedules is given a single empty set, so that
    every intermediary has at least one set and averages over sets stay well defined.

    Args:
        instance (Instance): the platform instance.

    Returns:
        dict[str, tuple[frozenset[str], ...]]: maps intermediary ID to its schedules.
    """
    return {
        intermediary.id: (
            tuple(frozenset(hist_set) for hist_set in intermediary.hist_sets)
            or (frozenset(),)
        )
        for intermediary in instance.intermediaries
    }


def configure_hist_sets(
    instance: Instance,
    hist_set_method: HistSetMethod | str,
) -> dict[str, tuple[frozenset[str], ...]]:
    """
    Support of the nominal distribution P-hat_t for each intermediary.

    These are the sets the stability constraints are built against. The result depends
    only on the instance and the method -- never on payments, costs, ambiguity levels,
    or any solve output -- so callers that must fix inputs before an Optimizer exists
    (e.g. assigning per-intermediary ambiguity levels by status-quo size) can compute
    it up front rather than reaching into a constructed Optimizer.

    Args:
        instance (Instance): the platform instance.
        hist_set_method (HistSetMethod | str): `original` keeps the recorded schedules
            as an empirical distribution; `union` collapses them to their union;
            `instance_farmers` uses the farmers assigned to the intermediary in this
            instance; `all` unions the previous two.

    Raises:
        ValueError: `hist_set_method` invalid.

    Returns:
        dict[str, tuple[frozenset[str], ...]]: maps intermediary ID to its active sets.
    """
    originals = original_hist_sets(instance)

    if hist_set_method == "original":
        return dict(originals)

    if hist_set_method == "union":
        return {
            intermediary_id: (frozenset().union(*hist_sets),)
            for intermediary_id, hist_sets in originals.items()
        }

    if hist_set_method in ("instance_farmers", "all"):
        active_hist_sets = {}
        for intermediary in instance.intermediaries:
            farmers = frozenset(
                farmer.id
                for farmer in instance.farmers
                if farmer.intermediary_id == intermediary.id
            )
            if hist_set_method == "all":
                farmers = farmers.union(*originals[intermediary.id])

            active_hist_sets[intermediary.id] = (farmers,)

        return active_hist_sets

    raise ValueError(
        "allowed hist_set_methods: union, all, instance_farmers, or original."
    )


def status_quo_quantities(
    instance: Instance,
    active_hist_sets: dict[str, tuple[frozenset[str], ...]],
) -> dict[str, float]:
    """
    Expected status-quo quantity E_{P-hat_t}[u_t' q] for each intermediary.

    This is the first term of the deviation potential rho_t; adding the ambiguity
    level eps_t gives rho_t itself. It must be evaluated against the same sets the
    stability constraints use, because the methods differ by more than a constant:
    `original` averages over the recorded schedules while the others collapse them
    into one set, so mixing methods mislabels rho_t rather than merely rescaling it.

    Args:
        instance (Instance): the platform instance.
        active_hist_sets (dict[str, tuple[frozenset[str], ...]]): active sets, as
            returned by `configure_hist_sets`.

    Returns:
        dict[str, float]: maps intermediary ID to expected status-quo quantity.
    """
    quantities = {}
    for intermediary_id, hist_sets in active_hist_sets.items():
        # Accumulate in instance-farmer order, and over instance farmers only, so
        # that sets naming farmers outside this instance contribute nothing.
        hist_quantity = 0
        for hist_set in hist_sets:
            for farmer in instance.farmers:
                if farmer.id in hist_set:
                    hist_quantity += farmer.quantity

        quantities[intermediary_id] = hist_quantity / len(hist_sets)

    return quantities


def status_quo_quantities_for_method(
    instance: Instance,
    hist_set_method: HistSetMethod | str,
) -> dict[str, float]:
    """
    Expected status-quo quantities under `hist_set_method`, without an Optimizer.

    Args:
        instance (Instance): the platform instance.
        hist_set_method (HistSetMethod | str): see `configure_hist_sets`.

    Returns:
        dict[str, float]: maps intermediary ID to expected status-quo quantity.
    """
    return status_quo_quantities(
        instance,
        configure_hist_sets(instance, hist_set_method),
    )
