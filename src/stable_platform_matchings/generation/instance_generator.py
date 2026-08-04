import json
import pickle

from pathlib import Path
import numpy as np
import osmnx as ox
import pandas as pd
from names_generator import generate_name
from pyproj import Transformer
from scipy.interpolate import interp1d
from scipy.special import gammaln, logsumexp
from scipy.stats import gaussian_kde
from typing import cast

from ..graphs.road_graphs import RoadGraph

"""
DEFINE GLOBAL CONSTANTS
"""
INDO_CRS = "EPSG:23867"  # Indonesia projected CRS
LL_CRS = "EPSG:4326"  # WGS84 Lat/Lon
MIN_CAPACITY, MAX_CAPACITY = 2, 9  # feel free to change this
RES = 250  # spatial grid resolution (meters)
MAX_DIST = 63000  # max. sampling distance

FALLBACK_SIGMA = 5000
FALLBACK_ALPHA = 0.3

DEFAULT_MILL = {
    "mill_id": "MILL",
    "location": [-0.682643, 102.501522],
}  # default mill used in previous experiments
DEFAULT_KDE_BANDWIDTH_FACTOR = 0.2
KDE_DIST_BUFFER = 10000
N_GAMMA_STEPS = 2000
TOL = 1e-10
CYCLE_LENGTH = 14


"""
INSTANCEGENERATOR CLASS
"""


class InstanceGenerator:
    def __init__(
        self,
        farmers_df_path: str | Path = "data/farmers.csv",  # full dataset of farmer pickups
        farmers_14_df_path: str | Path = "data/farmers_14.csv",  # 14 day dataset of farmer pickups
        intermediaries_df_path: str | Path = "data/intermediaries.csv",  # full dataset of intermediaries
        graph_path: str | Path = "data/graph_0-14960_00_new.pickle",  # pickle file of regional road graph (osmnx)
        alpha_path: str | Path = "data/precomputed_alpha.json",
        sigmas_path: str | Path = "data/precomputed_sigmas.json",
    ) -> None:

        # CRS transformers
        self.xy_to_ll = Transformer.from_crs(INDO_CRS, LL_CRS, always_xy=True)
        self.ll_to_xy = Transformer.from_crs(LL_CRS, INDO_CRS, always_xy=True)

        # load empirical data
        self.farmers_df, self.farmers_14_df = (
            pd.read_csv(farmers_df_path),
            pd.read_csv(farmers_14_df_path),
        )
        self.intermediaries_df = pd.read_csv(intermediaries_df_path)
        self.G, self.G_proj, self.bbox_m = self._init_graph(graph_path)

        # create spacial grid
        x_ax, y_ax = (
            np.arange(self.bbox_m[0], self.bbox_m[2], RES),
            np.arange(self.bbox_m[1], self.bbox_m[3], RES),
        )
        gx, gy = np.meshgrid(x_ax, y_ax, indexing="ij")
        self.grid_coords = np.vstack([gx.ravel(), gy.ravel()])

        # initialize KDEs
        self.intermediary_spatial_kde = self._init_intermediary_kde()
        self.farmer_spatial_kde = self._init_farmer_kde()
        self.gamma_lookups = self._init_gamma_kdes()

        # sigma and alpha values for clustering intensity (precomputed)
        with open(sigmas_path, mode="r", encoding="utf-8") as f:
            self.sigmas = json.load(f)
        with open(alpha_path, mode="r", encoding="utf-8") as f:
            self.alpha = json.load(f)

        # precompute farmer spatial priors on grid
        p_spatial = self.farmer_spatial_kde.evaluate(self.grid_coords)
        self.p_spatial = p_spatial / (p_spatial.sum() + TOL)

        # cache historical statistics
        self.hist_quantities = (
            self.farmers_df.groupby("intermediary_id")["quantity"].apply(list).to_dict()
        )

        counts_df = (
            self.farmers_df.groupby(["intermediary_id", "date"]).size().reset_index(name="count")
        )
        self.hist_n_farmers = counts_df.groupby("intermediary_id")["count"].apply(list).to_dict()

        empirical_n_intermediaries = self.farmers_14_df["intermediary_id"].nunique()

        daily_active_counts = self.farmers_14_df.groupby("date")["intermediary_id"].nunique()

        self.hist_inactive_rates = 1.0 - daily_active_counts / empirical_n_intermediaries

        self.hist_offsets = self._init_hist_offsets()

        # internal instance "state"
        self.intermediaries = {}
        self.mills = [DEFAULT_MILL]

        self.pickups_df = pd.DataFrame()
        self.cycle_length = CYCLE_LENGTH

    """
    --------------
    INITIALIZATION
    --------------
    """

    def _init_graph(self, graph_path):
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        G_proj = ox.project_graph(G, to_crs=INDO_CRS)
        nodes_proj, _ = ox.graph_to_gdfs(G_proj)
        return RoadGraph(G), G_proj, nodes_proj.total_bounds

    def _init_intermediary_kde(self):
        coords = self.intermediaries_df.drop_duplicates(["intermediary_id"])[
            ["intermediary_x", "intermediary_y"]
        ].T
        return gaussian_kde(coords, bw_method=DEFAULT_KDE_BANDWIDTH_FACTOR)

    def _init_farmer_kde(self):
        coords = self.farmers_df.drop_duplicates(["farmer_x", "farmer_y"])[
            ["farmer_x", "farmer_y"]
        ].T
        return gaussian_kde(coords, bw_method=DEFAULT_KDE_BANDWIDTH_FACTOR)

    def _init_gamma_kdes(self):
        # get historical distances by int
        intermediary_to_dists = (
            self.farmers_df.drop_duplicates(["intermediary_id", "farmer_x", "farmer_y"])  # type: ignore
            .groupby("intermediary_id")["distance"]
            .apply(np.array)
            .to_dict()
        )

        lookups = {}
        x_eval = np.linspace(0, MAX_DIST + KDE_DIST_BUFFER, N_GAMMA_STEPS)

        # for each int, fit a smoothed distance distribution
        for intermediary_id, dists in intermediary_to_dists.items():
            n = len(dists)
            h = 0.1 * np.mean(dists) + TOL
            shape = dists / h
            # evaluate Gamma log-PDF for each x
            pdf_values = np.zeros_like(x_eval)
            for i in range(n):
                s, scale = shape[i], h
                with np.errstate(divide="ignore", invalid="ignore"):
                    # Gamma log-PDF: (s-1)*log(x) - x/scale - (log(gamma(s)) + s*log(scale))
                    log_pdf = (
                        (s - 1) * np.log(x_eval + TOL)
                        - (x_eval / scale)
                        - (gammaln(s + TOL) + s * np.log(scale))
                    )
                pdf_values += np.exp(log_pdf)
            pdf_values /= n
            # interpolate for efficiency
            lookups[intermediary_id] = interp1d(
                x_eval, pdf_values, fill_value=0.0, bounds_error=False
            )
        return lookups

    def _init_hist_offsets(self):
        def compute_perturbations(group, period=14):
            group = group.sort_values("date").copy()
            dates = group["date"].values.astype("datetime64[D]")
            t = (dates - dates[0]).astype(int)
            cycle_index = np.round(t / period).astype(int)
            phase = int(np.round(np.median(t - period * cycle_index)))
            group["delta"] = t - (phase + period * cycle_index)
            return group

        pickups_df = self.farmers_df.copy()
        pickups_df["date"] = pd.to_datetime(pickups_df["date"])
        pickups_df = pickups_df.sort_values(["intermediary_id", "farmer_x", "farmer_y", "date"])

        pickup_gaps = (
            pickups_df.groupby(["intermediary_id", "farmer_x", "farmer_y"])["date"].diff().dt.days
        )

        pickup_gaps = pickup_gaps[(pickup_gaps > 0) & (pickup_gaps < 90)]

        deltas = (
            pickups_df.groupby(["intermediary_id", "farmer_x", "farmer_y"], group_keys=False).apply(
                compute_perturbations
            )
        )["delta"]

        deltas_trunc = deltas[deltas.abs() < 7]

        return deltas_trunc

    """
    ----------
    GENERATION
    ----------
    """

    def gen_intermediaries(self, n_intermediaries, seed):
        # create random seeds for reproducibility, one per intermediary
        seed_seq = np.random.SeedSequence(seed)
        intermediary_seeds = seed_seq.spawn(n_intermediaries)
        rngs = [
            np.random.default_rng(intermediary_seed) for intermediary_seed in intermediary_seeds
        ]

        # initialize possible int types
        types = list(self.gamma_lookups.keys())

        intermediaries = {}
        names = set()
        for i in range(n_intermediaries):
            rng = rngs[i]

            # generate unique name (no duplicates)
            while True:
                intermediary_id = generate_name(seed=int(rng.integers(0, 2**32 - 1)))
                if intermediary_id not in names:
                    names.add(intermediary_id)
                    break

            intermediary_type = rng.choice(types)

            # rejection sampling within bounding box
            while True:
                sample = self.intermediary_spatial_kde.resample(1, seed=rng).flatten()
                if (
                    self.bbox_m[0] <= sample[0] <= self.bbox_m[2]
                    and self.bbox_m[1] <= sample[1] <= self.bbox_m[3]
                ):
                    intermediary_xy = sample
                    break

            lon, lat = self.xy_to_ll.transform(intermediary_xy[0], intermediary_xy[1])
            intermediaries[intermediary_id] = {
                "xy": intermediary_xy,
                "ll": (lat, lon),
                "type": intermediary_type,
            }

        self.intermediaries = intermediaries

    def _compute_log_base_grid_prior(self, intermediary_xy, intermediary_type, alpha=0.25):
        grid_points = self.grid_coords.T
        dists = np.linalg.norm(grid_points - intermediary_xy, axis=1)

        dist_lookup = self.gamma_lookups[intermediary_type]
        p_dist_raw = dist_lookup(dists)

        MIN_RADIUS = RES / 2
        radius_correction = np.maximum(dists, MIN_RADIUS)

        # Convert radial density to grid-cell density
        p_dist_grid = p_dist_raw / radius_correction
        p_dist_grid = p_dist_grid / (p_dist_grid.sum() + TOL)

        # Farmer spatial prior over same grid support
        p_farmer_grid = self.p_spatial
        p_farmer_grid = p_farmer_grid / (p_farmer_grid.sum() + TOL)

        log_p_base = np.log(p_dist_grid + TOL) + alpha * np.log(p_farmer_grid + TOL)
        log_p_base -= logsumexp(log_p_base)

        return log_p_base

    def gen_farmers(self, intermediary_xy, intermediary_type, n_farmers, rng, sigma=500):
        # precompute distance from int to each grid point
        grid_points = self.grid_coords.T
        log_p_base = self._compute_log_base_grid_prior(
            intermediary_xy=intermediary_xy,
            intermediary_type=intermediary_type,
            alpha=self.alpha,
        )

        locs = []
        sigma_sq_2 = 2 * (sigma**2)
        acc_exp_kernels = np.zeros(len(grid_points))

        for k in range(n_farmers):
            if k == 0:
                log_p_cond = log_p_base
            else:
                # bayesian update: clustering influence (add in log space)
                log_local_factor = np.log(acc_exp_kernels + TOL) - np.log(k)
                log_p_cond = log_p_base + log_local_factor
                log_p_cond -= logsumexp(log_p_cond)

            p_sampling = np.exp(log_p_cond)

            # # numerical stability fallback
            # if np.isnan(p_sampling).any() or p_sampling.sum() == 0:
            #     p_sampling = self.p_spatial

            # sample farmer locations
            idx = rng.choice(len(p_sampling), p=p_sampling / p_sampling.sum())
            sampled_xy = self.grid_coords[:, idx]
            locs.append(sampled_xy)

            # update kernel for next farmer in sequence
            new_dist_sq = np.sum((grid_points - sampled_xy) ** 2, axis=1)
            acc_exp_kernels += np.exp(-new_dist_sq / sigma_sq_2)

        return np.array(locs)

    def gen_baseline_schedule(self, cycle_length, farmer_rngs, scale):
        intermediary_ids = list(self.intermediaries)
        synth_farmers = []

        # generate 14 days of pickups (baseline schedule)
        for intermediary_id in intermediary_ids:
            rng = farmer_rngs[intermediary_id]
            intermediary_data = self.intermediaries[intermediary_id]
            intermediary_type, intermediary_xy = intermediary_data["type"], intermediary_data["xy"]
            intermediary_sigma = self.sigmas.get(intermediary_type, FALLBACK_SIGMA)

            for cycle_phase in range(cycle_length):
                emp_count = rng.choice(self.hist_n_farmers[intermediary_type])
                raw_count = emp_count * scale
                n_farmers = int(np.floor(raw_count) + (rng.random() < raw_count % 1))

                farmer_xys = self.gen_farmers(
                    intermediary_xy=intermediary_xy,
                    intermediary_type=intermediary_type,
                    n_farmers=n_farmers,
                    rng=rng,
                    sigma=intermediary_sigma,
                )

                farmer_quantities = np.asarray(
                    rng.choice(
                        self.hist_quantities[intermediary_type], size=n_farmers, replace=True
                    ),
                    dtype=float,
                )

                for farmer_index, (farmer_xy, farmer_quantity) in enumerate(
                    zip(farmer_xys, farmer_quantities, strict=True)
                ):
                    farmer_id = f"{intermediary_id}_d{cycle_phase}_f{farmer_index}"
                    farmer_lon, farmer_lat = self.xy_to_ll.transform(farmer_xy[0], farmer_xy[1])

                    synth_farmers.append(
                        {
                            "farmer_id": farmer_id,
                            "farmer_x": farmer_xy[0],
                            "farmer_y": farmer_xy[1],
                            "farmer_lon": farmer_lon,
                            "farmer_lat": farmer_lat,
                            "cycle_phase": cycle_phase,
                            "quantity": float(farmer_quantity),
                            "intermediary_id": intermediary_id,
                        }
                    )

        return pd.DataFrame(synth_farmers)

    def gen_pickups(self, seed, scale=1.0, n_cycles=1, cycle_length=CYCLE_LENGTH, buffer_cycles=1):
        intermediary_ids = list(self.intermediaries)

        # initialize random seeding
        master_seed = np.random.SeedSequence(seed)
        farmer_seed, calendar_seed, inactivity_seed = master_seed.spawn(3)
        farmer_child_seeds = farmer_seed.spawn(len(intermediary_ids))
        farmer_rngs = {
            intermediary_id: np.random.default_rng(child_seed)
            for intermediary_id, child_seed in zip(
                intermediary_ids, farmer_child_seeds, strict=True
            )
        }
        calendar_rng = np.random.default_rng(calendar_seed)
        inactivity_rng = np.random.default_rng(inactivity_seed)

        # generate baseline schedule
        synth_farmers_df = self.gen_baseline_schedule(cycle_length, farmer_rngs, scale)

        # generate pickup calendar
        pickups = []

        for row in synth_farmers_df.itertuples():
            for cycle_index in range(
                -buffer_cycles,
                n_cycles + buffer_cycles,
            ):
                cycle_phase = cast(int, row.cycle_phase)
                nominal_day = cycle_phase + cycle_index * cycle_length

                pickups.append(
                    {
                        "farmer_id": row.farmer_id,
                        "farmer_x": row.farmer_x,
                        "farmer_y": row.farmer_y,
                        "farmer_lon": row.farmer_lon,
                        "farmer_lat": row.farmer_lat,
                        "cycle_phase": row.cycle_phase,
                        "quantity": row.quantity,
                        "intermediary_id": row.intermediary_id,
                        "nominal_day": nominal_day,
                    }
                )

        calendar_df = pd.DataFrame(pickups)

        calendar_df["schedule_offset"] = calendar_rng.choice(
            self.hist_offsets, size=len(calendar_df)
        )

        calendar_df["day"] = calendar_df["nominal_day"] + calendar_df["schedule_offset"]

        pickups_df = calendar_df[calendar_df["day"].between(0, n_cycles * cycle_length - 1)].copy()

        pickups_df = self._apply_random_inactivity(
            pickups_df,
            inactivity_rng,
        )

        pickups_df["scaled_quantity"] = pickups_df.groupby(["day", "intermediary_id"])[
            "quantity"
        ].transform(
            lambda x: self._cap_quantities(
                x.to_numpy(), capacity=MAX_CAPACITY, minimum_quantity=0.1, precision=1
            )
        )

        self.pickups_df = pickups_df
        self.cycle_length = cycle_length

    def gen_instance(
        self,
        instance_id,
        day,
        n_hist_sets=1,
        dev_mode="required_only",
    ):

        valid_dev_modes = {
            "required_only",
            "required_and_hist",
            "all",
        }

        if dev_mode not in valid_dev_modes:
            raise ValueError(
                f"Unknown dev_mode={dev_mode!r}. Expected one of {sorted(valid_dev_modes)}."
            )

        if n_hist_sets < 1:
            raise ValueError("n_hist_sets must be positive.")

        if self.pickups_df.empty:
            raise ValueError("pickups_df is empty. Run gen_pickups() first.")

        if day - n_hist_sets * self.cycle_length < 0:
            raise ValueError("n_hist_sets * cycle_length must be less than or equal to day")

        min_day = int(self.pickups_df["day"].min())
        max_day = int(self.pickups_df["day"].max())

        if not min_day <= day <= max_day:
            raise ValueError(f"day={day} is outside the generated range [{min_day}, {max_day}].")

        # ---------------------------------------------------------
        # Focal-day farmers that must be collected
        # ---------------------------------------------------------
        day_df = self.pickups_df.loc[self.pickups_df["day"] == day].copy()

        farmers = []
        farmer_ids = set()

        for row in day_df.itertuples(index=False):
            farmer_id = str(row.farmer_id)

            farmers.append(
                {
                    "farmer_id": farmer_id,
                    "quantity": float(row.scaled_quantity),
                    "location": (
                        float(row.farmer_lat),
                        float(row.farmer_lon),
                    ),
                    "intermediary_id": str(row.intermediary_id),
                }
            )

            farmer_ids.add(farmer_id)

        # ---------------------------------------------------------
        # Historical observations
        # ---------------------------------------------------------
        historical_days = {
            day - cycle_number * self.cycle_length for cycle_number in range(1, n_hist_sets + 1)
        }

        hist_df = self.pickups_df.loc[self.pickups_df["day"].isin(historical_days)].copy()

        # ---------------------------------------------------------
        # intermediaries and historical routes
        # ---------------------------------------------------------
        intermediaries = []

        # Keep the historical schedules in chronological order.
        historical_days_sorted = sorted(historical_days)

        for intermediary_id, intermediary_data in self.intermediaries.items():
            routes = []

            for historical_day in historical_days_sorted:
                mask = (
                    (hist_df["day"] == historical_day)
                    & (hist_df["intermediary_id"] == intermediary_id)
                )

                farmer_ids_hist = cast(
                    pd.Series,
                    hist_df.loc[mask, "farmer_id"],
                )

                route = farmer_ids_hist.astype(str).tolist()

                # Route IDs must belong to the farmer universe.
                route = [farmer_id for farmer_id in route if farmer_id in farmer_ids]

                # Remove accidental duplicates while preserving order.
                route = list(dict.fromkeys(route))

                # Retain empty routes because they represent an observed
                # historical schedule with no available farmers.
                routes.append(route)

            intermediaries.append(
                {
                    "intermediary_id": str(intermediary_id),
                    "capacity": MAX_CAPACITY,
                    "location": (
                        float(intermediary_data["ll"][0]),
                        float(intermediary_data["ll"][1]),
                    ),
                    "routes": routes,
                }
            )

        return {
            "instance_id": instance_id,
            "farmers": farmers,
            "intermediaries": intermediaries,
            "mills": [DEFAULT_MILL],
        }

    @staticmethod
    def _cap_quantities(
        quantities,
        capacity=9.0,
        minimum_quantity=0.1,
        precision=1,
    ):
        quantities = np.asarray(quantities, dtype=float)

        if quantities.ndim != 1:
            raise ValueError("quantities must be one-dimensional.")

        if len(quantities) == 0:
            return quantities.copy()

        if not np.isfinite(quantities).all():
            raise ValueError("quantities contain non-finite values.")

        # Enforce the farmer-level minimum.
        adjusted = np.maximum(quantities, minimum_quantity)
        adjusted = np.round(adjusted, precision)

        total = float(adjusted.sum())

        # Already feasible: preserve the adjusted quantities.
        if total <= capacity:
            return adjusted

        # Feasibility is impossible if minimum quantities alone exceed capacity.
        minimum_total = len(adjusted) * minimum_quantity
        if minimum_total > capacity:
            raise ValueError(
                f"Cannot fit {len(adjusted)} farmers within capacity "
                f"{capacity}: minimum total is {minimum_total}."
            )

        # Scale only the quantity above the mandatory 0.1 minimum.
        excess = adjusted - minimum_quantity
        available_excess = capacity - minimum_total

        if excess.sum() > 0:
            scaled = minimum_quantity + excess * available_excess / excess.sum()
        else:
            scaled = adjusted.copy()

        scaled = np.round(scaled, precision)
        scaled = np.maximum(scaled, minimum_quantity)

        # Repair rounding overflow.
        increment = 10 ** (-precision)

        while scaled.sum() > capacity + 1e-10:
            candidates = np.where(scaled >= minimum_quantity + increment)[0]

            if len(candidates) == 0:
                raise RuntimeError("Unable to repair rounded quantities within capacity.")

            idx = candidates[np.argmax(scaled[candidates])]
            scaled[idx] = np.round(
                scaled[idx] - increment,
                precision,
            )

        return scaled

    def _apply_random_inactivity(
        self,
        calendar_df: pd.DataFrame,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """Randomly deactivate intermediaries on each generated day.

        For each day, sample an empirical inactivity rate and randomly choose
        the corresponding number of intermediaries to deactivate. All pickups
        for an inactive day-int pair are removed.
        """
        if calendar_df.empty:
            return calendar_df.copy()

        intermediary_ids = np.asarray(list(self.intermediaries), dtype=object)
        n_intermediaries = len(intermediary_ids)

        if n_intermediaries == 0:
            return calendar_df.copy()

        inactivity_rates = np.asarray(
            self.hist_inactive_rates,
            dtype=float,
        )

        inactivity_rates = inactivity_rates[np.isfinite(inactivity_rates)]

        if len(inactivity_rates) == 0:
            return calendar_df.copy()

        inactive_pairs = []

        for day in sorted(calendar_df["day"].unique()):
            inactivity_rate = float(rng.choice(inactivity_rates))

            n_inactive = int(
                np.clip(
                    np.round(inactivity_rate * n_intermediaries),
                    0,
                    n_intermediaries,
                )
            )

            if n_inactive == 0:
                continue

            inactive_ids = rng.choice(
                intermediary_ids,
                size=n_inactive,
                replace=False,
            )

            inactive_pairs.extend((day, intermediary_id) for intermediary_id in inactive_ids)

        if not inactive_pairs:
            return calendar_df.copy()

        inactive_index = pd.MultiIndex.from_tuples(
            inactive_pairs,
            names=["day", "intermediary_id"],
        )

        pickup_index = pd.MultiIndex.from_frame(calendar_df[["day", "intermediary_id"]])

        return calendar_df.loc[~pickup_index.isin(inactive_index)].copy()
