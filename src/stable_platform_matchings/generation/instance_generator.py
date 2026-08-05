import json
import pickle
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import osmnx as ox
import pandas as pd
from names_generator import generate_name
from pyproj import Transformer
from scipy.interpolate import interp1d
from scipy.special import gammaln, logsumexp
from scipy.stats import gaussian_kde

from ..domain.entities import Farmer, Intermediary, Mill
from ..domain.instance import Instance
from ..graphs.road_graphs import RoadGraph

INDO_CRS = "EPSG:23867"  # Indonesia projected CRS
LL_CRS = "EPSG:4326"  # WGS84 Lat/Lon
MIN_QUANTITY, MAX_CAPACITY = 0.1, 9  # feel free to change this
RES = 250  # spatial grid resolution (meters)
MAX_DIST = 63000  # max. sampling distance

FALLBACK_SIGMA = 5000
FALLBACK_ALPHA = 0.3

DEFAULT_MILL = {
    "mill_id": "MILL",
    "location": [-0.682643, 102.501522],
}

DEFAULT_KDE_BANDWIDTH_FACTOR = 0.2
KDE_DIST_BUFFER = 10000
N_GAMMA_STEPS = 2000
TOL = 1e-10
CYCLE_LENGTH = 14


class InstanceGenerator:
    """
    Generates synthetic platform instances using real pickup data from Indonesia.

    Attributes:
        self.xy_to_ll (Transformer): transforms points in projected CRS to lat/lon format.
            to use this, call `self.xy_to_ll.transform()`. note that points are returned in
            Cartesian format, that is (lon/lat).
        self.ll_to_xy (Transformer): transforms points in lat/lon format to projected CRS.
            note that points are inputted in Cartesian format, that is (lon/lat).

        self.farmers_full_df (pd.DataFrame): a DataFrame containing historical farmer pickup data.
        self.farmers_14_df (pd.DataFrame): a DataFrame containing 14 days of historical farmer
            pickup data.
        self.intermediaries_df (pd.DataFrame): a DataFrame containing intermediary information.

        self.graph (RoadGraph): a RoadGraph of the platform road network.
        self.bbox_m (npt.NDArray[np.float64]): a bounding box of the platform road network.
        self.grid_coords (npt.NDArray[np.float64]): a stacked array of grid coordinates.

        self.intermediary_spatial_kde (gaussian_kde):
        self.farmer_spatial_kde (gaussian_kde):
        self.gamma_lookups (dict[str, interp1d]):

        self.alpha (float):
        self.sigmas (dict[str, float]):

        self.p_spatial ()

        self.hist_quantities
        self.hist_n_farmers
        self.hist_inactive_rates
        self.hist_offsets

        self.intermediaries (dict[str, dict[str, Any]], optional): a dict mapping intermediary
            IDs to dicts containing their location and type.
        self.mills = [DEFAULT_MILL]

        self.pickups_df (pd.DataFrame): a Data
        self.cycle_length = CYCLE_LENGTH
    """

    def __init__(
        self,
        farmers_full_csv_path: Path,
        farmers_14_csv_path: Path,
        intermediaries_csv_path: Path,
        graph_pkl_path: Path,
        alpha_json_path: Path,
        sigmas_json_path: Path,
    ) -> None:
        """
        Initializes the instance generator.

        Args:
            farmers_full_csv_path (Path): Path to a .csv file containing historical farmer pickup 
                data with columns `farmer_id`, `intermediary_id`, `quantity`, `date`, 
                `intermediary_lat`, `intermediary_lon`, `intermediary_x`, `intermediary_y`, 
                `farmer_lat`, `farmer_lon`, `farmer_x`, `farmer_y`, `distance`. 
                note that x,y columns are in local projected CRS.
            farmers_14_csv_path (Path): Path to a .csv file containing 14 days of historical farmer
                pickup data in the same format as `farmers_full_csv_path`.
            intermediaries_csv_path (Path): Path to a .csv file containing intermediary information
                with columns `intermediary_id`, `intermediary_lat`, `intermediary_lon`,
                `intermediary_x`, `intermediary_y`. note that x,y columns are in local CRS.
            graph_pkl_path (Path): Path to a .pickle file storing a networkx graph for the
                covered platform region, in this case an area in Indonesia.
            alpha_json_path (Path): Path to a .json file containing a single calibrated parameter
                value for weighting 2D global farmer density versus 1D distance density.
            sigmas_json_path (Path): Path to a .json file containing calibrated sigma parameter
                values for controlling sequential clustering during farmer network generation. each
                intermediary ID (str) is associated with a sigma value (float).
        """

        # CRS transformers
        self.xy_to_ll = Transformer.from_crs(INDO_CRS, LL_CRS, always_xy=True)
        self.ll_to_xy = Transformer.from_crs(LL_CRS, INDO_CRS, always_xy=True)

        # load empirical data
        self.farmers_full_df = pd.read_csv(farmers_full_csv_path)
        self.farmers_14_df = pd.read_csv(farmers_14_csv_path)
        self.intermediaries_df = pd.read_csv(intermediaries_csv_path)
        self.graph, self.bbox_m = self._init_graph_and_bbox(graph_pkl_path)

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
        self.gamma_lookups = self._init_gamma_lookups()

        # sigma and alpha values for clustering intensity (precomputed)
        with open(alpha_json_path, mode="r", encoding="utf-8") as f:
            self.alpha = json.load(f)
        with open(sigmas_json_path, mode="r", encoding="utf-8") as f:
            self.sigmas = json.load(f)

        # precompute farmer spatial priors on grid
        p_spatial = self.farmer_spatial_kde.evaluate(self.grid_coords)
        self.p_spatial = p_spatial / (p_spatial.sum() + TOL)

        # cache historical statistics
        self.hist_quantities = (
            self.farmers_full_df.groupby("intermediary_id")["quantity"].apply(list).to_dict()
        )

        counts_df = (
            self.farmers_full_df.groupby(["intermediary_id", "date"])
            .size()
            .reset_index(name="count")
        )
        self.hist_n_farmers = counts_df.groupby("intermediary_id")["count"].apply(list).to_dict()

        empirical_n_intermediaries = self.farmers_14_df["intermediary_id"].nunique()
        daily_active_counts = self.farmers_14_df.groupby("date")["intermediary_id"].nunique()
        self.hist_inactive_rates = 1.0 - daily_active_counts / empirical_n_intermediaries

        self.hist_offsets = self._init_hist_offsets()

        # internal instance "state"
        self.intermediaries = {}
        self.mills = []

        self.calendar_df = pd.DataFrame()
        self.cycle_length = CYCLE_LENGTH

    def gen_intermediaries(self, n_intermediaries: int, seed: int) -> None:
        """
        Initializes a set of platform intermediaries using KDE sampling.

        Args:
            n_intermediaries (int): the number of intermediaries to generate.
            seed (int): random seed for reproducibility

        Returns:
            None
        """

        # create random seeds for reproducibility, one per intermediary
        seed_seq = np.random.SeedSequence(seed)
        intermediary_seeds = seed_seq.spawn(n_intermediaries)
        rngs = [
            np.random.default_rng(intermediary_seed) for intermediary_seed in intermediary_seeds
        ]

        # initialize possible intermediary representative "types"
        representative_types = list(self.gamma_lookups.keys())

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

            intermediary_type = rng.choice(representative_types)

            # rejection sampling for locations within bounding box
            while True:
                sample = self.intermediary_spatial_kde.resample(1, seed=rng).flatten()
                if (
                    self.bbox_m[0] <= sample[0] <= self.bbox_m[2]
                    and self.bbox_m[1] <= sample[1] <= self.bbox_m[3]
                ):
                    intermediary_xy = sample
                    break

            # transform from local CRS to lat/lon
            lon, lat = self.xy_to_ll.transform(intermediary_xy[0], intermediary_xy[1])

            intermediaries[intermediary_id] = {
                "xy": intermediary_xy,
                "ll": (lat, lon),
                "type": intermediary_type,
            }

        self.intermediaries = intermediaries

    def gen_farmer_xys(
        self,
        intermediary_xy: tuple[float, float],
        intermediary_type: str,
        n_farmers: int,
        rng: np.random.Generator,
        sigma: float,
    ) -> list[npt.NDArray[np.float64]]:
        """
        Given an intermediary, generates multiple farmer coordinates according to
        that intermediary's type and location.

        Args:
            intermediary_xy (tuple[float, float]): the intermediary's location in local CRS.
            intermediary_type (str): the intermediary's representative type ID.
            n_farmers (int): number of farmers to be generated.
            rng (np.random.Generator): RNG for reproducibility.
            sigma (float): clustering weight.

        Returns:
            list[npt.NDArray[np.float64]]: a list of farmer coordinates corresponding to
                points in the spatial grid.
        """

        grid_coords_t = self.grid_coords.T

        # get base log probability density
        log_p_base = self._compute_log_base_grid_prior(
            intermediary_xy=intermediary_xy,
            intermediary_type=intermediary_type,
            alpha=self.alpha,
        )

        sigma_sq_2 = 2 * (sigma**2)

        # generate each farmer location and accumulate clustering influence
        locs = []
        accumulated_exponential_kernels = np.zeros(len(grid_coords_t))
        for k in range(n_farmers):
            if k == 0:
                log_p_cond = log_p_base  # start with base probability
            else:
                # bayesian update: clustering influence (add in log space)
                log_local_factor = np.log(accumulated_exponential_kernels + TOL) - np.log(k)
                log_p_cond = log_p_base + log_local_factor
                log_p_cond -= logsumexp(log_p_cond)

            p_sampling = np.exp(log_p_cond)

            # sample farmer location
            idx = rng.choice(len(p_sampling), p=p_sampling / p_sampling.sum())
            sampled_xy = self.grid_coords[:, idx]

            # add to locations and update kernel for next farmer in sequence
            locs.append(sampled_xy)
            new_dist_sq = np.sum((grid_coords_t - sampled_xy) ** 2, axis=1)
            accumulated_exponential_kernels += np.exp(-new_dist_sq / sigma_sq_2)

        return locs

    def gen_base_schedule(
        self, cycle_length: int, farmer_rngs: dict[str, np.random.Generator], scale: float
    ) -> pd.DataFrame:
        """
        Generates a base pickup schedule of length `cycle_length` days.

        Args:
            cycle_length (int): how many days in one pickup cycle, e.g. 14 days?
            farmer_rngs (dict[str, np.random.Generator]): a dict that maps farmer IDs to their
                respective random number generators for reproducibility.
            scale (float): a scaling factor to be applied to farmer counts.

        Returns:
            pd.DataFrame: a DataFrame containing the base schedule of farmer pickups.
        """

        synth_farmers = []
        for intermediary_id in self.intermediaries:
            # get farmer's RNG
            rng = farmer_rngs[intermediary_id]

            # get intermediary data
            intermediary_data = self.intermediaries[intermediary_id]
            intermediary_type = intermediary_data["type"]
            intermediary_xy = intermediary_data["xy"]
            intermediary_sigma = self.sigmas.get(intermediary_type, FALLBACK_SIGMA)

            for cycle_phase in range(cycle_length):
                # sample some empirical count, scale it,
                # and perform stochastic rounding if not integer
                empirical_count = rng.choice(self.hist_n_farmers[intermediary_type])
                raw_count = empirical_count * scale
                n_farmers = int(np.floor(raw_count) + (rng.random() < raw_count % 1))

                # sample farmer locations
                farmer_xys = self.gen_farmer_xys(
                    intermediary_xy=intermediary_xy,
                    intermediary_type=intermediary_type,
                    n_farmers=n_farmers,
                    rng=rng,
                    sigma=intermediary_sigma,
                )

                # sample farmer quantities from empirical historical distribution
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

    def gen_calendar(
        self,
        seed: int,
        n_cycles: int,
        scale: float = 1.0,
        cycle_length: int = CYCLE_LENGTH,
        buffer_cycles: int = 1,
    ):
        """
        Generates a calendar of farmer pickups from a base schedule.

        Args:
            seed (int): seed for reproducibility.
            n_cycles (int): number of cycles to generate.
            scale (float, optional): a scaling factor to be applied to farmer counts.
                defaults to 1.0.
            cycle_length (int, optional): the length of one cycle, in days.
                defaults to CYCLE_LENGTH.
            buffer_cycles (int, optional): used to avoid edge effects from applying random offsets
                to the start and end of the calendar. defaults to 1.
        """
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
        schedule_df = self.gen_base_schedule(cycle_length, farmer_rngs, scale)

        # generate pickup calendar
        calendar = []

        for pickup in schedule_df.itertuples():
            for cycle_index in range(
                -buffer_cycles,
                n_cycles + buffer_cycles,
            ):
                cycle_phase = cast(int, pickup.cycle_phase)
                nominal_day = cycle_phase + cycle_index * cycle_length

                calendar.append(
                    {
                        "farmer_id": pickup.farmer_id,
                        "farmer_x": pickup.farmer_x,
                        "farmer_y": pickup.farmer_y,
                        "farmer_lon": pickup.farmer_lon,
                        "farmer_lat": pickup.farmer_lat,
                        "cycle_phase": pickup.cycle_phase,
                        "quantity": pickup.quantity,
                        "intermediary_id": pickup.intermediary_id,
                        "nominal_day": nominal_day,
                    }
                )

        calendar_df = pd.DataFrame(calendar)

        # sample random day offsets to perturb pickups
        calendar_df["offset"] = calendar_rng.choice(self.hist_offsets, size=len(calendar_df))
        calendar_df["day"] = calendar_df["nominal_day"] + calendar_df["offset"]

        # discard start and end to avoid edge effects
        interior_mask = calendar_df["day"].between(0, n_cycles * cycle_length - 1)
        calendar_df = calendar_df[interior_mask]

        # randomly apply intermediary inactivity
        calendar_df = self._apply_random_inactivity(calendar_df, inactivity_rng)

        # scale quantities
        calendar_df["scaled_quantity"] = calendar_df.groupby(["day", "intermediary_id"])[
            "quantity"
        ].transform(
            lambda quantities: self._cap_quantities(
                quantities.to_numpy(),
                maximum_capacity=MAX_CAPACITY,
                minimum_quantity=MIN_QUANTITY,
                precision=1,
            )
        )

        # filter
        calendar_df = calendar_df[
            ["scaled_quantity", "day", "farmer_lat", "farmer_lon", "intermediary_id", "farmer_id"]
        ]

        # set attributes
        self.calendar_df = calendar_df
        self.cycle_length = cycle_length

    def gen_instance(
        self,
        instance_id: str,
        day: int,
        n_hist_sets: int = 1,
    ) -> Instance:
        """
        Generates a single platform instance from a day from a synthetic calendar.

        Args:
            instance_id (str): ID associated with this instance.
            day (int): the day number to choose in the synthetic calendar.
            n_hist_sets (int, optional): the number of historical pickups to consider.
                defaults to 1.

        Raises:
            ValueError: n_hist_sets must be at least 2.
            ValueError: calendar_df is empty.
            ValueError: n_hist_sets * cycle_length must be less than or equal to day
            ValueError: day is outside the generated range.

        Returns:
            Instance: a synthetic platform Instance.
        """

        if n_hist_sets < 1:
            raise ValueError("n_hist_sets must be at least 2.")

        if self.calendar_df.empty:
            raise ValueError("calendar_df is empty. Run gen_pickups() first.")

        if day - n_hist_sets * self.cycle_length < 0:
            raise ValueError("n_hist_sets * cycle_length must be less than or equal to day")

        min_day = int(self.calendar_df["day"].min())
        max_day = int(self.calendar_df["day"].max())

        if not min_day <= day <= max_day:
            raise ValueError(f"day={day} is outside the generated range [{min_day}, {max_day}].")

        # get farmers participating in that day's matching
        day_df = self.calendar_df.loc[self.calendar_df["day"] == day].copy()

        farmers = []
        farmer_ids = set()

        for pickup in day_df.itertuples(index=False):
            farmer_id = cast(str, pickup.farmer_id)
            farmer_location = (cast(float, pickup.farmer_lat), cast(float, pickup.farmer_lon))
            farmer_quantity = cast(float, pickup.scaled_quantity)
            intermediary_id = cast(str, pickup.intermediary_id)

            farmers.append(
                Farmer(
                    id=farmer_id,
                    location=farmer_location,
                    quantity=farmer_quantity,
                    intermediary_id=intermediary_id,
                )
            )

            farmer_ids.add(farmer_id)

        # get historical observations in `cycle_length`-day increments
        hist_days = {
            day - cycle_number * self.cycle_length for cycle_number in range(1, n_hist_sets + 1)
        }
        hist_df = self.calendar_df.loc[self.calendar_df["day"].isin(hist_days)].copy()
        hist_days_sorted = sorted(hist_days)

        # get each intermediary and their historical pickup sets
        intermediaries = []
        for intermediary_id, intermediary_data in self.intermediaries.items():
            hist_sets = []
            for historical_day in hist_days_sorted:
                mask = (hist_df["day"] == historical_day) & (
                    hist_df["intermediary_id"] == intermediary_id
                )

                farmer_ids_hist = cast(pd.Series, hist_df.loc[mask, "farmer_id"])

                hist_set = farmer_ids_hist.astype(str).tolist()

                # route IDs must belong to the farmer universe.
                hist_set = [farmer_id for farmer_id in hist_set if farmer_id in farmer_ids]

                # remove accidental duplicates while preserving order.
                hist_set = list(dict.fromkeys(hist_set))

                # retain empty routes because they represent an observed
                # historical schedule with no available farmers.
                hist_sets.append(hist_set)

            intermediary_location = (intermediary_data["ll"][0], intermediary_data["ll"][1])

            intermediaries.append(
                Intermediary(
                    id=intermediary_id,
                    capacity=MAX_CAPACITY,
                    location=intermediary_location,
                    hist_sets=hist_sets,
                )
            )

        # get default mill
        mill = Mill(id=DEFAULT_MILL["mill_id"], location=DEFAULT_MILL["location"])

        # construct instance, attach graph, and return
        instance = Instance(
            instance_id=instance_id, farmers=farmers, intermediaries=intermediaries, mill=mill
        )

        instance.set_graph(self.graph)

        return instance

    def _init_graph_and_bbox(
        self, graph_pkl_path: Path
    ) -> tuple[RoadGraph, npt.NDArray[np.float64]]:
        """
        Initializes graph and bounding box data for the instance.

        Args:
            graph_pkl_path (Path): Path to a .pickle file storing a networkx graph for the
                covered platform region.

        Returns:
            tuple[RoadGraph, npt.NDArray[np.float64]]: a tuple whose first argument is the RoadGraph
                for the road network and second argument is the corresponding bounding box.
        """
        with open(graph_pkl_path, "rb") as f:
            G = pickle.load(f)

        G_proj = ox.project_graph(G, to_crs=INDO_CRS)
        nodes_proj, _ = ox.graph_to_gdfs(G_proj)

        return RoadGraph(G), nodes_proj.total_bounds

    def _init_intermediary_kde(self) -> gaussian_kde:
        """Initializes Gaussian KDE for global intermediary spatial density."""
        coords = self.intermediaries_df.drop_duplicates(["intermediary_id"])[
            ["intermediary_x", "intermediary_y"]
        ].T
        return gaussian_kde(coords, bw_method=DEFAULT_KDE_BANDWIDTH_FACTOR)

    def _init_farmer_kde(self) -> gaussian_kde:
        """Initializes Gaussian KDE for global farmer spatial density."""
        coords = self.farmers_full_df.drop_duplicates(["farmer_x", "farmer_y"])[
            ["farmer_x", "farmer_y"]
        ].T
        return gaussian_kde(coords, bw_method=DEFAULT_KDE_BANDWIDTH_FACTOR)

    def _init_gamma_lookups(self) -> dict[str, interp1d]:
        """
        Initializes a Gamma KDE lookup table, using 1D interpolation for efficiency.

        Returns:
            dict[str, interp1d]: a dict mapping intermediary IDs to a interpolation object
                approximating a smoothed distance distribution.
        """
        # get historical distances by int
        intermediary_to_dists = (
            self.farmers_full_df.drop_duplicates(["intermediary_id", "farmer_x", "farmer_y"])
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

    def _init_hist_offsets(self) -> pd.Series:
        """
        Initializes a DataFrame of historical date offsets from a periodic cycle. In particular,
        this function determines how many days a pickup "deviated" from its usual day.
        For example, a pickup that usually takes place on the first Monday but is later found
        to happen on the first Wednesday would have an offset of +2.

        Returns:
            pd.DataFrame: a DataFrame of historical date offsets.
        """

        def _compute_offsets(group: pd.DataFrame, period: int = 14) -> pd.DataFrame:
            """Computes perturbations from a periodic cycle."""
            group = group.sort_values("date").copy()
            dates = pd.to_datetime(group["date"]).dt.normalize()
            t = (dates - dates.iloc[0]).dt.days.to_numpy()
            cycle_index = np.round(t / period).astype(int)
            phase = int(np.round(np.median(t - period * cycle_index)))
            group["delta"] = t - (phase + period * cycle_index)
            return group

        # copy to avoid mutating attribute df
        pickups_df = self.farmers_full_df.copy()

        pickups_df["date"] = pd.to_datetime(pickups_df["date"])
        pickups_df = pickups_df.sort_values(["intermediary_id", "farmer_x", "farmer_y", "date"])

        # compute gaps between pickups
        pickup_gaps = (
            pickups_df.groupby(["intermediary_id", "farmer_x", "farmer_y"]).date.diff().dt.days
        )

        # filter out excessively long pickup gaps (these represent breaks)
        pickup_gaps = pickup_gaps[(pickup_gaps > 0) & (pickup_gaps < 90)]

        # compute historical offsets
        hist_offsets = (
            pickups_df.groupby(["intermediary_id", "farmer_x", "farmer_y"], group_keys=False).apply(
                _compute_offsets
            )
        )["delta"]

        # return truncated offsets (to avoid collisions with competing 7-day offsets)
        return hist_offsets[hist_offsets.abs() < 7]

    def _compute_log_base_grid_prior(
        self, intermediary_xy: tuple[float, float], intermediary_type: str, alpha: float
    ) -> npt.NDArray[np.float64]:
        """
        Constructs a base farmer location probability density for an intermediary,
        taking into account global farmer density as well as that intermediary's "distance"
        preferences. Returns log density for numerical stability. Uses Euclidean norm for distance.

        Args:
            intermediary_xy (tuple[float, float]): the intermediary's location in local CRS.
            intermediary_type (str): the intermediary's representative type ID.
            alpha (float): weighting parameter.

        Returns:
            npt.NDArray[np.float64]: a vertically stacked array of log probabilities corresponding
                to each grid point in `grid_coords`.
        """

        dists = np.linalg.norm(self.grid_coords.T - intermediary_xy, axis=1)
        gamma_lookup = self.gamma_lookups[intermediary_type]

        # get raw distance distribution from gamma lookup table
        p_dist_raw = gamma_lookup(dists)

        # convert radial density to grid-cell density and apply a radial correction factor
        min_radius = RES / 2
        radius_correction = np.maximum(dists, min_radius)
        p_dist_grid = p_dist_raw / radius_correction
        p_dist_grid = p_dist_grid / (p_dist_grid.sum() + TOL)  # normalize

        # get farmer spatial prior over same grid support
        p_farmer_grid = self.p_spatial
        p_farmer_grid = p_farmer_grid / (p_farmer_grid.sum() + TOL)  # normalize

        # combine log probabilities, giving the global farmer density an `alpha` weight.
        log_p_base = np.log(p_dist_grid + TOL) + alpha * np.log(p_farmer_grid + TOL)
        log_p_base -= logsumexp(log_p_base)  # normalize in log space

        return log_p_base

    @staticmethod
    def _cap_quantities(
        quantities: npt.NDArray[np.float64],
        maximum_capacity: float = MAX_CAPACITY,
        minimum_quantity: float = MIN_QUANTITY,
        precision: int = 1,
    ) -> npt.NDArray[np.float16]:
        """
        Scales a vector of quantities such that their sum remains under a capacity limit.
        Operates with fixed precision (decimal places).

        Args:
            quantities (npt.NDArray[np.float64]): an array of farmer quantities.
            maximum_capacity (float, optional): maximum truck capacity. defaults to MAX_CAPACITY.
            minimum_quantity (float, optional): _description_. defaults to MIN_CAPACITY.
            precision (int, optional): decimal place precision. defaults to 1.

        Raises:
            ValueError: quantities must be one-dimensional.
            ValueError: quantities contain non-finite values.
            ValueError: Cannot fit farmers within capacity.
            RuntimeError: Unable to repair rounded quantities within capacity.

        Returns:
            npt.NDArray[np.float64]: _description_
        """

        # validation checks
        if quantities.ndim != 1:
            raise ValueError("quantities must be one-dimensional.")

        if not np.isfinite(quantities).all():
            raise ValueError("quantities contain non-finite values.")

        # enforce minimum
        adjusted = np.maximum(quantities, minimum_quantity)
        adjusted = np.round(adjusted, precision)

        total = float(adjusted.sum())

        # already feasible: preserve the adjusted quantities.
        if total <= maximum_capacity:
            return adjusted

        # feasibility is impossible if minimum quantities alone exceed capacity.
        minimum_total = len(adjusted) * minimum_quantity
        if minimum_total > maximum_capacity:
            raise ValueError(
                f"Cannot fit {len(adjusted)} farmers within capacity "
                f"{maximum_capacity}: minimum total is {minimum_total}."
            )

        # scale only the quantity above the mandatory 0.1 minimum.
        excess = adjusted - minimum_quantity
        available_excess = maximum_capacity - minimum_total

        if excess.sum() > 0:
            scaled = minimum_quantity + excess * available_excess / excess.sum()
        else:
            scaled = adjusted.copy()

        scaled = np.round(scaled, precision)
        scaled = np.maximum(scaled, minimum_quantity)

        # repair rounding overflow.
        increment = 10 ** (-precision)
        while scaled.sum() > maximum_capacity + TOL:
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

        Args:
            calendar_df (pd.DataFrame): a DataFrame containing a calendar of pickups.
            rng (np.random.Generator): RNG for reproducibility.

        Returns:
            pd.DataFrame: a calendar with random inactivity.
        """

        intermediary_ids = np.asarray(list(self.intermediaries), dtype=object)
        n_intermediaries = len(intermediary_ids)

        if calendar_df.empty or n_intermediaries == 0:
            return calendar_df.copy()

        # get historical inactivity rates
        inactivity_rates = np.asarray(self.hist_inactive_rates, dtype=float)
        inactivity_rates = inactivity_rates[np.isfinite(inactivity_rates)]

        if len(inactivity_rates) == 0:
            return calendar_df.copy()

        inactive_pairs = []
        for day in sorted(calendar_df["day"].unique()):
            # sample some inactivity rate
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

            # choose some random intermediaries to be set inactive
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
