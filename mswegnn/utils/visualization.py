## Libraries
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from copy import copy
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, LogNorm
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib import ticker
import sys

from mswegnn.database.graph_creation import (
    graph_from_mesh,
    MultiscaleMesh,
    remove_ghost_cells,
)
from mswegnn.utils.miscellaneous import (
    add_null_time_start,
    WD_to_FAT,
    plot_line_with_deviation,
)
from mswegnn.utils.miscellaneous import (
    get_time_vector,
    get_mass_conservation_loss,
    get_rollout_loss,
    get_CSI,
    get_F1,
)
from mswegnn.utils.miscellaneous import get_velocity, get_Froude
from mswegnn.utils.dataset import (
    get_input_water,
    get_inflow_volume,
    to_temporal,
    separate_multiscale_node_features,
)
from mswegnn.training.loss import get_mean_error

# from mswegnn.training.train import rollout_test
from mswegnn.utils.scaling import get_none_scalers

WD_color = LinearSegmentedColormap.from_list("", ["white", "MediumBlue"])
V_color = LinearSegmentedColormap.from_list("", ["white", "darkviolet"])
diff_color = LinearSegmentedColormap.from_list("", ["#E66100", "white", "#5D3A9B"])
diff_color_positive = LinearSegmentedColormap.from_list("", ["white", "#5D3A9B"])
diff_color_negative = LinearSegmentedColormap.from_list("", ["#E66100", "white"])
FAT_color = LinearSegmentedColormap.from_list("", ["MediumBlue", "white"])


def get_coords(pos):
    """
    Returns array of dimensions (n_nodes, 2) containing x and y coordinates of each node
    ------
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    """
    if isinstance(pos, dict):
        coordinates = np.array([xy for xy in pos.values()])
    else:
        coordinates = pos
    return coordinates


def get_corners(pos):
    """
    Returns the coordinates of the corners of a grid
    ------
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    """
    BL = min(pos.values())  # bottom-left
    TR = max(pos.values())  # top-right
    BR = (BL[0], TR[1])  # bottom-right
    TL = (TR[0], BL[1])  # top-left

    return BL, TR, BR, TL


def plot_loss(train_losses, val_losses=None, scale="log"):
    """
    Plot losses after training
    ------
    *_losses: list
        training (and validation) losses during training
    name: str
        give a name to save the plot as and image
    scale: str
        options: "linear", "log", "symlog", "logit", ...
    """
    plt.plot(train_losses, "b-")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale(scale)

    if val_losses is not None:
        plt.plot(val_losses, "r-")
        plt.legend(["Training", "Validation"], loc="upper right")
    plt.title("Loss vs. No. of epochs")

    return None


def plot_faces(mesh, ax=None, face_value=None, **kwargs):
    """Plots the mesh with face values if specified"""
    ax = ax or plt.gca()

    node_position = 0
    patches = []
    for num_nodes in mesh.nodes_per_face:
        face_node = mesh.face_nodes[node_position : (node_position + num_nodes)]
        face_nodes_x = mesh.node_x[face_node]
        face_nodes_y = mesh.node_y[face_node]
        face = np.stack((face_nodes_x, face_nodes_y)).T
        node_position += num_nodes
        patches.append(mpl.patches.Polygon(face, closed=True))

    collection = PatchCollection(patches, edgecolor="k", linewidths=0.1, **kwargs)
    collection.set_array(face_value)
    ax.add_collection(collection)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(mesh.node_x.min(), mesh.node_x.max())
    ax.set_ylim(mesh.node_y.min(), mesh.node_y.max())

    return ax


def plot_mesh(mesh, ax=None, **kwargs):
    ax = ax or plt.gca()

    graph, pos = graph_from_mesh(mesh)
    nx.draw(graph, pos, width=1, node_size=2, node_color="r", edge_color="k", **kwargs)

    return ax


class BasePlotMap(object):
    """
    Base class for plotting a map defined by either a graph (graph) or an unstructured mesh (mesh)

    ------
    map_: np.array or torch.tensor (shape [N], [N, 1] or [N_x, N_y])
        represents a single feature for each point in the domain
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    graph: networkx.classes.graph.Graph
        networkx graph with nodes and edges
    mesh: Mesh
        unstructured mesh
    scaler: sklearn.preprocessing._data scaler
        scaler object used for normalizing the data
    colorbar: bool (default=True)
        if True, shows colorbar in plot
    """

    def __init__(
        self,
        map,
        pos=None,
        graph=None,
        mesh=None,
        scaler=None,
        edge_index=None,
        difference_plot=False,
        **kwargs,
    ):
        self.map = map
        self.scaler = scaler
        self.pos = pos
        self.graph = graph
        self.mesh = mesh
        self.edge_index = edge_index
        self.kwargs = {**kwargs}
        self.difference_plot = difference_plot

        self.map = self._check_device(self.map)
        self._check_map_type()

    def _check_map_dimension(self, map):
        """map must be of dimension [N] when plotting"""
        if len(map.shape) > 1:
            map = map.reshape(-1)
        return map

    def _scale_map(self, map):
        """Scales back map, given scaler"""
        if self.scaler is not None:
            if len(map.shape) == 1:
                map = map.reshape(-1, 1)
            map = self.scaler.inverse_transform(map)
        map = self._check_map_dimension(map)
        return map

    def _check_device(self, map):
        """Convert map to cpu"""
        if isinstance(map, torch.Tensor):
            if map.device.type != "cpu":
                map = map.to("cpu")
            map = map.numpy()
        return map

    def _check_map_type(self):
        if self.graph is None and self.mesh is None:
            raise AttributeError(
                "BasePlotMap must receive either a graph 'graph' or a Mesh 'mesh'"
            )

    def _get_vmin(self, map):
        if "vmin" not in self.kwargs:
            self.kwargs["vmin"] = map.min()

    def _get_vmax(self, map):
        if "vmax" not in self.kwargs:
            self.kwargs["vmax"] = map.max()

    def _create_axes(self, ax=None):
        if ax is None:
            ax = plt.gca()
        return ax

    def _get_cmap(self):
        if self.difference_plot:
            if self.kwargs["vmin"] >= 0:
                self.kwargs["vmin"] = 0
                self.kwargs["cmap"] = diff_color_positive
            elif self.kwargs["vmax"] <= 0:
                self.kwargs["vmax"] = 0
                self.kwargs["cmap"] = diff_color_negative
            else:
                self.kwargs["cmap"] = diff_color
        elif "cmap" not in self.kwargs:
            self.kwargs["cmap"] = plt.cm.plasma

    def _add_colorbar(self, ax=None, colorbar=True, logscale=False):
        self.kwargs["vmax"] = self._check_device(self.kwargs["vmax"])
        self.kwargs["vmin"] = self._check_device(self.kwargs["vmin"])
        if self.difference_plot:
            if self.kwargs["vmin"] >= 0:
                ticks_interval = np.linspace(0, self.kwargs["vmax"], 5, endpoint=True)
                norm = plt.Normalize(vmin=0, vmax=self.kwargs["vmax"])
            elif self.kwargs["vmax"] <= 0:
                ticks_interval = np.linspace(self.kwargs["vmin"], 0, 5, endpoint=True)
                norm = plt.Normalize(vmin=self.kwargs["vmin"], vmax=0)
            else:
                ticks_interval = np.linspace(
                    self.kwargs["vmin"], self.kwargs["vmax"], 5, endpoint=True
                )
                norm = TwoSlopeNorm(
                    vmin=self.kwargs["vmin"], vcenter=0, vmax=self.kwargs["vmax"]
                )
        elif logscale and self.kwargs["vmin"] >= 0:
            ticks_interval = np.logspace(-3, -1, 3, endpoint=True)
            norm = LogNorm(vmin=1e-3, vmax=self.kwargs["vmax"], clip=True)
        else:
            ticks_interval = np.linspace(
                self.kwargs["vmin"], self.kwargs["vmax"], 5, endpoint=True
            )
            norm = plt.Normalize(vmin=self.kwargs["vmin"], vmax=self.kwargs["vmax"])

        if colorbar:
            decimals = 2
            if logscale:
                plt.colorbar(
                    plt.cm.ScalarMappable(norm=norm, cmap=self.kwargs["cmap"]),
                    ticks=ticks_interval,
                    fraction=0.05,
                    shrink=0.9,
                    ax=ax,
                )
            else:
                plt.colorbar(
                    plt.cm.ScalarMappable(norm=norm, cmap=self.kwargs["cmap"]),
                    ticks=np.sign(ticks_interval)
                    * np.floor(np.abs(ticks_interval) * 10**decimals)
                    / 10**decimals,
                    fraction=0.05,
                    shrink=0.9,
                    ax=ax,
                )

        return norm

    def plot_map(self, ax=None, colorbar=True, logscale=False):
        self.map = self._scale_map(self.map)
        self._get_vmin(self.map)
        self._get_vmax(self.map)
        ax = self._create_axes(ax=ax)
        self._get_cmap()
        norm = self._add_colorbar(ax=ax, colorbar=colorbar, logscale=logscale)

        if self.graph is not None:
            """Plot map as graph"""
            nx.draw_networkx_nodes(
                self.graph,
                pos=self.pos,
                node_color=self.map,
                node_shape="s",
                node_size=20,
                ax=ax,
                **self.kwargs,
            )
        elif self.mesh is not None:
            """Plot map as mesh"""
            vmin = self.kwargs.pop("vmin")
            vmax = self.kwargs.pop("vmax")
            plot_faces(self.mesh, ax=ax, face_value=self.map, norm=norm, **self.kwargs)

        return ax

    def plot_edge_map(self, ax=None, colorbar=True):
        self.map = self._scale_map(self.map)
        self._get_vmin(self.map)
        self._get_vmax(self.map)
        ax = self._create_axes(ax=ax)
        self._get_cmap()
        norm = self._add_colorbar(ax=ax, colorbar=colorbar)
        self.kwargs["edge_vmin"] = self.kwargs.pop("vmin")
        self.kwargs["edge_vmax"] = self.kwargs.pop("vmax")
        self.kwargs["edge_cmap"] = self.kwargs.pop("cmap")
        edge_list = self.edge_index.T.numpy()

        if self.graph is None:
            raise NotImplementedError("This function only works with graphs as input")
        else:
            """Plot edges of a graph"""
            nx.draw_networkx_edges(
                self.graph,
                pos=self.pos,
                edgelist=edge_list,
                edge_color=self.map,
                ax=ax,
                **self.kwargs,
            )

        return ax


class TemporalPlotMap(BasePlotMap):
    """Plot class for maps with temporal attributes

    ------
    map: np.array-like (shape [N, T] or [N, 1])
        temporal matrix of the map to be plotted
    time_step: int
        time step at which to plot the map
    temporal_res: int, [minutes]
        temporal resolution of the temporal dataset
    """

    def __init__(self, map, temporal_res, time_start=0, **map_kwargs):
        super().__init__(map, **map_kwargs)
        self.temporal_res = temporal_res
        self.time_start = time_start
        self.total_time = self.map.shape[1]

    def _get_map_at_time_step(self, map):
        if self.total_time > 1:
            map = map[:, self.time_step]
        return map

    def _get_current_time_step(self):
        # Take function out -> pytest
        self.time_in_minutes = (
            self.time_start + 1 + self.time_step % self.total_time
        ) * self.temporal_res
        self.time_in_hours = int(self.time_in_minutes / 60)

    def plot_map(self, time_step, ax=None, colorbar=True, logscale=False):
        self.time_step = time_step
        self._get_current_time_step()
        map = self._get_map_at_time_step(self.map)

        map = self._scale_map(map)
        self._get_vmin(map)
        self._get_vmax(map)
        ax = self._create_axes(ax=ax)
        self._get_cmap()
        norm = self._add_colorbar(ax=ax, colorbar=colorbar, logscale=logscale)

        if self.graph is not None:
            """Plot map as graph"""
            nx.draw_networkx_nodes(
                self.graph,
                pos=self.pos,
                node_color=map,
                node_shape="s",
                node_size=20,
                ax=ax,
                **self.kwargs,
            )
        elif self.mesh is not None:
            """Plot map as mesh"""
            mesh_kwargs = copy(self.kwargs)
            vmin = mesh_kwargs.pop("vmin")
            vmax = mesh_kwargs.pop("vmax")
            plot_faces(self.mesh, ax=ax, face_value=map, norm=norm, **mesh_kwargs)

        return ax


class QuiverPlotMap(BasePlotMap):
    """Plot class for vector fields

    ------
    field_x, field_y: torch.tensor or numpy.array
        vector field in x and y direction
    """

    def __init__(self, field_x, field_y, temporal_res, time_start=0, **map_kwargs):
        super().__init__(map=None, **map_kwargs)
        assert field_x.shape == field_y.shape, "Input fields must have same dimension"
        self.temporal_res = temporal_res
        self.time_start = time_start
        self.total_time = field_x.shape[1]

        self.field_x = self._check_device(field_x)
        self.field_y = self._check_device(field_y)

    def _get_mask(self):
        """Creates mask for very small field values"""
        epsilon = 0.001  # threshold for neglecting small values
        self.mask = self.field_module >= epsilon

    def _get_field_at_time_step(self, field):
        if self.total_time > 1:
            field = field[:, self.time_step]
        return field

    def _get_current_time_step(self):
        self.time_in_minutes = (
            self.time_start + 1 + self.time_step % self.total_time
        ) * self.temporal_res
        self.time_in_hours = self.time_in_minutes / 60

    def quiver_plot(self, time_step, ax=None, colorbar=True):
        """Plot quiver map"""
        self.time_step = time_step
        self._get_current_time_step()

        field_x = self._get_field_at_time_step(self.field_x)
        field_y = self._get_field_at_time_step(self.field_y)
        self.field_module = np.sqrt(field_x**2 + field_y**2)

        self._get_mask()

        field_x = self._scale_map(field_x) / self.field_module
        field_y = self._scale_map(field_y) / self.field_module
        self._get_vmin(self.field_module)
        self._get_vmax(self.field_module)
        ax = self._create_axes(ax=ax)
        self._get_cmap()
        norm = self._add_colorbar(ax=ax, colorbar=colorbar)

        if self.pos is not None:
            coordinates = get_coords(self.pos)
        else:
            coordinates = self.mesh.face_xy
        X = coordinates[:, 0]
        Y = coordinates[:, 1]
        quiver_kwargs = {
            key: self.kwargs[key] for key in self.kwargs.keys() - {"vmin", "vmax"}
        }

        q = ax.quiver(
            X[self.mask],
            Y[self.mask],
            field_x[self.mask],
            field_y[self.mask],
            self.field_module[self.mask],
            scale_units="height",
            width=0.0035,
            scale=10,
            **quiver_kwargs,
        )

        ax.axis([-1, X.max() + 1, -1, X.max() + 1])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        return ax


def correct_plt_units(ax, pos, x_label=True, y_label=True):
    """Corrects the units of the plot from m to km if the distance is too large

    Args:
        ax (matplotlib.axes.Axes): axis to correct
        pos (torch.Tensor): positions of the nodes (shape: [num_nodes, 2])
        x_label (bool, optional): whether to add x label. Defaults to True.
        y_label (bool, optional): whether to add y label. Defaults to True.
    """
    exp_size = int(f"{pos.mean():.2e}"[-2:])
    distance_unit = "m" if exp_size < 3 else "km"
    if distance_unit == "km":
        m2km = lambda x, _: f"{x/1000:g}"
        ax.xaxis.set_major_formatter(m2km)
        ax.yaxis.set_major_formatter(m2km)

    if x_label:
        ax.set_xlabel(f"x distance [{distance_unit}]")
    if y_label:
        ax.set_ylabel(f"y distance [{distance_unit}]")

    return ax


class DEMPlotMap(BasePlotMap):
    """Plot digital elevation model(DEM)"""

    def __init__(self, map, **map_kwargs):
        super().__init__(map, **map_kwargs)
        self.kwargs["cmap"] = "terrain"

    def _add_axes_info(self, ax, title=True, x_label=True, y_label=True):
        if title:
            ax.set_title("DEM (m)")
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        correct_plt_units(ax, self.pos, x_label, y_label)

        ax.set_xticks(
            np.linspace(
                self.pos.min(0)[0].round(-3), self.pos.max(0)[0].round(-3), num=5
            )
        )
        ax.set_yticks(
            np.linspace(
                self.pos.min(0)[1].round(-3), self.pos.max(0)[1].round(-3), num=5
            )
        )

    def _add_breach_location(self, ax, breach_coordinates):
        """Plot breach location at the breach coordinates with a red cross
        ------
        ax: matplotlib.axes._subplots.AxesSubplot
            axes to plot the breach location
        breach_coordinates: list or torch.tensor
            list of tuples with the x and y coordinates of the breaches
        """
        for breach in breach_coordinates:
            ax.scatter(
                breach[0], breach[1], s=200, c="r", marker="x", zorder=3, linewidths=5
            )


def plot_rollout_diff_in_time_all(
    diff_rollout, temporal_res, type_loss="RMSE", time_start=0, ax=None
):
    """Plot average node error distribution across time for a given simulation"""
    ax = ax or plt.gca()

    V_unit = "$m^2$/s"

    # WD plot
    lns = plot_rollout_diff_in_time_var(
        diff_rollout,
        temporal_res,
        type_loss,
        dim=0,
        time_start=time_start,
        ax=ax,
        label="h",
        c="royalblue",
    )

    ax.set_ylabel(f"h {type_loss} [m]")
    ax.set_xlabel("Time [h]")
    ax.set_xlim(0)

    ax2 = ax.twinx()

    # V
    lin_V = plot_rollout_diff_in_time_var(
        diff_rollout,
        temporal_res,
        type_loss,
        dim=1,
        time_start=time_start,
        ax=ax2,
        label="|q|",
        c="purple",
    )
    lns = lns + lin_V
    ax2.set_ylabel(f'|q| {type_loss} ["$m^2$/s"]')

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)

    return ax, ax2


def plot_rollout_diff_in_time_var(
    diff_rollout,
    temporal_res,
    type_loss="RMSE",
    dim=0,
    time_start=0,
    ax=None,
    **plot_kwargs,
):
    """Plot average node error distribution for a variable across time for a given simulation
    Variable axis is identified by dim: 0 = WD, 1 = VX/V, 2 = VY
    """
    diff_rollout = diff_rollout[:, dim, :].to("cpu")

    ax = ax or plt.gca()

    time_stop = diff_rollout.shape[-1]
    time_vector = np.linspace(
        0, (time_start + time_stop) * temporal_res / 60, time_stop + time_start + 1
    )

    average_diff_t = get_mean_error(diff_rollout, type_loss).numpy()

    average_diff_t = add_null_time_start(time_start, average_diff_t)

    return ax.plot(time_vector, average_diff_t, marker=".", **plot_kwargs)


def plot_breach_distribution(dataset, ax=None, with_label=True, **plt_kwargs):
    """Plot the breach distribution of a dataset of simulations."""
    assert isinstance(dataset, list), "This function works for a list of simulations"
    assert (
        dataset[0].mesh.num_nodes == dataset[1].mesh.num_nodes
    ), "All simulations must have the same mesh"

    ax = ax or plt.gca()

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    remove_ghost_cells(copy(dataset[0].mesh.meshes[0])).plot_boundary(ax=ax, c="k")

    breach_coordinates = np.array(
        [[data.mesh.face_xy[node.item()] for node in data.node_BC] for data in dataset]
    ).squeeze()

    plt.scatter(
        breach_coordinates[:, 0],
        breach_coordinates[:, 1],
        s=80,
        marker="X",
        zorder=3,
        **plt_kwargs,
    )

    if with_label:
        for i, breach in enumerate(breach_coordinates):
            plt.annotate(i, (breach[0], breach[1]), ha="right", va="bottom")

    pos = dataset[0].mesh.face_xy
    correct_plt_units(ax, pos)

    return ax


class PlotRollout:
    """Explore predictions vs real simulations
    plots DEM, losses in time, water depth, discharges, and difference between the two at the last time step

    ------
    model: torch.nn.Module,     trained model
    dataset: torch_geometric.data.data.Data,    data created in create_dataset
    scalers: dict,    dictionary of scalers used for training
    type_loss: str,     type of loss used to compute summed differences
    **temporal_test_dataset_parameters: dict,    parameters for temporal dataset
    """

    def __init__(
        self,
        model,
        dataset,
        scalers=None,
        type_loss="RMSE",
        **temporal_test_dataset_parameters,
    ):
        super().__init__()
        self.time_start = temporal_test_dataset_parameters["time_start"]
        self.time_stop = temporal_test_dataset_parameters["time_stop"]
        self.temporal_res = dataset.temporal_res
        self.V_unit = "$m^2$/s"
        self.V_label = "Discharge"
        self.V_symbol = "q"
        self.model = model
        self.dataset = dataset
        self.type_loss = type_loss
        self.DEM = dataset.DEM
        self.water_threshold = 0
        self.num_scales = (
            dataset.mesh.num_meshes if isinstance(dataset.mesh, MultiscaleMesh) else 1
        )

        # convert to temporal dataset for predictions
        temporal_dataset = to_temporal(
            dataset, rollout_steps=-1, **temporal_test_dataset_parameters
        )[0]

        # scalers
        self.scalers = scalers if scalers is not None else get_none_scalers()

        # plotting info
        self.pos = dataset.mesh.face_xy
        mesh = dataset.mesh
        self.default_plot_kwargs = {"pos": self.pos, "mesh": mesh}
        self.default_temporal_plot_kwargs = self.default_plot_kwargs | {
            "time_start": self.time_start,
            "temporal_res": self.temporal_res,
        }

        self.breach_coordinates = [
            self.pos[node.item()] for node in self.dataset.node_BC
        ]

        # get rollouts
        self.predicted_rollout = rollout_test(model, temporal_dataset).detach()
        self.real_rollout = temporal_dataset.y.detach()
        self.diff_rollout = self.predicted_rollout - self.real_rollout
        self.input_water = get_input_water(temporal_dataset).unsqueeze(-1)

        # get maps
        self._get_maps(
            self.real_rollout,
            self.predicted_rollout,
            self.diff_rollout,
            self.input_water,
            self.DEM,
        )

        # time vector
        self.total_time_steps = self.real_rollout.shape[-1] + self.time_start
        self.time_vector = get_time_vector(self.total_time_steps, self.temporal_res)

    def mesh_scale_plot(self, scale):
        mesh = self.dataset.mesh
        assert isinstance(
            mesh, MultiscaleMesh
        ), "This function only works for multiscale meshes"
        self.default_plot_kwargs["mesh"] = mesh.meshes[scale]
        self.default_temporal_plot_kwargs["mesh"] = mesh.meshes[scale]

        predicted_rollout = separate_multiscale_node_features(
            self.predicted_rollout, self.dataset.node_ptr
        )[scale]
        real_rollout = separate_multiscale_node_features(
            self.real_rollout, self.dataset.node_ptr
        )[scale]
        diff_rollout = separate_multiscale_node_features(
            self.diff_rollout, self.dataset.node_ptr
        )[scale]
        input_water = separate_multiscale_node_features(
            self.input_water, self.dataset.node_ptr
        )[scale]
        DEM = separate_multiscale_node_features(self.DEM, self.dataset.node_ptr)[scale]

        self._get_maps(real_rollout, predicted_rollout, diff_rollout, input_water, DEM)

    def _get_maps(
        self, real_rollout, predicted_rollout, diff_rollout, input_water, DEM
    ):
        self._get_maxs(real_rollout, predicted_rollout, diff_rollout)
        self.DEMPlot = DEMPlotMap(DEM, **self.default_plot_kwargs)
        self._get_WDPlots(real_rollout, predicted_rollout, diff_rollout, input_water)
        self._get_FATPlots(real_rollout, predicted_rollout)

        self._get_VPlots(real_rollout, predicted_rollout, diff_rollout, input_water)

    def _get_maxs(self, real_rollout, predicted_rollout, diff_rollout):
        self.WD_max = max(real_rollout[:, 0, :].max(), predicted_rollout[:, 0, :].max())

        self.max_diff_WD = diff_rollout[:, 0, :].max()
        self.min_diff_WD = diff_rollout[:, 0, :].min()

        self.V_max = max(
            abs(predicted_rollout[:, 1:, :]).max(), abs(real_rollout[:, 1:, :]).max()
        )
        self.max_diff_V = diff_rollout[:, 1:, :].max()
        self.min_diff_V = diff_rollout[:, 1:, :].min()

    def _reset_maxs(self, *plotmap):
        for plot in plotmap:
            plot.kwargs.pop("vmax")

    def _plot_temporal_errors(self, diff_rollout, ax):
        axs = plot_rollout_diff_in_time_all(
            diff_rollout,
            ax=ax,
            type_loss=self.type_loss,
            temporal_res=self.temporal_res,
            time_start=self.time_start,
        )
        return axs

    def _plot_DEM(self, ax):
        self.DEMPlot.plot_map(ax=ax)
        self.DEMPlot._add_axes_info(ax=ax)
        self.DEMPlot._add_breach_location(
            ax=ax, breach_coordinates=self.breach_coordinates
        )

    def _get_WDPlots(self, real_rollout, predicted_rollout, diff_rollout, input_water):
        # Water depth
        self.real_WD = TemporalPlotMap(
            real_rollout[:, 0, :],
            **self.default_temporal_plot_kwargs,
            scaler=self.scalers["WD_scaler"],
            cmap=WD_color,
            vmax=self.WD_max,
        )

        self.predicted_WD = TemporalPlotMap(
            predicted_rollout[:, 0, :],
            **self.default_temporal_plot_kwargs,
            scaler=self.scalers["WD_scaler"],
            cmap=WD_color,
            vmin=0,
            vmax=self.WD_max,
        )

        self.difference_WD = TemporalPlotMap(
            diff_rollout[:, 0, :],
            **self.default_temporal_plot_kwargs,
            scaler=self.scalers["WD_scaler"],
            difference_plot=True,
            vmin=self.min_diff_WD,
            vmax=self.max_diff_WD,
        )

        self.init_WD = TemporalPlotMap(
            input_water[:, 0, :],
            **self.default_plot_kwargs
            | {"time_start": -1, "temporal_res": self.temporal_res},
            cmap=WD_color,
            vmax=self.WD_max,
        )

    def _get_VQuiverPlots(
        self, real_rollout, predicted_rollout, diff_rollout, input_water
    ):
        # Vector velocity
        self.real_V = QuiverPlotMap(
            real_rollout[:, 1, :],
            real_rollout[:, 2, :],
            **self.default_temporal_plot_kwargs,
            scaler=self.scalers["V_scaler"],
            cmap=V_color,
            vmax=self.V_max,
        )

        self.predicted_V = QuiverPlotMap(
            predicted_rollout[:, 1, :],
            predicted_rollout[:, 2, :],
            **self.default_temporal_plot_kwargs,
            scaler=self.scalers["V_scaler"],
            cmap=V_color,
            vmax=self.V_max,
        )

        self.difference_V = QuiverPlotMap(
            diff_rollout[:, 1, :],
            diff_rollout[:, 2, :],
            **self.default_temporal_plot_kwargs,
            scaler=self.scalers["V_scaler"],
            difference_plot=True,
            vmin=self.min_diff_V,
            vmax=self.max_diff_V,
        )

        self.init_V = QuiverPlotMap(
            input_water[:, 1, :],
            input_water[:, 2, :],
            **self.default_plot_kwargs
            | {"time_start": -1, "temporal_res": self.temporal_res},
            cmap=V_color,
            vmax=self.V_max,
        )

    def _get_VPlots(self, real_rollout, predicted_rollout, diff_rollout, input_water):
        # Scalar velocity
        self.real_V = TemporalPlotMap(
            real_rollout[:, 1, :],
            **self.default_temporal_plot_kwargs,
            scaler=self.scalers["V_scaler"],
            cmap=V_color,
            vmax=self.V_max,
        )

        self.predicted_V = TemporalPlotMap(
            predicted_rollout[:, 1, :],
            **self.default_temporal_plot_kwargs,
            scaler=self.scalers["V_scaler"],
            cmap=V_color,
            vmin=0,
            vmax=self.V_max,
        )

        self.difference_V = TemporalPlotMap(
            diff_rollout[:, 1, :],
            **self.default_temporal_plot_kwargs,
            scaler=self.scalers["V_scaler"],
            difference_plot=True,
            vmin=self.min_diff_V,
            vmax=self.max_diff_V,
        )

        self.init_V = TemporalPlotMap(
            input_water[:, 1, :],
            **self.default_plot_kwargs
            | {"time_start": -1, "temporal_res": self.temporal_res},
            cmap=V_color,
            vmax=self.V_max,
        )

    def _get_FATPlots(self, real_rollout, predicted_rollout):
        # Flood arrival times
        real_FAT = WD_to_FAT(
            real_rollout[:, 0, :],
            self.temporal_res,
            self.water_threshold,
            self.time_start,
        )
        predicted_FAT = WD_to_FAT(
            predicted_rollout[:, 0, :],
            self.temporal_res,
            self.water_threshold,
            self.time_start,
        )
        diff_FAT = real_FAT - predicted_FAT
        max_diff_FAT = diff_FAT.abs().max()

        self.pred_FATPlot = BasePlotMap(
            predicted_FAT, **self.default_plot_kwargs, cmap=FAT_color
        )
        self.real_FATPlot = BasePlotMap(
            real_FAT, **self.default_plot_kwargs, cmap=FAT_color
        )
        self.diff_FATPlot = BasePlotMap(
            diff_FAT,
            **self.default_plot_kwargs,
            difference_plot=True,
            vmin=-max_diff_FAT,
            vmax=max_diff_FAT,
        )

    def plot_BC(self, ax=None):
        """Plots boundary condition in time"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))

        type_BC_dict = {1: "Water depth [m]", 2: "Discharge [$m^3$/s]"}

        if isinstance(self.dataset.mesh, MultiscaleMesh):
            inflow_nodes = (
                self.dataset.BC[:1].T * self.dataset.edge_BC_length[:1]
            ).T  # only finest scale
        else:
            inflow_nodes = (self.dataset.BC.T * self.dataset.edge_BC_length).T

        type_BC = self.dataset.type_BC.item()
        time_vector = get_time_vector(inflow_nodes.shape[-1] - 1, self.temporal_res)
        [
            plot_line_with_deviation(
                time_vector, inflow, ax=ax, label=f"{type_BC_dict[type_BC]}"
            )
            for inflow in inflow_nodes.cpu()
        ]
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(f"{type_BC_dict[type_BC]}")
        ax.set_title("Boundary conditions")

        return ax

    def explore_rollout(self, time_step=-1, scale=None, logscale=False):
        fig, axs = plt.subplots(
            2,
            4,
            figsize=(6 * 4, 11),
            facecolor="white",
            gridspec_kw={"width_ratios": [1, 1, 1, 1]},
            constrained_layout=True,
        )

        if scale is not None:
            self.mesh_scale_plot(scale=scale)

        self._plot_DEM(ax=axs[0, 0])

        # water depth
        self._reset_maxs(self.real_WD, self.predicted_WD, self.difference_WD)
        self.real_WD.plot_map(time_step=time_step, ax=axs[0, 1])
        self.predicted_WD.plot_map(time_step=time_step, ax=axs[0, 2])
        self.difference_WD.plot_map(time_step=time_step, ax=axs[0, 3])

        axs[0, 1].set_ylabel("Water depth [m]")
        axs[0, 1].set_title("Ground-truth")
        axs[0, 2].set_title("Predicted")
        axs[0, 3].set_title("Difference")

        self._plot_temporal_errors(self.diff_rollout, ax=axs[1, 0])

        # velocities
        axs[1, 1].set_ylabel(f"{self.V_label} [{self.V_unit}]")

        self._reset_maxs(self.real_V, self.predicted_V, self.difference_V)
        self.real_V.plot_map(time_step=time_step, ax=axs[1, 1], logscale=logscale)
        self.predicted_V.plot_map(time_step=time_step, ax=axs[1, 2], logscale=logscale)
        self.difference_V.plot_map(time_step=time_step, ax=axs[1, 3])

        return fig

    def explore_multiscale_rollout(self, time_step=-1, variable="WD", logscale=False):
        """Plot the multiscale rollout of a variable (options: 'WD', 'V')"""
        assert isinstance(
            self.dataset.mesh, MultiscaleMesh
        ), "This function only works for multiscale meshes"
        fig, axs = plt.subplots(
            self.num_scales,
            4,
            figsize=(4 * 4, self.num_scales * 4),
            facecolor="white",
            gridspec_kw={"width_ratios": [1, 1, 1, 1]},
            constrained_layout=True,
        )

        for i in range(self.num_scales):
            self.mesh_scale_plot(scale=i)

            self._plot_DEM(ax=axs[i, 0])

            # water depth
            if variable == "WD":
                self.real_WD.plot_map(time_step=time_step, ax=axs[i, 1], colorbar=False)
                self.predicted_WD.plot_map(time_step=time_step, ax=axs[i, 2])
                self.difference_WD.plot_map(time_step=time_step, ax=axs[i, 3])
                axs[i, 1].set_ylabel("Water depth [m]")
            # velocities
            elif variable == "V":
                axs[i, 1].set_ylabel(f"{self.V_label} [{self.V_unit}]")
                self.real_V.plot_map(
                    time_step=time_step, ax=axs[i, 1], colorbar=False, logscale=logscale
                )
                self.predicted_V.plot_map(
                    time_step=time_step, ax=axs[i, 2], logscale=logscale
                )
                self.difference_V.plot_map(time_step=time_step, ax=axs[i, 3])

        axs[0, 1].set_title("Ground-truth")
        axs[0, 2].set_title("Predicted")
        axs[0, 3].set_title("Difference")

        return fig

    def compare_h_rollout(self, plot_times=[1, 6, 24, 40], scale=None):
        plot_times = plot_times + [-1]  # add final time step
        if scale is not None:
            self.mesh_scale_plot(scale=scale)

        n_plots = len(plot_times)
        width_ratios = [0.9] * (n_plots * 2 - 2) + [1, 1]
        fig = plt.figure(figsize=(n_plots * 4, 17), facecolor="white")
        spec = mpl.gridspec.GridSpec(
            ncols=2 * len(plot_times),
            nrows=9,
            height_ratios=[1, 1, 0.9, 1, 1, 1, 1, 1, 1],
            width_ratios=width_ratios,
        )

        ax01 = fig.add_subplot(spec[0:2, 1:3])
        ax02 = fig.add_subplot(spec[0:2, n_plots * 2 - 3 : n_plots * 2 - 1])

        self._plot_DEM(ax=ax01)
        self.plot_BC(ax02)
        # ax02.set_title('')
        # ax01.set_title('')

        colorbar = False
        for i, time_step in enumerate(plot_times):
            if time_step == -1:
                colorbar = True
            ax1 = fig.add_subplot(spec[3:5, i * 2 : i * 2 + 2])
            ax2 = fig.add_subplot(spec[5:7, i * 2 : i * 2 + 2])
            ax3 = fig.add_subplot(spec[7:9, i * 2 : i * 2 + 2])
            if i == 0:
                ax1.set_ylabel(f"Ground-truth [m]")
                ax2.set_ylabel(f"Predictions [m]")
                ax3.set_ylabel(f"Difference [m]")
            self.real_WD.plot_map(time_step=time_step, ax=ax1, colorbar=colorbar)
            self.predicted_WD.plot_map(time_step=time_step, ax=ax2, colorbar=colorbar)
            self.difference_WD.plot_map(time_step=time_step, ax=ax3, colorbar=colorbar)
            ax1.set_title(f"time: {self.real_WD.time_in_hours} h")

        fig.subplots_adjust(wspace=0, hspace=0)

        return None

    def compare_v_rollout(self, plot_times=[1, 6, 24, 40], scale=None, logscale=False):
        plot_times = plot_times + [-1]  # add final time step
        if scale is not None:
            self.mesh_scale_plot(scale=scale)

        n_plots = len(plot_times)
        width_ratios = [0.9] * (n_plots * 2 - 2) + [1, 1]
        fig = plt.figure(figsize=(n_plots * 4, 17), facecolor="white")
        spec = mpl.gridspec.GridSpec(
            ncols=2 * len(plot_times),
            nrows=9,
            height_ratios=[1, 1, 0.9, 1, 1, 1, 1, 1, 1],
            width_ratios=width_ratios,
        )

        ax01 = fig.add_subplot(spec[0:2, 1:3])
        ax02 = fig.add_subplot(spec[0:2, n_plots * 2 - 3 : n_plots * 2 - 1])

        self._plot_DEM(ax=ax01)
        self.plot_BC(ax02)

        colorbar = False
        for i, time_step in enumerate(plot_times):
            if time_step == -1:
                colorbar = True
            ax1 = fig.add_subplot(spec[3:5, i * 2 : i * 2 + 2])
            ax2 = fig.add_subplot(spec[5:7, i * 2 : i * 2 + 2])
            ax3 = fig.add_subplot(spec[7:9, i * 2 : i * 2 + 2])
            if i == 0:
                ax1.set_ylabel(f"Ground-truth [{self.V_unit}]")
                ax2.set_ylabel(f"Predictions [{self.V_unit}]")
                ax3.set_ylabel(f"Difference [{self.V_unit}]")

            self.real_V.plot_map(
                time_step=time_step, ax=ax1, colorbar=colorbar, logscale=logscale
            )
            self.predicted_V.plot_map(
                time_step=time_step, ax=ax2, colorbar=colorbar, logscale=logscale
            )
            self.difference_V.plot_map(time_step=time_step, ax=ax3, colorbar=colorbar)
            ax1.set_title(f"time: {self.real_V.time_in_hours} h")

        fig.subplots_adjust(wspace=0, hspace=0)

        return None

    def compare_FAT(self, water_threshold=0, scale=None):
        self.water_threshold = water_threshold
        if scale is not None:
            self.mesh_scale_plot(scale=scale)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")

        self.real_FATPlot.plot_map(ax=axs[0])
        self.pred_FATPlot.plot_map(ax=axs[1])
        self.diff_FATPlot.plot_map(ax=axs[2])

        axs[0].set_title("FAT Ground-truth [h]")
        axs[1].set_title("FAT Predictions [h]")
        axs[2].set_title("FAT Difference [h]")

        fig.tight_layout()

        return None

    def compare_Froude(self, time_step):
        real_vel = get_velocity(
            torch.norm(self.real_rollout[:, 1:, :], dim=1), self.real_rollout[:, 0, :]
        )
        predicted_vel = get_velocity(
            torch.norm(self.predicted_rollout[:, 1:, :], dim=1),
            self.predicted_rollout[:, 0, :],
        )
        self.real_fr = get_Froude(real_vel, self.real_rollout[:, 0, :])
        self.predicted_fr = get_Froude(predicted_vel, self.predicted_rollout[:, 0, :])
        self.diff_fr = self.real_fr - self.predicted_fr
        max_diff_fr = self.diff_fr[:, time_step].max()
        max_fr = max(self.real_fr.max(), self.predicted_fr.max())

        fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")

        real_FrPlot = TemporalPlotMap(
            self.real_fr, **self.default_temporal_plot_kwargs, cmap=V_color, vmin=0
        )

        pred_FrPlot = TemporalPlotMap(
            self.predicted_fr, **self.default_temporal_plot_kwargs, cmap=V_color, vmin=0
        )

        diff_FrPlot = TemporalPlotMap(
            self.diff_fr,
            **self.default_temporal_plot_kwargs,
            difference_plot=True,
            vmin=-max_diff_fr,
            vmax=max_diff_fr,
        )

        real_FrPlot.plot_map(time_step=time_step, ax=axs[0])
        pred_FrPlot.plot_map(time_step=time_step, ax=axs[1])
        diff_FrPlot.plot_map(time_step=time_step, ax=axs[2])

        axs[0].set_title("Froude Ground-truth [h]")
        axs[1].set_title("Froude Predictions [h]")
        axs[2].set_title("Froude Difference [h]")

        fig.tight_layout()

        return None

    def create_video(self, logscale=False, interval=200, blit=False, **anim_kwargs):
        """
        This function seems to work only on Jupyter webpages (not on Visual Studio Code)
        For more information on how to roll please refer to http://news.rr.nihalnavath.com/posts/rollout-d628137f
        """
        from IPython.display import clear_output
        from matplotlib.animation import FuncAnimation

        fig, axs = plt.subplots(
            2,
            4,
            figsize=(6 * 4, 11),
            facecolor="white",
            gridspec_kw={"width_ratios": [1, 1, 1, 1]},
            constrained_layout=True,
        )

        self._plot_DEM(ax=axs[0, 0])

        axs[1, 0].set_ylabel(self.type_loss)
        axs[1, 0].set_xlabel("Time [h]")
        average_diff_t = get_mean_error(self.diff_rollout, self.type_loss).cpu().numpy()
        max_avg_WD = average_diff_t[0].max()
        max_avg_V = average_diff_t[1].max()

        self.add_initial_colorbars(axs, logscale=logscale)

        def animate(time_step):
            """
            Function used to create video of the simulation
            """
            for axx in axs:
                for ax in axx[1:]:
                    ax.cla()

            axs[1, 0].cla()

            ax, axv = self._plot_temporal_errors(
                self.diff_rollout[:, :, :time_step], ax=axs[1, 0]
            )

            ax.set_xlim(
                0, (self.real_WD.total_time + self.time_start) * self.temporal_res / 60
            )
            ax.set_ylim(0, max_avg_WD * 1.1)
            axv.set_ylim(0, max_avg_V * 1.1)
            axv.ticklabel_format(style="sci", scilimits=(-1, 3), useMathText=True)
            ax.ticklabel_format(style="sci", scilimits=(-1, 3), useMathText=True)

            # water depth
            self.real_WD.plot_map(time_step=time_step, ax=axs[0, 1], colorbar=False)
            self.predicted_WD.plot_map(
                time_step=time_step, ax=axs[0, 2], colorbar=False
            )
            self.difference_WD.plot_map(
                time_step=time_step, ax=axs[0, 3], colorbar=False
            )

            current_time = self.real_WD.time_in_hours
            axs[0, 1].set_title(f"Ground-truth h [m]\ntime {current_time} h")
            axs[0, 2].set_title(f"Predicted h [m]\ntime {current_time} h")
            axs[0, 3].set_title(f"Difference h [m]\ntime {current_time} h")

            # velocities
            self.real_V.plot_map(
                time_step=time_step, ax=axs[1, 1], colorbar=False, logscale=logscale
            )
            self.predicted_V.plot_map(
                time_step=time_step, ax=axs[1, 2], colorbar=False, logscale=logscale
            )
            self.difference_V.plot_map(
                time_step=time_step, ax=axs[1, 3], colorbar=False
            )
            axs[1, 1].set_title(
                f"Ground-truth |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h"
            )
            axs[1, 2].set_title(
                f"Predicted |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h"
            )
            axs[1, 3].set_title(
                f"Difference |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h"
            )

            fig.subplots_adjust(wspace=0.4, hspace=0.3)

            clear_output(wait=True)
            print("It: %i" % time_step)
            sys.stdout.flush()
            return fig

        frames = self.real_WD.total_time
        self.anim = FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=blit, **anim_kwargs
        )
        plt.close()

    def create_multiscale_video(
        self, variable="WD", interval=200, blit=False, **anim_kwargs
    ):
        """
        Creates a video of the evolution of a given hydraulic variable at all scales
        (Valid only for multiscale meshes)

        Parameters:
        variable: name of the hydraulic variable to plot in video (options: 'WD' or 'V')
        """
        from IPython.display import clear_output
        from matplotlib.animation import FuncAnimation

        assert isinstance(
            self.dataset.mesh, MultiscaleMesh
        ), "This function only works for multiscale meshes"
        fig, axs = plt.subplots(
            self.num_scales,
            4,
            figsize=(5 * 4, self.num_scales * 4),
            facecolor="white",
            gridspec_kw={"width_ratios": [1, 1, 1, 1]},
            constrained_layout=True,
        )

        axs[0, 1].set_title("Ground-truth")
        axs[0, 2].set_title("Predicted")
        axs[0, 3].set_title("Difference")

        for i in range(self.num_scales):
            self.mesh_scale_plot(scale=i)

            self.DEMPlot.plot_map(ax=axs[i, 0])
            self.DEMPlot._add_axes_info(
                ax=axs[i, 0], title=False, x_label=i // (self.num_scales - 1)
            )
            if i == 0:
                axs[i, 0].set_title("DEM (m)")
                self.DEMPlot._add_breach_location(
                    ax=axs[i, 0], breach_coordinates=self.breach_coordinates
                )

            # water depth
            if variable == "WD":
                self.predicted_WD.kwargs["vmin"] = 0
                self.predicted_WD.kwargs["vmax"] = self.WD_max
                self.predicted_WD._get_cmap()
                self.predicted_WD._add_colorbar(ax=axs[i, 2], colorbar=True)

                self.difference_WD._get_cmap()
                self.difference_WD._add_colorbar(ax=axs[i, 3], colorbar=True)
            # velocities
            elif variable == "V":
                self.predicted_V.kwargs["vmin"] = 0
                self.predicted_V.kwargs["vmax"] = self.V_max
                self.predicted_V._get_cmap()
                self.predicted_V._add_colorbar(ax=axs[i, 2], colorbar=True)

                self.difference_V._get_cmap()
                self.difference_V._add_colorbar(ax=axs[i, 3], colorbar=True)

        def animate(time_step):
            """
            Function used to create video of the simulation
            """
            for axx in axs:
                for ax in axx[1:]:
                    ax.cla()

            for i in range(self.num_scales):
                self.mesh_scale_plot(scale=i)
                # water depth
                if variable == "WD":
                    self.real_WD.plot_map(
                        time_step=time_step, ax=axs[i, 1], colorbar=False
                    )
                    self.predicted_WD.plot_map(
                        time_step=time_step, ax=axs[i, 2], colorbar=False
                    )
                    self.difference_WD.plot_map(
                        time_step=time_step, ax=axs[i, 3], colorbar=False
                    )
                    current_time = self.real_WD.time_in_hours
                    axs[0, 1].set_title(f"Ground-truth h [m]\ntime {current_time} h")
                    axs[0, 2].set_title(f"Predicted h [m]\ntime {current_time} h")
                    axs[0, 3].set_title(f"Difference h [m]\ntime {current_time} h")
                # velocities
                elif variable == "V":
                    axs[i, 1].set_ylabel(f"{self.V_label} [{self.V_unit}]")
                    self.real_V.plot_map(
                        time_step=time_step, ax=axs[i, 1], colorbar=False
                    )
                    self.predicted_V.plot_map(
                        time_step=time_step, ax=axs[i, 2], colorbar=False
                    )
                    self.difference_V.plot_map(
                        time_step=time_step, ax=axs[i, 3], colorbar=False
                    )
                    current_time = self.real_V.time_in_hours
                    axs[1, 1].set_title(
                        f"Ground-truth |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h"
                    )
                    axs[1, 2].set_title(
                        f"Predicted |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h"
                    )
                    axs[1, 3].set_title(
                        f"Difference |{self.V_symbol}| [{self.V_unit}]\ntime {current_time} h"
                    )
                else:
                    for ax in axs[1, 1:]:
                        ax.axis("off")

            fig.subplots_adjust(wspace=0.4, hspace=0.3)

            clear_output(wait=True)
            print("It: %i" % time_step)
            sys.stdout.flush()
            return fig

        if variable == "WD":
            frames = self.real_WD.total_time
        elif variable == "V":
            frames = self.real_V.total_time
        self.anim = FuncAnimation(
            fig, animate, frames=frames, interval=interval, blit=blit, **anim_kwargs
        )
        plt.close()

    def add_initial_colorbars(self, axs, logscale=False):
        self.predicted_WD.kwargs["vmin"] = 0
        self.predicted_WD.kwargs["vmax"] = self.WD_max
        self.predicted_WD._get_cmap()
        self.predicted_WD._add_colorbar(ax=axs[0, 2], colorbar=True)

        self.difference_WD._get_cmap()
        self.difference_WD._add_colorbar(ax=axs[0, 3], colorbar=True)

        if not logscale:
            self.predicted_V.kwargs["vmin"] = 0
            self.predicted_V.kwargs["vmax"] = self.V_max
        self.predicted_V._get_cmap()
        self.predicted_V._add_colorbar(ax=axs[1, 2], colorbar=True, logscale=logscale)

        self.difference_V._get_cmap()
        self.difference_V._add_colorbar(ax=axs[1, 3], colorbar=True)

    def save_video(self, path, fps=5, dpi=250, **save_kwargs):
        self.anim.save(
            f"{path}.mp4",
            writer="ffmpeg",
            fps=fps,
            dpi=dpi,
            metadata={"title": "test_dataset", "artist": "Roberto Bentivoglio"},
            **save_kwargs,
        )

    def HTML_plot(self):
        from IPython.display import HTML

        HTML(self.anim.to_html5_video())

    def _get_CSI(self, water_threshold=0):
        return get_CSI(
            self.predicted_rollout, self.real_rollout, water_threshold=water_threshold
        )

    def _get_F1(self, water_threshold=0):
        return get_F1(
            self.predicted_rollout, self.real_rollout, water_threshold=water_threshold
        )

    def _plot_metric(self, metric_name="CSI", water_thresholds=[0.05, 0.3], ax=None):
        """Plots metric in time for different water_thresholds
        -------
        metric_function:
            options: CSI, F1
        """
        metrics_dict = {"CSI": self._get_CSI, "F1": self._get_F1}
        metric_function = metrics_dict[metric_name]

        ax = ax or plt.gca()

        for wt in water_thresholds:
            metric = metric_function(water_threshold=wt).to("cpu").numpy()
            metric = add_null_time_start(self.time_start, metric)
            plot_line_with_deviation(
                self.time_vector, metric, label=f"{metric_name}_{wt}"
            )

        ax.set_xlabel("Time [h]")
        ax.set_ylabel(f"{metric_name} score")
        ax.set_ylim(0, 1)
        ax.grid()
        ax.legend(loc=4)

        return ax

    def _plot_mass_conservation(self, normalized=True, with_cum=True, ax=None):
        """Plots mass conservation in time for one or more simulations
        ------
        normalized: bool (default=True)
            Volume error is divided by the inflow volume
        with_cum: bool (default=True)
            Plots cumulative error distribution in time
        """
        ax = ax or plt.gca()

        mass_loss = (
            get_mass_conservation_loss(
                self.predicted_rollout[
                    self.dataset.node_ptr[0] : self.dataset.node_ptr[1]
                ],
                self.dataset,
            ).cpu()
            * 1e6
        )
        mass_loss = add_null_time_start(self.time_start + 1, mass_loss)

        inflow_volume = torch.tensor(
            [
                get_inflow_volume(self.dataset, self.dataset.BC[:, t : t + 2].mean(1))
                for t in range(mass_loss.shape[0] - 1)
            ]
        )
        inflow_volume = add_null_time_start(self.time_start, inflow_volume)

        if with_cum:
            cum_mass_loss = np.nancumsum(mass_loss, axis=-1)
            if normalized:
                cum_mass_loss = cum_mass_loss / np.nancumsum(inflow_volume, axis=-1)
            plot_line_with_deviation(self.time_vector, cum_mass_loss, label="Cumulated")
        else:
            if normalized:
                mass_loss = mass_loss / inflow_volume
            plot_line_with_deviation(
                self.time_vector, mass_loss, label=r"Per $\Delta$t"
            )

        ax.set_title("Mass conservation")
        ax.set_xlabel("Time [h]")
        if normalized:
            ax.set_ylabel("Normalized volume error [-]")
        else:
            ax.set_ylabel("Volume error [$m^3$]")
        ax.grid()
        ax.legend()

        return ax

    def _get_rollout_loss(self, type_loss="RMSE", only_where_water=False):
        return get_rollout_loss(
            self.predicted_rollout,
            self.real_rollout,
            type_loss=type_loss,
            only_where_water=only_where_water,
        )
