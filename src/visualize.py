"""
Visualization Module.

Provides visualization capabilities for the crowd flow mapping simulation
including animations, flow field plots, and error heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

from .config import VizConfig


@dataclass
class PlotData:
    """Container for simulation plot data."""
    people_state: List[Dict]
    scan_result: List[Dict]
    ground_truth: Optional[np.ndarray]
    estimated_flow: Optional[np.ndarray]
    error_heatmap: Optional[np.ndarray]
    robot_pose: Tuple[float, float, float]
    step: int
    time: float


class Visualizer:
    """
    Visualization engine for the crowd flow mapping simulation.

    Supports static plots, animations, and various visualization modes
    including quiver plots for flow fields and heatmaps for errors.

    Attributes:
        config: Visualization configuration
    """

    def __init__(self, config: Optional[VizConfig] = None):
        """
        Initialize the visualizer.

        Args:
            config: Visualization configuration (uses default if None)
        """
        self.config = config or VizConfig()
        self._figure: Optional[plt.Figure] = None
        self._axes: Dict[str, plt.Axes] = {}
        self._animation: Optional[FuncAnimation] = None
        self._frame_data: List[PlotData] = []

    def create_figure(
        self,
        mode: str = "full",
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a figure with appropriate subplots.

        Args:
            mode: Visualization mode - "full", "simulation", "flow", "error"
            figsize: Optional figure size override

        Returns:
            Matplotlib figure object
        """
        if figsize is None:
            figsize = self.config.figure_size

        if mode == "full":
            fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=self.config.dpi)
            self._axes = {
                "simulation": axes[0, 0],
                "robot": axes[0, 1],
                "flow": axes[1, 0],
                "error": axes[1, 1]
            }
        elif mode == "simulation":
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=self.config.dpi)
            self._axes = {
                "simulation": axes[0],
                "robot": axes[1]
            }
        elif mode == "flow":
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=self.config.dpi)
            self._axes = {
                "ground_truth": axes[0],
                "estimated": axes[1]
            }
        elif mode == "error":
            fig, axes = plt.subplots(1, 1, figsize=(8, 6), dpi=self.config.dpi)
            self._axes = {"error": axes}
        else:
            fig, axes = plt.subplots(1, 1, figsize=figsize, dpi=self.config.dpi)
            self._axes = {"main": axes}

        self._figure = fig
        return fig

    def plot_people(
        self,
        people_state: List[Dict],
        ax: Optional[plt.Axes] = None,
        show_velocity: bool = True
    ) -> None:
        """
        Plot people as circles with optional velocity vectors.

        Args:
            people_state: List of person state dictionaries
            ax: Matplotlib axes (uses "simulation" if None)
            show_velocity: Whether to show velocity arrows
        """
        if ax is None:
            ax = self._axes.get("simulation")
        if ax is None:
            return

        ax.clear()

        positions_x = [p["x"] for p in people_state]
        positions_y = [p["y"] for p in people_state]

        if positions_x:
            circles = [
                Circle((p["x"], p["y"]), radius=0.3, facecolor=self.config.person_color,
                       edgecolor='black', alpha=0.7)
                for p in people_state
            ]
            collection = PatchCollection(circles, match_original=True)
            ax.add_collection(collection)

            ax.scatter(positions_x, positions_y, c=self.config.person_color,
                      s=20, zorder=5, edgecolors='black', linewidths=0.5)

            if show_velocity:
                for person in people_state:
                    dx = person["vx"] * 0.5
                    dy = person["vy"] * 0.5
                    ax.arrow(person["x"], person["y"], dx, dy,
                            head_width=0.1, head_length=0.05,
                            fc='darkblue', ec='darkblue', alpha=0.5)

        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('People Simulation')
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

    def plot_robot(
        self,
        robot_pose: Tuple[float, float, float],
        scan_result: List[Dict],
        ax: Optional[plt.Axes] = None,
        scan_range: float = 10.0,
        fov: float = 2.094
    ) -> None:
        """
        Plot robot with its scan field of view.

        Args:
            robot_pose: Robot pose (x, y, heading)
            scan_result: List of scan detections
            ax: Matplotlib axes (uses "robot" if None)
            scan_range: Scan range in meters
            fov: Field of view in radians
        """
        if ax is None:
            ax = self._axes.get("robot")
        if ax is None:
            return

        ax.clear()

        rx, ry, heading = robot_pose

        half_fov = fov / 2
        wedge = Wedge(
            (rx, ry), scan_range,
            np.degrees(heading - half_fov),
            np.degrees(heading + half_fov),
            facecolor=self.config.robot_scan_color,
            alpha=0.2,
            edgecolor=self.config.robot_scan_color,
            linewidth=1
        )
        ax.add_patch(wedge)

        detection_x = [d["x"] for d in scan_result]
        detection_y = [d["y"] for d in scan_result]
        if detection_x:
            ax.scatter(detection_x, detection_y, c='green', s=100,
                      marker='*', zorder=10, label='Detected')

        ax.scatter([rx], [ry], c=self.config.robot_color, s=200,
                  marker='^', zorder=15, label='Robot')

        arrow_len = 1.5
        ax.arrow(rx, ry, arrow_len * np.cos(heading),
                arrow_len * np.sin(heading),
                head_width=0.3, head_length=0.2,
                fc=self.config.robot_color, ec=self.config.robot_color,
                zorder=15)

        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Scanner')
        ax.legend(loc='upper right')
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

    def plot_flow_field(
        self,
        flow_field: np.ndarray,
        ax: Optional[plt.Axes] = None,
        mode: str = "quiver",
        title: str = "Flow Field",
        grid_resolution: float = 1.0,
        show_density: bool = True
    ) -> None:
        """
        Plot a flow field using quiver arrows or density.

        Args:
            flow_field: Flow field array (H, W, 3)
            ax: Matplotlib axes (uses "flow" if None)
            mode: Plot mode - "quiver", "density", "combined"
            title: Plot title
            grid_resolution: Grid cell size
            show_density: Whether to show density as background
        """
        if ax is None:
            ax = self._axes.get("flow")
        if ax is None:
            return

        ax.clear()

        density = flow_field[:, :, 0]
        vx = flow_field[:, :, 1]
        vy = flow_field[:, :, 2]

        h, w = density.shape
        extent = [0, w * grid_resolution, 0, h * grid_resolution]

        if show_density and mode in ("combined", "density"):
            im = ax.imshow(density, extent=extent, origin='lower',
                          cmap=self.config.flow_field_colormap, alpha=0.7,
                          aspect='equal', vmin=0.0, vmax=1.0)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Density', rotation=270, labelpad=15, fontsize=9)

        if mode in ("quiver", "combined"):
            skip = max(1, min(h, w) // 10)

            y_indices = np.arange(0, h, skip)
            x_indices = np.arange(0, w, skip)

            quiver_vx = vx[np.ix_(y_indices, x_indices)]
            quiver_vy = vy[np.ix_(y_indices, x_indices)]

            x_coords = (x_indices + 0.5) * grid_resolution
            y_coords = (y_indices + 0.5) * grid_resolution

            magnitude = np.sqrt(quiver_vx**2 + quiver_vy**2)
            norm = Normalize(vmin=0, vmax=np.max(magnitude) if np.max(magnitude) > 0 else 1)

            ax.quiver(x_coords, y_coords, quiver_vx, quiver_vy,
                     magnitude, cmap='plasma', scale=self.config.arrow_scale * 20,
                     norm=norm, width=self.config.arrow_width)

        ax.set_xlim(0, w * grid_resolution)
        ax.set_ylim(0, h * grid_resolution)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        if self.config.show_grid:
            ax.grid(True, alpha=0.3, color='white', linestyle='--')

    def plot_error_heatmap(
        self,
        error_field: np.ndarray,
        ax: Optional[plt.Axes] = None,
        title: str = "Error Heatmap",
        grid_resolution: float = 1.0
    ) -> None:
        """
        Plot error heatmap.

        Args:
            error_field: Error array (H, W)
            ax: Matplotlib axes (uses "error" if None)
            title: Plot title
            grid_resolution: Grid cell size
        """
        if ax is None:
            ax = self._axes.get("error")
        if ax is None:
            return

        ax.clear()

        h, w = error_field.shape
        extent = [0, w * grid_resolution, 0, h * grid_resolution]

        im = ax.imshow(error_field, extent=extent, origin='lower',
                      cmap=self.config.error_colormap, aspect='auto')
        plt.colorbar(im, ax=ax, label='Error')

        ax.set_xlim(0, w * grid_resolution)
        ax.set_ylim(0, h * grid_resolution)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        if self.config.show_grid:
            ax.grid(True, alpha=0.3, color='white', linestyle='--')

    def plot_comparison(
        self,
        ground_truth: np.ndarray,
        estimated: np.ndarray,
        grid_resolution: float = 1.0
    ) -> None:
        """
        Create a side-by-side comparison of ground truth and estimated flow fields.

        Args:
            ground_truth: Ground truth flow field
            estimated: Estimated flow field
            grid_resolution: Grid cell size
        """
        self.create_figure(mode="flow")
        self.plot_flow_field(ground_truth,
                           self._axes.get("ground_truth"),
                           mode="combined",
                           title="Ground Truth",
                           grid_resolution=grid_resolution)
        self.plot_flow_field(estimated,
                           self._axes.get("estimated"),
                           mode="combined",
                           title="Estimated",
                           grid_resolution=grid_resolution)
        plt.tight_layout()

    def record_frame(self, plot_data: PlotData) -> None:
        """
        Record a frame for animation.

        Args:
            plot_data: Plot data container
        """
        self._frame_data.append(plot_data)

    def create_simulation_animation(
        self,
        grid_resolution: float = 1.0,
        scan_range: float = 10.0,
        fov: float = 2.094
    ) -> FuncAnimation:
        """
        Create an animation of the simulation.

        Args:
            grid_resolution: Grid cell size
            scan_range: Robot scan range
            fov: Robot field of view

        Returns:
            Matplotlib FuncAnimation object
        """
        if not self._frame_data:
            warnings.warn("No frame data recorded. Call record_frame() first.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size,
                                dpi=self.config.dpi)
        fig_axes = {
            "simulation": axes[0, 0],
            "robot": axes[0, 1],
            "ground_truth": axes[1, 0],
            "estimated": axes[1, 1]
        }

        def update(frame_idx):
            if frame_idx >= len(self._frame_data):
                return

            data = self._frame_data[frame_idx]

            self.plot_people(data.people_state, fig_axes["simulation"])

            self.plot_robot(data.robot_pose, data.scan_result,
                          fig_axes["robot"], scan_range, fov)

            if data.ground_truth is not None:
                self.plot_flow_field(data.ground_truth, fig_axes["ground_truth"],
                                    mode="combined",
                                    title=f"Ground Truth (t={data.time:.1f}s)",
                                    grid_resolution=grid_resolution)

            if data.estimated_flow is not None:
                self.plot_flow_field(data.estimated_flow, fig_axes["estimated"],
                                    mode="combined",
                                    title=f"Estimated (t={data.time:.1f}s)",
                                    grid_resolution=grid_resolution)

            fig.suptitle(f"Step {data.step} - Time {data.time:.1f}s")

        self._animation = FuncAnimation(
            fig, update,
            frames=len(self._frame_data),
            interval=1000 // self.config.animation_fps,
            blit=False
        )

        return self._animation

    def save_animation(
        self,
        animation: FuncAnimation,
        filename: str,
        fps: Optional[int] = None,
        dpi: Optional[int] = None
    ) -> None:
        """
        Save animation to file.

        Args:
            animation: Matplotlib FuncAnimation object
            filename: Output filename
            fps: Frames per second
            dpi: Output DPI
        """
        if fps is None:
            fps = self.config.animation_fps
        if dpi is None:
            dpi = self.config.animation_dpi

        filename_lower = filename.lower()
        if filename_lower.endswith('.gif'):
            writer = PillowWriter(fps=fps)
        elif filename_lower.endswith('.mp4'):
            writer = FFMpegWriter(fps=fps)
        else:
            writer = PillowWriter(fps=fps)

        animation.save(filename, writer=writer, dpi=dpi)

    def plot_error_history(
        self,
        error_history: List[float],
        ax: Optional[plt.Axes] = None
    ) -> None:
        """
        Plot error over time.

        Args:
            error_history: List of error values
            ax: Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(error_history, linewidth=2, color='red')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Combined Error Score')
        ax.set_title('Error Over Time')
        ax.grid(True, alpha=0.3)

        if len(error_history) > 0:
            ax.fill_between(range(len(error_history)), error_history,
                           alpha=0.3, color='red')

    def create_dashboard(
        self,
        people_state: List[Dict],
        scan_result: List[Dict],
        ground_truth: np.ndarray,
        estimated: np.ndarray,
        error: float,
        robot_pose: Tuple[float, float, float],
        grid_resolution: float = 1.0
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with all visualizations.

        Args:
            people_state: Current people states
            scan_result: Current scan results
            ground_truth: Ground truth flow field
            estimated: Estimated flow field
            error: Current error value
            robot_pose: Robot pose
            grid_resolution: Grid cell size

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 10), dpi=self.config.dpi)

        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax_sim = fig.add_subplot(gs[0, 0])
        self.plot_people(people_state, ax_sim)

        ax_robot = fig.add_subplot(gs[0, 1])
        self.plot_robot(robot_pose, scan_result, ax_robot)

        ax_stats = fig.add_subplot(gs[0, 2])
        ax_stats.axis('off')
        stats_text = f"Error: {error:.4f}\nPeople: {len(people_state)}\nDetections: {len(scan_result)}"
        ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                     fontsize=14, verticalalignment='center',
                     horizontalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax_gt = fig.add_subplot(gs[1, 0])
        self.plot_flow_field(ground_truth, ax_gt, mode="combined",
                            title="Ground Truth", grid_resolution=grid_resolution)

        ax_est = fig.add_subplot(gs[1, 1])
        self.plot_flow_field(estimated, ax_est, mode="combined",
                            title="Estimated", grid_resolution=grid_resolution)

        ax_error = fig.add_subplot(gs[1, 2])
        error_field = np.abs(ground_truth[:, :, 1] - estimated[:, :, 1])
        + np.abs(ground_truth[:, :, 2] - estimated[:, :, 2])
        self.plot_error_heatmap(error_field, ax_error,
                               title="Velocity Error",
                               grid_resolution=grid_resolution)

        ax_combined = fig.add_subplot(gs[2, :])
        self.plot_comparison_minimal(ground_truth, estimated, ax_combined,
                                    grid_resolution)

        return fig

    def plot_comparison_minimal(
        self,
        ground_truth: np.ndarray,
        estimated: np.ndarray,
        ax: plt.Axes,
        grid_resolution: float = 1.0
    ) -> None:
        """
        Create a minimal combined comparison plot.

        Args:
            ground_truth: Ground truth flow field
            estimated: Estimated flow field
            ax: Matplotlib axes
            grid_resolution: Grid cell size
        """
        ax.clear()

        diff = np.sqrt(
            (ground_truth[:, :, 1] - estimated[:, :, 1])**2 +
            (ground_truth[:, :, 2] - estimated[:, :, 2])**2
        )

        h, w = diff.shape
        extent = [0, w * grid_resolution, 0, h * grid_resolution]

        im = ax.imshow(diff, extent=extent, origin='lower',
                      cmap='hot_r', aspect='auto')
        plt.colorbar(im, ax=ax, label='Velocity Difference')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Velocity Error Distribution')
        ax.grid(True, alpha=0.3, color='white', linestyle='--')

    def show(self) -> None:
        """Display all pending plots."""
        plt.show()

    def save_figure(self, filename: str, dpi: Optional[int] = None) -> None:
        """
        Save current figure to file.

        Args:
            filename: Output filename
            dpi: Output DPI
        """
        if self._figure is not None:
            if dpi is None:
                dpi = self.config.dpi
            self._figure.savefig(filename, dpi=dpi, bbox_inches='tight')

    def close(self) -> None:
        """Close all figures and clear state."""
        plt.close('all')
        self._figure = None
        self._axes.clear()
        self._animation = None
        self._frame_data.clear()

    def clear_frames(self) -> None:
        """Clear recorded frames."""
        self._frame_data.clear()
