"""
Flow Mapper Module.

Reconstructs the local flow field based on robot scanner measurements
using various interpolation methods on a 2D grid.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    gaussian_filter = None

from .config import GridConfig, MapperConfig


class FlowMapper:
    """
    Reconstructs flow field from scan measurements using grid-based interpolation.

    The mapper maintains a 2D grid where each cell stores:
    - density: number of detected people in the cell
    - avg_velocity: average velocity vector of detected people

    Attributes:
        grid_config: Grid configuration
        mapper_config: Mapper configuration
    """

    def __init__(
        self,
        grid_config: Optional[GridConfig] = None,
        mapper_config: Optional[MapperConfig] = None
    ):
        """
        Initialize the flow mapper.

        Args:
            grid_config: Grid configuration (uses default if None)
            mapper_config: Mapper configuration (uses default if None)
        """
        self.grid_config = grid_config or GridConfig()
        self.mapper_config = mapper_config or MapperConfig()

        self._initialize_grids()

        self._flow_field_history: List[np.ndarray] = []
        self._max_history = 100

    def _initialize_grids(self) -> None:
        """Initialize the flow field grids."""
        num_cells_x = self.grid_config.num_cells_x
        num_cells_y = self.grid_config.num_cells_y

        self._density_grid = np.zeros((num_cells_y, num_cells_x))
        self._velocity_x_grid = np.zeros((num_cells_y, num_cells_x))
        self._velocity_y_grid = np.zeros((num_cells_y, num_cells_x))
        self._count_grid = np.zeros((num_cells_y, num_cells_x))
        self._confidence_grid = np.zeros((num_cells_y, num_cells_x))
        self._cumulative_velocity_x = np.zeros((num_cells_y, num_cells_x))
        self._cumulative_velocity_y = np.zeros((num_cells_y, num_cells_x))

    def update(
        self,
        scan_result: List[Dict],
        robot_pose: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Update the flow field based on new scan results.

        Args:
            scan_result: List of scan detections from the robot scanner
            robot_pose: Robot pose (x, y, heading)

        Returns:
            Updated flow field array of shape (H, W, 3)
        """
        robot_x, robot_y, robot_heading = robot_pose

        self._reset_temp_grids()

        if not scan_result:
            self._apply_temporal_smoothing()
            return self.get_flow_field()

        for detection in scan_result:
            self._process_detection(detection, robot_pose)

        self._apply_interpolation()

        self._apply_temporal_smoothing()

        flow_field = self.get_flow_field()
        self._flow_field_history.append(flow_field.copy())

        if len(self._flow_field_history) > self._max_history:
            self._flow_field_history.pop(0)

        return flow_field

    def _reset_temp_grids(self) -> None:
        """Reset temporary grids for new measurement integration."""
        self._temp_density = np.zeros_like(self._density_grid)
        self._temp_velocity_x = np.zeros_like(self._velocity_x_grid)
        self._temp_velocity_y = np.zeros_like(self._velocity_y_grid)
        self._temp_count = np.zeros_like(self._count_grid)

    def _process_detection(
        self,
        detection: Dict,
        robot_pose: Tuple[float, float, float]
    ) -> None:
        """
        Process a single scan detection and update relevant cells.

        Args:
            detection: Scan detection dictionary
            robot_pose: Robot pose (x, y, heading)
        """
        robot_x, robot_y, robot_heading = robot_pose

        det_x = detection["x"]
        det_y = detection["y"]

        cell_x, cell_y = self._world_to_grid(det_x, det_y)
        if not self._is_valid_cell(cell_x, cell_y):
            return

        rel_vel = detection["relative_velocity"]

        if self.mapper_config.update_method == "nearest":
            self._update_nearest(cell_x, cell_y, detection, rel_vel)
        elif self.mapper_config.update_method == "gaussian":
            self._update_gaussian(cell_x, cell_y, detection, rel_vel, robot_pose)
        elif self.mapper_config.update_method == "linear":
            self._update_linear(cell_x, cell_y, detection, rel_vel)
        else:
            self._update_nearest(cell_x, cell_y, detection, rel_vel)

    def _update_nearest(
        self,
        cell_x: int,
        cell_y: int,
        detection: Dict,
        rel_vel: List[float]
    ) -> None:
        """Update using nearest neighbor method."""
        self._temp_density[cell_y, cell_x] += 1
        self._temp_velocity_x[cell_y, cell_x] += rel_vel[0]
        self._temp_velocity_y[cell_y, cell_x] += rel_vel[1]
        self._temp_count[cell_y, cell_x] += 1

    def _update_gaussian(
        self,
        cell_x: int,
        cell_y: int,
        detection: Dict,
        rel_vel: List[float],
        robot_pose: Tuple[float, float, float]
    ) -> None:
        """Update using Gaussian kernel weighting."""
        kernel_radius_cells = int(
            self.mapper_config.kernel_radius / self.grid_config.grid_resolution
        )
        kernel_radius_cells = max(1, kernel_radius_cells)

        half_size = kernel_radius_cells
        y_start = max(0, cell_y - half_size)
        y_end = min(self._density_grid.shape[0], cell_y + half_size + 1)
        x_start = max(0, cell_x - half_size)
        x_end = min(self._density_grid.shape[1], cell_x + half_size + 1)

        sigma = kernel_radius_cells / 2.0

        for cy in range(y_start, y_end):
            for cx in range(x_start, x_end):
                dist = np.sqrt((cx - cell_x)**2 + (cy - cell_y)**2)
                weight = np.exp(-(dist**2) / (2 * sigma**2))

                self._temp_density[cy, cx] += weight
                self._temp_velocity_x[cy, cx] += weight * rel_vel[0]
                self._temp_velocity_y[cy, cx] += weight * rel_vel[1]
                self._temp_count[cy, cx] += weight

    def _update_linear(
        self,
        cell_x: int,
        cell_y: int,
        detection: Dict,
        rel_vel: List[float]
    ) -> None:
        """Update using linear interpolation (bilinear)."""
        radius = 1
        y_start = max(0, cell_y - radius)
        y_end = min(self._density_grid.shape[0], cell_y + radius + 1)
        x_start = max(0, cell_x - radius)
        x_end = min(self._density_grid.shape[1], cell_x + radius + 1)

        for cy in range(y_start, y_end):
            for cx in range(x_start, x_end):
                dist = np.sqrt((cx - cell_x)**2 + (cy - cell_y)**2)
                if dist <= radius + 0.5:
                    weight = max(0, 1 - dist / (radius + 1))

                    self._temp_density[cy, cx] += weight
                    self._temp_velocity_x[cy, cx] += weight * rel_vel[0]
                    self._temp_velocity_y[cy, cx] += weight * rel_vel[1]
                    self._temp_count[cy, cx] += weight

    def _apply_interpolation(self) -> None:
        """Apply interpolation results to main grids."""
        mask = self._temp_count > 0

        avg_vx = np.zeros_like(self._temp_velocity_x)
        avg_vy = np.zeros_like(self._temp_velocity_y)

        avg_vx[mask] = self._temp_velocity_x[mask] / self._temp_count[mask]
        avg_vy[mask] = self._temp_velocity_y[mask] / self._temp_count[mask]

        self._density_grid[mask] = self._temp_density[mask]
        self._velocity_x_grid[mask] = avg_vx[mask]
        self._velocity_y_grid[mask] = avg_vy[mask]
        self._count_grid[mask] = self._temp_count[mask]

        self._cumulative_velocity_x[mask] += avg_vx[mask]
        self._cumulative_velocity_y[mask] += avg_vy[mask]

        self._confidence_grid[mask] = np.minimum(
            self._confidence_grid[mask] + 0.1, 1.0
        )
        self._confidence_grid[~mask] = np.maximum(
            self._confidence_grid[~mask] * 0.95, 0.0
        )

    def _apply_temporal_smoothing(self) -> None:
        """Apply temporal smoothing to reduce noise."""
        if self.mapper_config.temporal_smoothing < 1.0:
            decay = self.mapper_config.temporal_smoothing
            self._density_grid *= decay
            self._velocity_x_grid *= decay
            self._velocity_y_grid *= decay

    def _world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid indices.

        Args:
            world_x: X position in world coordinates
            world_y: Y position in world coordinates

        Returns:
            Tuple of (grid_x, grid_y) indices
        """
        grid_x = int((world_x - self.grid_config.origin_x) / self.grid_config.grid_resolution)
        grid_y = int((world_y - self.grid_config.origin_y) / self.grid_config.grid_resolution)
        return grid_x, grid_y

    def _grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        Convert grid indices to world coordinates.

        Args:
            grid_x: Grid X index
            grid_y: Grid Y index

        Returns:
            Tuple of (world_x, world_y) coordinates
        """
        world_x = self.grid_config.origin_x + (grid_x + 0.5) * self.grid_config.grid_resolution
        world_y = self.grid_config.origin_y + (grid_y + 0.5) * self.grid_config.grid_resolution
        return world_x, world_y

    def _is_valid_cell(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid indices are valid."""
        return (
            0 <= grid_x < self.grid_config.num_cells_x and
            0 <= grid_y < self.grid_config.num_cells_y
        )

    def get_flow_field(self) -> np.ndarray:
        """
        Get the current flow field.

        Returns:
            Flow field array of shape (H, W, 3)
            [.., 0] = density
            [.., 1] = vx (average velocity x)
            [.., 2] = vy (average velocity y)
        """
        flow_field = np.zeros((*self._density_grid.shape, 3))
        flow_field[:, :, 0] = self._density_grid
        flow_field[:, :, 1] = self._velocity_x_grid
        flow_field[:, :, 2] = self._velocity_y_grid
        return flow_field

    def get_density_grid(self) -> np.ndarray:
        """Get the density grid."""
        return self._density_grid.copy()

    def get_velocity_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the velocity grids."""
        return self._velocity_x_grid.copy(), self._velocity_y_grid.copy()

    def get_confidence_grid(self) -> np.ndarray:
        """Get the confidence grid."""
        return self._confidence_grid.copy()

    def get_grid_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the X and Y coordinates of grid cell centers.

        Returns:
            Tuple of (X, Y) coordinate arrays of shape (H, W)
        """
        num_cells_x = self.grid_config.num_cells_x
        num_cells_y = self.grid_config.num_cells_y

        x_coords = np.zeros((num_cells_y, num_cells_x))
        y_coords = np.zeros((num_cells_y, num_cells_x))

        for cy in range(num_cells_y):
            for cx in range(num_cells_x):
                x_coords[cy, cx], y_coords[cy, cx] = self._grid_to_world(cx, cy)

        return x_coords, y_coords

    def get_flow_magnitude(self) -> np.ndarray:
        """
        Get the magnitude of the flow velocity at each cell.

        Returns:
            Velocity magnitude array of shape (H, W)
        """
        return np.sqrt(self._velocity_x_grid**2 + self._velocity_y_grid**2)

    def get_flow_direction(self) -> np.ndarray:
        """
        Get the direction (angle) of the flow velocity at each cell.

        Returns:
            Direction angle array in radians of shape (H, W)
        """
        return np.arctan2(self._velocity_y_grid, self._velocity_x_grid)

    def reset(self) -> None:
        """Reset the flow mapper to initial state."""
        self._initialize_grids()
        self._flow_field_history.clear()

    def apply_gaussian_smoothing(self, sigma: float = 1.0) -> None:
        """
        Apply Gaussian smoothing to the flow field.

        Args:
            sigma: Standard deviation for Gaussian kernel
        """
        if not HAS_SCIPY or gaussian_filter is None:
            return
        self._density_grid = gaussian_filter(self._density_grid, sigma)
        self._velocity_x_grid = gaussian_filter(self._velocity_x_grid, sigma)
        self._velocity_y_grid = gaussian_filter(self._velocity_y_grid, sigma)

    def normalize_density(self) -> None:
        """Normalize density values to [0, 1] range."""
        max_density = np.max(self._density_grid)
        if max_density > 0:
            self._density_grid /= max_density

    def get_coverage(self) -> float:
        """
        Get the percentage of grid cells that have been observed.

        Returns:
            Coverage percentage (0-1)
        """
        total_cells = self._density_grid.size
        observed_cells = np.sum(self._confidence_grid > 0)
        return observed_cells / total_cells if total_cells > 0 else 0.0

    def get_statistics(self) -> Dict:
        """Get flow field statistics."""
        return {
            "total_density": float(np.sum(self._density_grid)),
            "avg_density": float(np.mean(self._density_grid)),
            "max_density": float(np.max(self._density_grid)),
            "avg_velocity_magnitude": float(np.mean(self.get_flow_magnitude())),
            "coverage": self.get_coverage(),
            "num_cells_x": self.grid_config.num_cells_x,
            "num_cells_y": self.grid_config.num_cells_y
        }
