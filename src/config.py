"""
Configuration module for crowd flow mapping simulation.

This module centralizes all configuration parameters for the simulation,
scanner, mapper, analyzer, and visualization components.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path
import yaml


@dataclass
class SimulatorConfig:
    """Configuration for the people simulator."""
    width: float = 20.0           # Environment width (m)
    height: float = 20.0          # Environment height (m)
    num_people: int = 50          # Initial number of people
    min_speed: float = 0.3        # Minimum velocity magnitude (m/s)
    max_speed: float = 1.5        # Maximum velocity magnitude (m/s)
    person_radius: float = 0.3    # Person radius (m)
    boundary_behavior: str = "reflect"  # "bounce" | "wrap" | "reflect"
    velocity_std: float = 0.1     # Velocity random perturbation std
    interaction_range: float = 1.5  # Person interaction range (m)
    interaction_strength: float = 0.5  # Repulsion strength between people


@dataclass
class ScannerConfig:
    """Configuration for the robot scanner."""
    scan_range: float = 10.0      # Maximum scan range (m)
    fov: float = 2.094            # Field of view (rad) ~120 degrees
    num_rays: int = 36            # Number of scan rays
    noise_std_distance: float = 0.1  # Distance noise std (m)
    noise_std_velocity: float = 0.05  # Velocity noise std (m/s)
    noise_std_angle: float = 2.0  # Angle noise std (degrees)
    min_detectable_distance: float = 0.5  # Minimum detectable distance (m)


@dataclass
class GridConfig:
    """Configuration for the flow field grid."""
    grid_resolution: float = 1.0  # Grid cell size (m)
    origin_x: float = 0.0         # Grid origin X
    origin_y: float = 0.0         # Grid origin Y
    _width: float = 20.0          # Internal width storage
    _height: float = 20.0         # Internal height storage

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        self._width = value

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        self._height = value

    @property
    def num_cells_x(self) -> int:
        return int(self.width / self.grid_resolution)

    @property
    def num_cells_y(self) -> int:
        return int(self.height / self.grid_resolution)


@dataclass
class MapperConfig:
    """Configuration for the flow field mapper."""
    update_method: str = "gaussian"  # "nearest" | "gaussian" | "linear"
    kernel_radius: float = 2.0      # Gaussian kernel radius (m)
    confidence_threshold: float = 0.1  # Minimum confidence to update cell
    temporal_smoothing: float = 0.7  # Temporal smoothing factor (0-1)
    velocity_estimation_window: int = 5  # Number of frames for velocity estimation


@dataclass
class AnalyzerConfig:
    """Configuration for the flow field analyzer."""
    error_metric: str = "mse"      # "mse" | "mae" | "rmse"
    velocity_weight: float = 0.7   # Weight for velocity error
    density_weight: float = 0.3    # Weight for density error
    report_format: str = "json"    # "text" | "json"
    save_error_maps: bool = True   # Whether to save error heatmaps


@dataclass
class VizConfig:
    """Configuration for the visualizer."""
    figure_size: Tuple[int, int] = (12, 10)
    dpi: int = 100
    colormap: str = "viridis"
    arrow_scale: float = 0.3
    arrow_width: float = 0.005
    show_grid: bool = True
    save_format: str = "png"       # "mp4" | "gif" | "png"
    animation_fps: int = 30
    animation_dpi: int = 100
    person_color: str = "blue"
    robot_color: str = "red"
    robot_scan_color: str = "green"
    flow_field_colormap: str = "RdYlBu_r"
    error_colormap: str = "hot"


@dataclass
class RobotConfig:
    """Configuration for robot behavior."""
    initial_x: float = 10.0       # Initial X position
    initial_y: float = 10.0        # Initial Y position
    initial_heading: float = 0.0  # Initial heading (rad)
    movement_speed: float = 1.0   # Movement speed (m/s)
    rotation_speed: float = 0.5   # Rotation speed (rad/s)
    patrol_mode: str = "random"   # "random" | "grid" | "follow_flow"
    step_pattern: str = "static"   # "static" | "patrol" | "adaptive"


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""
    timestep: float = 0.1         # Simulation timestep (s)
    max_steps: int = 500          # Maximum simulation steps
    seed: Optional[int] = 42      # Random seed for reproducibility
    save_results: bool = True
    results_dir: str = "results"
    data_dir: str = "data"


@dataclass
class GlobalConfig:
    """Global configuration container for all subsystems."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    mapper: MapperConfig = field(default_factory=MapperConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "GlobalConfig":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(
            simulation=SimulationConfig(**data.get('simulation', {})),
            simulator=SimulatorConfig(**data.get('simulator', {})),
            scanner=ScannerConfig(**data.get('scanner', {})),
            grid=GridConfig(**data.get('grid', {})),
            mapper=MapperConfig(**data.get('mapper', {})),
            analyzer=AnalyzerConfig(**data.get('analyzer', {})),
            viz=VizConfig(**data.get('viz', {})),
            robot=RobotConfig(**data.get('robot', {}))
        )

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalConfig":
        """Create configuration from a dictionary."""
        return cls(
            simulation=SimulationConfig(**data.get('simulation', {})),
            simulator=SimulatorConfig(**data.get('simulator', {})),
            scanner=ScannerConfig(**data.get('scanner', {})),
            grid=GridConfig(**data.get('grid', {})),
            mapper=MapperConfig(**data.get('mapper', {})),
            analyzer=AnalyzerConfig(**data.get('analyzer', {})),
            viz=VizConfig(**data.get('viz', {})),
            robot=RobotConfig(**data.get('robot', {}))
        )

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary."""
        return {
            'simulation': self.simulation.__dict__,
            'simulator': self.simulator.__dict__,
            'scanner': self.scanner.__dict__,
            'grid': self.grid.__dict__,
            'mapper': self.mapper.__dict__,
            'analyzer': self.analyzer.__dict__,
            'viz': self.viz.__dict__,
            'robot': self.robot.__dict__
        }

    def save(self, path: str) -> None:
        """Save configuration to a YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def default(cls) -> "GlobalConfig":
        """Get default configuration."""
        return cls()

    def sync_grid_with_simulator(self) -> None:
        """Synchronize grid dimensions with simulator configuration."""
        self.grid.width = self.simulator.width
        self.grid.height = self.simulator.height


def create_default_config() -> GlobalConfig:
    """Factory function to create default configuration."""
    return GlobalConfig.default()


def create_test_config() -> GlobalConfig:
    """Factory function to create a test configuration with smaller values."""
    config = GlobalConfig()
    config.simulator.width = 10.0
    config.simulator.height = 10.0
    config.simulator.num_people = 20
    config.scanner.scan_range = 5.0
    config.scanner.num_rays = 24
    config.scanner.min_detectable_distance = 0.1
    config.robot.initial_x = 5.0
    config.robot.initial_y = 5.0
    config.robot.patrol_mode = "random"
    config.robot.scan_range = 5.0
    config.simulation.max_steps = 100
    config.sync_grid_with_simulator()
    return config
