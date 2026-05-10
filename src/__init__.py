"""
Crowd Flow Mapping Package.

A simulation system for robot-based crowd flow field reconstruction.
"""

from .config import (
    GlobalConfig,
    SimulatorConfig,
    ScannerConfig,
    GridConfig,
    MapperConfig,
    AnalyzerConfig,
    VizConfig,
    RobotConfig,
    SimulationConfig,
    create_default_config,
    create_test_config
)

from .people_simulator import PeopleSimulator, Person
from .robot_scanner import RobotScanner, ScanResult
from .flow_mapper import FlowMapper
from .flow_analyzer import FlowAnalyzer
from .visualize import Visualizer, PlotData

__version__ = "0.1.0"

__all__ = [
    "GlobalConfig",
    "SimulatorConfig",
    "ScannerConfig",
    "GridConfig",
    "MapperConfig",
    "AnalyzerConfig",
    "VizConfig",
    "RobotConfig",
    "SimulationConfig",
    "create_default_config",
    "create_test_config",
    "PeopleSimulator",
    "Person",
    "RobotScanner",
    "ScanResult",
    "FlowMapper",
    "FlowAnalyzer",
    "Visualizer",
    "PlotData"
]
