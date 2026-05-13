"""
Main Module for Crowd Flow Mapping Simulation.

This module orchestrates all simulation components including the people
simulator, robot scanner, flow mapper, flow analyzer, and visualizer.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt

from src.config import (
    GlobalConfig, SimulatorConfig, ScannerConfig, GridConfig,
    MapperConfig, AnalyzerConfig, VizConfig, RobotConfig,
    SimulationConfig, create_default_config, create_test_config
)
from src.people_simulator import PeopleSimulator
from src.robot_scanner import RobotScanner
from src.flow_mapper import FlowMapper
from src.flow_analyzer import FlowAnalyzer
from src.visualize import Visualizer, PlotData


@dataclass
class SimulationStep:
    """Container for data from a single simulation step."""
    step: int
    time: float
    people_state: List[Dict]
    scan_result: List[Dict]
    ground_truth: Optional[np.ndarray] = None
    estimated_flow: Optional[np.ndarray] = None
    errors: Optional[Dict] = None


@dataclass
class SimulationResults:
    """Container for complete simulation results."""
    config: GlobalConfig
    steps: List[SimulationStep] = field(default_factory=list)
    final_errors: Optional[Dict] = None
    error_summary: Optional[Dict] = None
    total_time: float = 0.0

    def get_error_history(self) -> List[float]:
        """Get combined error score over time."""
        return [s.errors["combined_score"] for s in self.steps if s.errors]

    def get_velocity_error_history(self) -> List[float]:
        """Get velocity MSE over time."""
        return [s.errors["velocity_mse"] for s in self.steps if s.errors]

    def get_density_error_history(self) -> List[float]:
        """Get density MSE over time."""
        return [s.errors["density_mse"] for s in self.steps if s.errors]

    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return {
            "config": self.config.to_dict(),
            "num_steps": len(self.steps),
            "final_errors": self.final_errors,
            "error_summary": self.error_summary,
            "total_time": self.total_time
        }


class RobotController:
    """
    Controls robot movement and scanning behavior.

    Supports multiple patrol modes including static, random walk,
    grid-based scanning, and flow-following.
    """

    def __init__(self, config: RobotConfig, scanner: RobotScanner):
        """
        Initialize robot controller.

        Args:
            config: Robot configuration
            scanner: Robot scanner instance
        """
        self.config = config
        self.scanner = scanner
        self._patrol_targets: List[Tuple[float, float]] = []
        self._current_target_idx: int = 0
        self._velocity = np.zeros(2)

    def initialize(self) -> Tuple[float, float, float]:
        """
        Initialize robot position and heading.

        Returns:
            Tuple of (x, y, heading)
        """
        x = self.config.initial_x
        y = self.config.initial_y
        heading = self.config.initial_heading
        self.scanner.set_pose(x, y, heading)
        return x, y, heading

    def update(
        self,
        current_pose: Tuple[float, float, float],
        dt: float,
        flow_field: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """
        Update robot pose based on patrol mode.

        Args:
            current_pose: Current (x, y, heading)
            dt: Time step
            flow_field: Optional flow field for adaptive movement

        Returns:
            Updated (x, y, heading)
        """
        x, y, heading = current_pose

        if self.config.patrol_mode == "static":
            pass

        elif self.config.patrol_mode == "random":
            # Random heading changes - increased probability for better exploration
            if random.random() < 0.1:  # 10% chance per step
                heading += random.uniform(-np.pi/2, np.pi/2)

            # Move robot
            x += self.config.movement_speed * np.cos(heading) * dt
            y += self.config.movement_speed * np.sin(heading) * dt

            # Environment boundaries (sync with simulator config)
            # For test config: 10x10 environment
            min_bound = 0.5
            max_bound = 9.5

            # Clip to bounds and reflect heading
            if x < min_bound:
                x = min_bound
                heading = np.pi - heading  # Reflect
            elif x > max_bound:
                x = max_bound
                heading = np.pi - heading  # Reflect

            if y < min_bound:
                y = min_bound
                heading = -heading  # Reflect
            elif y > max_bound:
                y = max_bound
                heading = -heading  # Reflect

            # Normalize heading to [-π, π]
            while heading > np.pi:
                heading -= 2 * np.pi
            while heading < -np.pi:
                heading += 2 * np.pi

        elif self.config.patrol_mode == "grid":
            if not self._patrol_targets:
                self._generate_grid_targets()
            self._move_to_target(
                (x, y, heading), dt
            )

        self._velocity = np.array([
            self.config.movement_speed * np.cos(heading),
            self.config.movement_speed * np.sin(heading)
        ])

        self.scanner.set_pose(x, y, heading)
        self.scanner.set_velocity(self._velocity[0], self._velocity[1])

        return x, y, heading

    def _generate_grid_targets(self) -> None:
        """Generate grid-based patrol targets within environment bounds."""
        # Environment is 10x10, use grid points within bounds
        grid_size = 5.0
        margin = 1.0  # Keep patrol away from edges
        for gx in np.arange(margin, 10 - margin, grid_size):
            for gy in np.arange(margin, 10 - margin, grid_size):
                self._patrol_targets.append((gx, gy))
        random.shuffle(self._patrol_targets)

    def _move_to_target(
        self,
        current_pose: Tuple[float, float, float],
        dt: float
    ) -> None:
        """Move towards current patrol target."""
        x, y, heading = current_pose

        if self._current_target_idx >= len(self._patrol_targets):
            self._current_target_idx = 0
            random.shuffle(self._patrol_targets)

        target_x, target_y = self._patrol_targets[self._current_target_idx]
        dx = target_x - x
        dy = target_y - y
        dist = np.sqrt(dx*dx + dy*dy)

        if dist < 0.5:
            self._current_target_idx += 1
            return

        target_heading = np.arctan2(dy, dx)
        heading_diff = target_heading - heading
        while heading_diff > np.pi:
            heading_diff -= 2 * np.pi
        while heading_diff < -np.pi:
            heading_diff += 2 * np.pi

        heading += np.clip(heading_diff, -self.config.rotation_speed * dt,
                          self.config.rotation_speed * dt)

        x += self.config.movement_speed * np.cos(heading) * dt
        y += self.config.movement_speed * np.sin(heading) * dt


class SimulationRunner:
    """
    Main simulation orchestrator.

    Coordinates all simulation components and manages the simulation loop.
    """

    def __init__(self, config: Optional[GlobalConfig] = None):
        """
        Initialize the simulation runner.

        Args:
            config: Global configuration (uses default if None)
        """
        self.config = config or create_default_config()
        self.config.sync_grid_with_simulator()

        self.simulator = PeopleSimulator(self.config.simulator)
        self.scanner = RobotScanner(self.config.scanner)
        self.mapper = FlowMapper(self.config.grid, self.config.mapper)
        self.analyzer = FlowAnalyzer(self.config.analyzer)
        self.visualizer = Visualizer(self.config.viz)
        self.robot_controller = RobotController(self.config.robot, self.scanner)

        self.results: SimulationResults = SimulationResults(config=self.config)
        self.current_step: int = 0
        self.current_time: float = 0.0
        self.robot_pose: Tuple[float, float, float] = (0, 0, 0)

        self._is_setup: bool = False
        self._is_running: bool = False

    def setup(self, seed: Optional[int] = None) -> None:
        """
        Set up the simulation.

        Args:
            seed: Optional random seed
        """
        if seed is not None:
            self.config.simulation.seed = seed
            np.random.seed(seed)
            random.seed(seed)

        self.simulator.spawn_people(
            count=self.config.simulator.num_people,
            distribution="random"
        )

        self.robot_pose = self.robot_controller.initialize()

        self.mapper.reset()
        self.analyzer.reset()
        self.visualizer.clear_frames()

        self.results = SimulationResults(config=self.config)
        self.current_step = 0
        self.current_time = 0.0

        self._is_setup = True

    def step(self) -> Dict:
        """
        Execute one simulation step.

        Returns:
            Dictionary with step results
        """
        if not self._is_setup:
            raise RuntimeError("Call setup() before running step()")

        dt = self.config.simulation.timestep

        self.simulator.update(dt)

        self.robot_pose = self.robot_controller.update(
            self.robot_pose, dt, self.mapper.get_flow_field()
        )

        people_state = self.simulator.get_state()
        scan_result = self.scanner.scan(people_state)

        ground_truth = self.simulator.get_ground_truth_flow_field(self.config.grid)

        robot_vel = (
            float(self.robot_controller._velocity[0]),
            float(self.robot_controller._velocity[1])
        )
        estimated_flow = self.mapper.update(scan_result, self.robot_pose, robot_vel)

        errors = self.analyzer.compute_errors(ground_truth, estimated_flow)

        step_data = SimulationStep(
            step=self.current_step,
            time=self.current_time,
            people_state=people_state,
            scan_result=scan_result,
            ground_truth=ground_truth.copy(),
            estimated_flow=estimated_flow.copy(),
            errors=errors
        )
        self.results.steps.append(step_data)

        plot_data = PlotData(
            people_state=people_state,
            scan_result=scan_result,
            ground_truth=ground_truth,
            estimated_flow=estimated_flow,
            error_heatmap=None,
            robot_pose=self.robot_pose,
            step=self.current_step,
            time=self.current_time
        )
        self.visualizer.record_frame(plot_data)

        self.current_step += 1
        self.current_time += dt

        return {
            "step": self.current_step - 1,
            "time": self.current_time,
            "num_people": len(people_state),
            "num_detections": len(scan_result),
            "errors": errors
        }

    def run(
        self,
        num_steps: Optional[int] = None,
        visualize: bool = False,
        verbose: bool = True
    ) -> SimulationResults:
        """
        Run the simulation for multiple steps.

        Args:
            num_steps: Number of steps (uses config if None)
            visualize: Whether to create real-time visualization
            verbose: Whether to print progress

        Returns:
            SimulationResults object
        """
        if num_steps is None:
            num_steps = self.config.simulation.max_steps

        if not self._is_setup:
            self.setup()

        self._is_running = True
        start_time = time.time()

        try:
            for i in range(num_steps):
                self.step()

                if verbose and (i + 1) % 10 == 0:
                    errors = self.results.steps[-1].errors
                    print(f"Step {i+1}/{num_steps} | "
                          f"Time: {self.current_time:.1f}s | "
                          f"Detections: {len(self.results.steps[-1].scan_result)} | "
                          f"Error: {errors['combined_score']:.4f}")

                if visualize and (i + 1) % 5 == 0:
                    self._update_visualization()

        finally:
            self._is_running = False

        self.results.total_time = time.time() - start_time
        self.results.final_errors = self.results.steps[-1].errors if self.results.steps else None
        self.results.error_summary = self.analyzer.get_error_summary()

        if verbose:
            self._print_summary()

        return self.results

    def run_episode(self, max_steps: Optional[int] = None) -> SimulationResults:
        """
        Run a complete simulation episode.

        Alias for run() with default settings.

        Args:
            max_steps: Maximum steps

        Returns:
            SimulationResults object
        """
        return self.run(num_steps=max_steps, verbose=True)

    def _update_visualization(self) -> None:
        """Update real-time visualization."""
        if self.results.steps:
            latest = self.results.steps[-1]
            self.visualizer.create_figure(mode="full")
            self.visualizer.plot_people(latest.people_state)
            self.visualizer.plot_robot(
                self.robot_pose,
                latest.scan_result,
                scan_range=self.config.scanner.scan_range,
                fov=self.config.scanner.fov
            )
            if latest.ground_truth is not None:
                self.visualizer.plot_flow_field(
                    latest.ground_truth,
                    mode="combined",
                    title="Ground Truth",
                    grid_resolution=self.config.grid.grid_resolution
                )
            if latest.estimated_flow is not None:
                self.visualizer.plot_flow_field(
                    latest.estimated_flow,
                    mode="combined",
                    title="Estimated",
                    grid_resolution=self.config.grid.grid_resolution
                )
            plt.pause(0.01)

    def _print_summary(self) -> None:
        """Print simulation summary."""
        print("\n" + "=" * 50)
        print("Simulation Complete")
        print("=" * 50)
        print(f"Total Steps: {len(self.results.steps)}")
        print(f"Total Time: {self.current_time:.2f}s")
        print(f"Execution Time: {self.results.total_time:.2f}s")
        if self.results.final_errors:
            print(f"\nFinal Error Score: {self.results.final_errors['combined_score']:.6f}")
            print(f"Velocity MSE: {self.results.final_errors['velocity_mse']:.6f}")
            print(f"Density MSE: {self.results.final_errors['density_mse']:.6f}")
        print("=" * 50)

    def get_results(self) -> SimulationResults:
        """Get simulation results."""
        return self.results

    def save_results(self, path: str) -> None:
        """
        Save simulation results to file.

        Args:
            path: Output path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".json":
            with open(output_path, 'w') as f:
                json.dump(self.results.to_dict(), f, indent=2)
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            np.save(output_path / "steps.npy",
                   [s.__dict__ for s in self.results.steps])
            with open(output_path / "summary.json", 'w') as f:
                json.dump(self.results.to_dict(), f, indent=2)

    def create_animation(self) -> 'FuncAnimation':
        """
        Create animation from recorded frames.

        Returns:
            Matplotlib FuncAnimation object
        """
        return self.visualizer.create_simulation_animation(
            grid_resolution=self.config.grid.grid_resolution,
            scan_range=self.config.scanner.scan_range,
            fov=self.config.scanner.fov
        )

    def plot_final_state(self, save_path: str = None) -> None:
        """Plot final simulation state."""
        if not self.results.steps:
            return

        latest = self.results.steps[-1]

        fig = self.visualizer.create_figure(mode="full")

        self.visualizer.plot_people(latest.people_state)

        self.visualizer.plot_robot(
            self.robot_pose,
            latest.scan_result,
            scan_range=self.config.scanner.scan_range,
            fov=self.config.scanner.fov
        )

        if latest.ground_truth is not None and latest.estimated_flow is not None:
            # Plot ground truth on the flow axis (bottom-left)
            self.visualizer.plot_flow_field(
                latest.ground_truth,
                self.visualizer._axes.get("flow"),
                mode="combined",
                title="Ground Truth",
                grid_resolution=self.config.grid.grid_resolution
            )

            # Plot estimated on the error axis (bottom-right)
            self.visualizer.plot_flow_field(
                latest.estimated_flow,
                self.visualizer._axes.get("error"),
                mode="combined",
                title="Estimated",
                grid_resolution=self.config.grid.grid_resolution
            )

        fig.subplots_adjust(hspace=0.3, wspace=0.45, top=0.92, bottom=0.08)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            self.visualizer.show()

    def reset(self) -> None:
        """Reset the simulation runner."""
        self.simulator.reset()
        self.mapper.reset()
        self.analyzer.reset()
        self.visualizer.close()
        self.results = SimulationResults(config=self.config)
        self.current_step = 0
        self.current_time = 0.0
        self._is_setup = False


def main():
    """Main entry point."""
    import matplotlib.pyplot as plt

    config = create_test_config()

    runner = SimulationRunner(config)
    runner.setup()

    results = runner.run(num_steps=100, verbose=True)

    runner.plot_final_state()

    runner.visualizer.close()


if __name__ == "__main__":
    main()
