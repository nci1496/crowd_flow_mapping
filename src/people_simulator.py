"""
People Simulator Module.

Simulates crowd movement in a 2D environment with configurable behaviors
including random walking, boundary handling, and person-person interactions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .config import SimulatorConfig, GridConfig


@dataclass
class Person:
    """Represents a single person in the simulation."""
    id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.3

    def to_dict(self) -> Dict:
        """Convert person to dictionary format."""
        return {
            "id": self.id,
            "x": float(self.x),
            "y": float(self.y),
            "vx": float(self.vx),
            "vy": float(self.vy)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Person":
        """Create person from dictionary."""
        return cls(
            id=data["id"],
            x=data["x"],
            y=data["y"],
            vx=data["vx"],
            vy=data["vy"],
            radius=data.get("radius", 0.3)
        )


class PeopleSimulator:
    """
    Simulates crowd movement in a 2D environment.

    Attributes:
        config: Simulator configuration parameters
        people: List of Person objects
        state: Numpy array of shape (N, 4) containing [x, y, vx, vy]
        next_id: Counter for generating unique person IDs
    """

    def __init__(self, config: SimulatorConfig):
        """
        Initialize the people simulator.

        Args:
            config: SimulatorConfig containing simulation parameters
        """
        self.config = config
        self.people: List[Person] = []
        self.state: Optional[np.ndarray] = None
        self.next_id: int = 0
        self.time: float = 0.0
        self.step_count: int = 0

    def spawn_people(
        self,
        count: Optional[int] = None,
        distribution: str = "random",
        positions: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None
    ) -> None:
        """
        Spawn people in the environment.

        Args:
            count: Number of people to spawn (uses config.num_people if None)
            distribution: Spawn distribution - "random", "clustered", or "uniform"
            positions: Optional predefined positions array (N, 2)
            velocities: Optional predefined velocities array (N, 2)
        """
        if count is None:
            count = self.config.num_people

        self.people = []
        self.next_id = 0

        if positions is not None:
            for i in range(len(positions)):
                pos = positions[i]
                vel = velocities[i] if velocities is not None else self._generate_velocity()
                self._add_person(pos[0], pos[1], vel[0], vel[1])
        elif distribution == "random":
            for _ in range(count):
                x = np.random.uniform(
                    self.config.person_radius,
                    self.config.width - self.config.person_radius
                )
                y = np.random.uniform(
                    self.config.person_radius,
                    self.config.height - self.config.person_radius
                )
                vx, vy = self._generate_velocity()
                self._add_person(x, y, vx, vy)
        elif distribution == "clustered":
            num_clusters = max(3, count // 10)
            cluster_centers = np.random.uniform(
                [self.config.person_radius] * 2,
                [self.config.width - self.config.person_radius,
                 self.config.height - self.config.person_radius],
                size=(num_clusters, 2)
            )
            cluster_sizes = np.random.multinomial(count, [1/num_clusters] * num_clusters)

            for c_idx, c_size in enumerate(cluster_sizes):
                center = cluster_centers[c_idx]
                for _ in range(c_size):
                    angle = np.random.uniform(0, 2 * np.pi)
                    r = np.random.exponential(self.config.person_radius * 2)
                    x = np.clip(center[0] + r * np.cos(angle),
                                self.config.person_radius,
                                self.config.width - self.config.person_radius)
                    y = np.clip(center[1] + r * np.sin(angle),
                               self.config.person_radius,
                               self.config.height - self.config.person_radius)
                    vx, vy = self._generate_velocity()
                    self._add_person(x, y, vx, vy)
        elif distribution == "uniform":
            grid_size = int(np.ceil(np.sqrt(count)))
            spacing_x = self.config.width / grid_size
            spacing_y = self.config.height / grid_size
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= count:
                        break
                    x = (i + 0.5) * spacing_x
                    y = (j + 0.5) * spacing_y
                    vx, vy = self._generate_velocity()
                    self._add_person(x, y, vx, vy)
                    idx += 1
                if idx >= count:
                    break
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        self._update_state()

    def _add_person(self, x: float, y: float, vx: float, vy: float) -> Person:
        """Add a person to the simulation."""
        person = Person(
            id=self.next_id,
            x=x, y=y, vx=vx, vy=vy,
            radius=self.config.person_radius
        )
        self.people.append(person)
        self.next_id += 1
        return person

    def _generate_velocity(self) -> Tuple[float, float]:
        """Generate a random velocity vector."""
        speed = np.random.uniform(self.config.min_speed, self.config.max_speed)
        angle = np.random.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle) + np.random.normal(0, self.config.velocity_std)
        vy = speed * np.sin(angle) + np.random.normal(0, self.config.velocity_std)
        return vx, vy

    def _update_state(self) -> None:
        """Update the numpy state array from people list."""
        n = len(self.people)
        self.state = np.zeros((n, 4))
        for i, person in enumerate(self.people):
            self.state[i] = [person.x, person.y, person.vx, person.vy]

    def update(self, dt: float) -> None:
        """
        Update all people positions for one timestep.

        Args:
            dt: Time step in seconds
        """
        if self.state is None or len(self.people) == 0:
            return

        self._compute_interactions()

        positions = self.state[:, :2]
        velocities = self.state[:, 2:4]

        positions += velocities * dt

        self._handle_boundaries()

        self._update_people_from_state()

        self.time += dt
        self.step_count += 1

    def _compute_interactions(self) -> None:
        """Compute repulsion forces between nearby people."""
        if len(self.people) < 2:
            return

        positions = self.state[:, :2]
        n = len(self.people)

        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)

                if dist < self.config.interaction_range and dist > 0.001:
                    repulsion_strength = (
                        self.config.interaction_strength *
                        (self.config.interaction_range - dist) / self.config.interaction_range
                    )
                    fx = -repulsion_strength * dx / dist
                    fy = -repulsion_strength * dy / dist

                    self.state[i, 2] += fx
                    self.state[i, 3] += fy
                    self.state[j, 2] -= fx
                    self.state[j, 3] -= fy

    def _handle_boundaries(self) -> None:
        """Handle boundary collisions based on configured behavior."""
        positions = self.state[:, :2]
        velocities = self.state[:, 2:4]

        if self.config.boundary_behavior == "reflect":
            mask_left = positions[:, 0] < self.config.person_radius
            mask_right = positions[:, 0] > self.config.width - self.config.person_radius
            mask_bottom = positions[:, 1] < self.config.person_radius
            mask_top = positions[:, 1] > self.config.height - self.config.person_radius

            positions[:, 0] = np.clip(
                positions[:, 0],
                self.config.person_radius,
                self.config.width - self.config.person_radius
            )
            positions[:, 1] = np.clip(
                positions[:, 1],
                self.config.person_radius,
                self.config.height - self.config.person_radius
            )

            velocities[mask_left | mask_right, 0] *= -1
            velocities[mask_bottom | mask_top, 1] *= -1

        elif self.config.boundary_behavior == "bounce":
            mask_left = positions[:, 0] < self.config.person_radius
            mask_right = positions[:, 0] > self.config.width - self.config.person_radius
            mask_bottom = positions[:, 1] < self.config.person_radius
            mask_top = positions[:, 1] > self.config.height - self.config.person_radius

            positions[mask_left, 0] = self.config.person_radius
            positions[mask_right, 0] = self.config.width - self.config.person_radius
            positions[mask_bottom, 1] = self.config.person_radius
            positions[mask_top, 1] = self.config.height - self.config.person_radius

            velocities[mask_left | mask_right, 0] *= -0.9
            velocities[mask_bottom | mask_top, 1] *= -0.9

        elif self.config.boundary_behavior == "wrap":
            positions[:, 0] = np.mod(
                positions[:, 0] - self.config.person_radius,
                self.config.width - 2 * self.config.person_radius
            ) + self.config.person_radius
            positions[:, 1] = np.mod(
                positions[:, 1] - self.config.person_radius,
                self.config.height - 2 * self.config.person_radius
            ) + self.config.person_radius

    def _update_people_from_state(self) -> None:
        """Sync people list with updated state array."""
        for i, person in enumerate(self.people):
            person.x = self.state[i, 0]
            person.y = self.state[i, 1]
            person.vx = self.state[i, 2]
            person.vy = self.state[i, 3]

    def get_state(self) -> List[Dict]:
        """
        Get current state of all people.

        Returns:
            List of dictionaries with person state
        """
        return [person.to_dict() for person in self.people]

    def get_positions(self) -> np.ndarray:
        """Get positions of all people as numpy array."""
        if self.state is None:
            return np.zeros((0, 2))
        return self.state[:, :2].copy()

    def get_velocities(self) -> np.ndarray:
        """Get velocities of all people as numpy array."""
        if self.state is None:
            return np.zeros((0, 2))
        return self.state[:, 2:4].copy()

    def get_ground_truth_flow_field(self, grid_config: GridConfig) -> np.ndarray:
        """
        Compute the ground truth flow field on a grid.

        Args:
            grid_config: Grid configuration

        Returns:
            Flow field array of shape (grid_h, grid_w, 3)
            [.., 0] = density (people per cell)
            [.., 1] = vx (average velocity x)
            [.., 2] = vy (average velocity y)
        """
        num_cells_x = int(grid_config.width / grid_config.grid_resolution)
        num_cells_y = int(grid_config.height / grid_config.grid_resolution)

        flow_field = np.zeros((num_cells_y, num_cells_x, 3))

        if self.state is None or len(self.people) == 0:
            return flow_field

        positions = self.state[:, :2]
        velocities = self.state[:, 2:4]

        for cell_y in range(num_cells_y):
            for cell_x in range(num_cells_x):
                cell_center_x = (
                    grid_config.origin_x +
                    (cell_x + 0.5) * grid_config.grid_resolution
                )
                cell_center_y = (
                    grid_config.origin_y +
                    (cell_y + 0.5) * grid_config.grid_resolution
                )

                half_res = grid_config.grid_resolution / 2
                in_cell = (
                    (positions[:, 0] >= cell_center_x - half_res) &
                    (positions[:, 0] < cell_center_x + half_res) &
                    (positions[:, 1] >= cell_center_y - half_res) &
                    (positions[:, 1] < cell_center_y + half_res)
                )

                count = np.sum(in_cell)
                flow_field[cell_y, cell_x, 0] = count

                if count > 0:
                    flow_field[cell_y, cell_x, 1] = np.mean(velocities[in_cell, 0])
                    flow_field[cell_y, cell_x, 2] = np.mean(velocities[in_cell, 1])

        return flow_field

    def add_person(self, x: float, y: float, vx: float, vy: float) -> Person:
        """Add a single person at the specified position."""
        person = self._add_person(x, y, vx, vy)
        self._update_state()
        return person

    def remove_person(self, person_id: int) -> bool:
        """Remove a person by ID."""
        self.people = [p for p in self.people if p.id != person_id]
        self._update_state()
        return person_id not in [p.id for p in self.people]

    def reset(self) -> None:
        """Reset the simulator to initial state."""
        self.people = []
        self.state = None
        self.next_id = 0
        self.time = 0.0
        self.step_count = 0

    def get_stats(self) -> Dict:
        """Get current simulation statistics."""
        if self.state is None or len(self.people) == 0:
            return {
                "num_people": 0,
                "avg_speed": 0.0,
                "avg_velocity": [0.0, 0.0],
                "time": self.time,
                "step": self.step_count
            }

        velocities = self.state[:, 2:4]
        speeds = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)

        return {
            "num_people": len(self.people),
            "avg_speed": float(np.mean(speeds)),
            "avg_velocity": [float(np.mean(velocities[:, 0])),
                           float(np.mean(velocities[:, 1]))],
            "time": self.time,
            "step": self.step_count
        }
