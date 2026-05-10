"""
Robot Scanner Module.

Simulates a robot sensor that scans the environment using ray casting
to detect people within its field of view.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .config import ScannerConfig


@dataclass
class ScanResult:
    """Represents a single scan detection result."""
    target_id: int
    distance: float
    angle: float  # Angle relative to robot heading (degrees)
    relative_velocity: Tuple[float, float]
    x: float  # Detected x position
    y: float  # Detected y position

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "target_id": self.target_id,
            "distance": float(self.distance),
            "angle": float(self.angle),
            "relative_velocity": [float(v) for v in self.relative_velocity],
            "x": float(self.x),
            "y": float(self.y)
        }


class RobotScanner:
    """
    Simulates a robot-mounted sensor using ray casting.

    The scanner emits rays within its field of view and detects
    people within range, optionally with measurement noise.

    Attributes:
        config: Scanner configuration parameters
        position: Current (x, y) position of the scanner
        heading: Current heading angle in radians
    """

    def __init__(self, config: ScannerConfig):
        """
        Initialize the robot scanner.

        Args:
            config: ScannerConfig containing scanner parameters
        """
        self.config = config
        self.position: np.ndarray = np.zeros(2)
        self.heading: float = 0.0
        self.robot_velocity: np.ndarray = np.zeros(2)
        self._visible_target_ids: List[int] = []
        self._last_scan_result: List[ScanResult] = []

    def set_pose(self, x: float, y: float, heading: float) -> None:
        """
        Set the scanner pose.

        Args:
            x: X position
            y: Y position
            heading: Heading angle in radians
        """
        self.position = np.array([x, y])
        self.heading = heading

    def set_velocity(self, vx: float, vy: float) -> None:
        """
        Set the robot's velocity for relative velocity calculation.

        Args:
            vx: Velocity in X direction
            vy: Velocity in Y direction
        """
        self.robot_velocity = np.array([vx, vy])

    def scan(self, people_state: List[Dict]) -> List[Dict]:
        """
        Perform a scan of the environment.

        Args:
            people_state: List of person states from the simulator

        Returns:
            List of scan results for detected people
        """
        self._visible_target_ids = []
        self._last_scan_result = []

        if not people_state:
            return []

        detections = []

        for person in people_state:
            detection = self._check_detection(person)
            if detection is not None:
                detections.append(detection)
                self._visible_target_ids.append(person["id"])

        self._last_scan_result = detections
        return [d.to_dict() for d in detections]

    def _check_detection(self, person: Dict) -> Optional[ScanResult]:
        """
        Check if a person is detectable by the scanner.

        Args:
            person: Person state dictionary

        Returns:
            ScanResult if detected, None otherwise
        """
        person_pos = np.array([person["x"], person["y"]])
        person_vel = np.array([person["vx"], person["vy"]])

        dx = person_pos[0] - self.position[0]
        dy = person_pos[1] - self.position[1]
        distance = np.sqrt(dx * dx + dy * dy)

        if distance < self.config.min_detectable_distance:
            return None

        if distance > self.config.scan_range:
            return None

        angle_to_target = np.arctan2(dy, dx)
        relative_angle = self._normalize_angle(angle_to_target - self.heading)

        half_fov = self.config.fov / 2
        if abs(relative_angle) > half_fov:
            return None

        detected_distance = distance
        if self.config.noise_std_distance > 0:
            detected_distance += np.random.normal(0, self.config.noise_std_distance)
            detected_distance = max(0.1, detected_distance)

        detected_angle_deg = np.degrees(relative_angle)
        if self.config.noise_std_angle > 0:
            detected_angle_deg += np.random.normal(0, self.config.noise_std_angle)

        relative_vel = person_vel - self.robot_velocity

        if self.config.noise_std_velocity > 0:
            relative_vel = relative_vel + np.random.normal(
                0, self.config.noise_std_velocity, 2
            )

        return ScanResult(
            target_id=person["id"],
            distance=detected_distance,
            angle=detected_angle_deg,
            relative_velocity=(float(relative_vel[0]), float(relative_vel[1])),
            x=float(person_pos[0]),
            y=float(person_pos[1])
        )

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def get_visible_targets(self) -> List[int]:
        """Get IDs of currently visible targets."""
        return self._visible_target_ids.copy()

    def get_last_scan(self) -> List[Dict]:
        """Get the last scan result as dictionaries."""
        return [d.to_dict() for d in self._last_scan_result]

    def get_scan_rays(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get the ray geometry for visualization.

        Returns:
            List of (origin, direction) tuples for each ray
        """
        num_rays = self.config.num_rays
        half_fov = self.config.fov / 2

        if num_rays == 1:
            angles = [self.heading]
        else:
            angles = np.linspace(
                self.heading - half_fov,
                self.heading + half_fov,
                num_rays
            )

        rays = []
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            rays.append((self.position.copy(), direction))

        return rays

    def get_fov_boundaries(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the field of view boundary lines for visualization.

        Returns:
            Tuple of (left_boundary, right_boundary, arc_points)
        """
        half_fov = self.config.fov / 2

        left_angle = self.heading - half_fov
        right_angle = self.heading + half_fov

        left_boundary = self.position + self.config.scan_range * np.array([
            np.cos(left_angle), np.sin(left_angle)
        ])
        right_boundary = self.position + self.config.scan_range * np.array([
            np.cos(right_angle), np.sin(right_angle)
        ])

        num_arc_points = 20
        arc_angles = np.linspace(left_angle, right_angle, num_arc_points)
        arc_points = np.column_stack([
            self.position[0] + self.config.scan_range * np.cos(arc_angles),
            self.position[1] + self.config.scan_range * np.sin(arc_angles)
        ])

        return left_boundary, right_boundary, arc_points

    def cast_ray(
        self,
        angle: float,
        people_state: List[Dict]
    ) -> Optional[Dict]:
        """
        Cast a single ray at the specified angle and find the closest hit.

        Args:
            angle: Ray angle in radians (absolute, not relative to heading)
            people_state: List of person states

        Returns:
            Detection result for the closest person, or None
        """
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        closest_hit = None
        closest_dist = self.config.scan_range

        for person in people_state:
            person_pos = np.array([person["x"], person["y"]])

            hit = self._ray_circle_intersection(
                self.position, ray_dir, person_pos, person.get("radius", 0.3)
            )

            if hit is not None and hit < closest_dist:
                closest_dist = hit
                person_vel = np.array([person["vx"], person["vy"]])
                relative_vel = person_vel - self.robot_velocity

                rel_angle = self._normalize_angle(angle - self.heading)
                rel_angle_deg = np.degrees(rel_angle)

                closest_hit = {
                    "target_id": person["id"],
                    "distance": hit,
                    "angle": rel_angle_deg,
                    "relative_velocity": relative_vel.tolist(),
                    "x": float(person_pos[0]),
                    "y": float(person_pos[1])
                }

        return closest_hit

    def _ray_circle_intersection(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        circle_center: np.ndarray,
        circle_radius: float
    ) -> Optional[float]:
        """
        Calculate intersection between a ray and a circle.

        Args:
            ray_origin: Origin point of the ray
            ray_dir: Direction vector of the ray (normalized)
            circle_center: Center of the circle
            circle_radius: Radius of the circle

        Returns:
            Distance to intersection, or None if no intersection
        """
        oc = ray_origin - circle_center

        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - circle_radius * circle_radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        t = (-b - np.sqrt(discriminant)) / (2 * a)

        if t > 0:
            return float(t)

        t = (-b + np.sqrt(discriminant)) / (2 * a)
        if t > 0:
            return float(t)

        return None

    def is_in_fov(self, x: float, y: float) -> bool:
        """
        Check if a point is within the field of view.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if the point is within FOV
        """
        dx = x - self.position[0]
        dy = y - self.position[1]
        distance = np.sqrt(dx * dx + dy * dy)

        if distance > self.config.scan_range:
            return False

        angle_to_point = np.arctan2(dy, dx)
        relative_angle = abs(self._normalize_angle(angle_to_point - self.heading))

        return relative_angle <= self.config.fov / 2

    def get_detection_probability(
        self,
        distance: float,
        occlusion_factor: float = 1.0
    ) -> float:
        """
        Calculate probability of detection based on distance.

        Args:
            distance: Distance to target
            occlusion_factor: Factor reducing detection probability (0-1)

        Returns:
            Detection probability (0-1)
        """
        if distance > self.config.scan_range:
            return 0.0

        base_prob = 1.0 - (distance / self.config.scan_range) ** 2

        return base_prob * occlusion_factor
