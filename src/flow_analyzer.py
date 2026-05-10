"""
Flow Analyzer Module.

Analyzes and compares ground truth flow fields with estimated/reconstructed
flow fields, computing various error metrics and generating reports.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

from .config import AnalyzerConfig


class FlowAnalyzer:
    """
    Analyzes errors between ground truth and estimated flow fields.

    Computes various error metrics including MSE, MAE, and generates
    error reports and heatmaps.

    Attributes:
        config: Analyzer configuration parameters
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """
        Initialize the flow analyzer.

        Args:
            config: Analyzer configuration (uses default if None)
        """
        self.config = config or AnalyzerConfig()
        self._error_history: List[Dict] = []
        self._error_heatmap_history: List[np.ndarray] = []
        self._max_history = 100

    def compute_errors(
        self,
        ground_truth: np.ndarray,
        estimated: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute errors between ground truth and estimated flow fields.

        Args:
            ground_truth: Ground truth flow field array (H, W, 3)
            estimated: Estimated flow field array (H, W, 3)
            mask: Optional mask to exclude certain cells from analysis

        Returns:
            Dictionary containing various error metrics
        """
        if ground_truth.shape != estimated.shape:
            raise ValueError(
                f"Shape mismatch: ground_truth {ground_truth.shape} vs "
                f"estimated {estimated.shape}"
            )

        gt_density = ground_truth[:, :, 0]
        gt_vx = ground_truth[:, :, 1]
        gt_vy = ground_truth[:, :, 2]

        est_density = estimated[:, :, 0]
        est_vx = estimated[:, :, 1]
        est_vy = estimated[:, :, 2]

        if mask is None:
            mask = np.ones_like(gt_density, dtype=bool)
        else:
            mask = mask.astype(bool)

        gt_magnitude = np.sqrt(gt_vx**2 + gt_vy**2)
        est_magnitude = np.sqrt(est_vx**2 + est_vy**2)

        velocity_error = self.compute_velocity_error(
            ground_truth, estimated, mask
        )
        density_error = self.compute_density_error(
            ground_truth, estimated, mask
        )

        gt_mag_flat = gt_magnitude[mask]
        est_mag_flat = est_magnitude[mask]

        if len(gt_mag_flat) > 0:
            magnitude_mse = float(np.mean((gt_mag_flat - est_mag_flat) ** 2))
            magnitude_mae = float(np.mean(np.abs(gt_mag_flat - est_mag_flat)))
        else:
            magnitude_mse = 0.0
            magnitude_mae = 0.0

        direction_error = self._compute_direction_error(
            gt_vx, gt_vy, est_vx, est_vy, mask
        )

        combined_score = (
            self.config.velocity_weight * velocity_error.get("mse", 0) +
            self.config.density_weight * density_error.get("mse", 0)
        )

        errors = {
            "velocity_mse": velocity_error.get("mse", 0),
            "velocity_mae": velocity_error.get("mae", 0),
            "velocity_rmse": velocity_error.get("rmse", 0),
            "density_mse": density_error.get("mse", 0),
            "density_mae": density_error.get("mae", 0),
            "density_rmse": density_error.get("rmse", 0),
            "magnitude_mse": magnitude_mse,
            "magnitude_mae": magnitude_mae,
            "direction_error": direction_error,
            "combined_score": combined_score,
            "timestamp": datetime.now().isoformat(),
            "num_evaluated_cells": int(np.sum(mask))
        }

        self._error_history.append(errors)
        if len(self._error_history) > self._max_history:
            self._error_history.pop(0)

        return errors

    def compute_velocity_error(
        self,
        ground_truth: np.ndarray,
        estimated: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute velocity-specific errors.

        Args:
            ground_truth: Ground truth flow field
            estimated: Estimated flow field
            mask: Optional mask

        Returns:
            Dictionary with velocity error metrics
        """
        gt_vx = ground_truth[:, :, 1].flatten()
        gt_vy = ground_truth[:, :, 2].flatten()

        est_vx = estimated[:, :, 1].flatten()
        est_vy = estimated[:, :, 2].flatten()

        if mask is not None:
            mask_flat = mask.flatten()
            gt_vx = gt_vx[mask_flat]
            gt_vy = gt_vy[mask_flat]
            est_vx = est_vx[mask_flat]
            est_vy = est_vy[mask_flat]
        else:
            mask_flat = np.ones(len(gt_vx), dtype=bool)

        if len(gt_vx) == 0:
            return {"mse": 0.0, "mae": 0.0, "rmse": 0.0}

        if self.config.error_metric == "mse":
            mse_vx = np.mean((gt_vx - est_vx) ** 2)
            mse_vy = np.mean((gt_vy - est_vy) ** 2)
            mse = (mse_vx + mse_vy) / 2

            mae_vx = np.mean(np.abs(gt_vx - est_vx))
            mae_vy = np.mean(np.abs(gt_vy - est_vy))
            mae = (mae_vx + mae_vy) / 2

            rmse = np.sqrt(mse)

        elif self.config.error_metric == "mae":
            mae_vx = np.mean(np.abs(gt_vx - est_vx))
            mae_vy = np.mean(np.abs(gt_vy - est_vy))
            mae = (mae_vx + mae_vy) / 2

            mse_vx = np.mean((gt_vx - est_vx) ** 2)
            mse_vy = np.mean((gt_vy - est_vy) ** 2)
            mse = (mse_vx + mse_vy) / 2
            rmse = np.sqrt(mse)

        else:
            mse_vx = np.mean((gt_vx - est_vx) ** 2)
            mse_vy = np.mean((gt_vy - est_vy) ** 2)
            mse = (mse_vx + mse_vy) / 2
            mae_vx = np.mean(np.abs(gt_vx - est_vx))
            mae_vy = np.mean(np.abs(gt_vy - est_vy))
            mae = (mae_vx + mae_vy) / 2
            rmse = np.sqrt(mse)

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse)
        }

    def compute_density_error(
        self,
        ground_truth: np.ndarray,
        estimated: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute density-specific errors.

        Args:
            ground_truth: Ground truth flow field
            estimated: Estimated flow field
            mask: Optional mask

        Returns:
            Dictionary with density error metrics
        """
        gt_density = ground_truth[:, :, 0].flatten()
        est_density = estimated[:, :, 0].flatten()

        if mask is not None:
            mask_flat = mask.flatten()
            gt_density = gt_density[mask_flat]
            est_density = est_density[mask_flat]

        if len(gt_density) == 0:
            return {"mse": 0.0, "mae": 0.0, "rmse": 0.0}

        if self.config.error_metric in ("mse", "rmse"):
            mse = np.mean((gt_density - est_density) ** 2)
            rmse = np.sqrt(mse)
        else:
            mse = np.mean((gt_density - est_density) ** 2)
            rmse = np.sqrt(mse)

        mae = np.mean(np.abs(gt_density - est_density))

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse)
        }

    def _compute_direction_error(
        self,
        gt_vx: np.ndarray,
        gt_vy: np.ndarray,
        est_vx: np.ndarray,
        est_vy: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute angular direction error between velocity vectors.

        Args:
            gt_vx: Ground truth velocity X component
            gt_vy: Ground truth velocity Y component
            est_vx: Estimated velocity X component
            est_vy: Estimated velocity Y component
            mask: Optional mask

        Returns:
            Mean angular error in degrees
        """
        gt_vx_flat = gt_vx.flatten()
        gt_vy_flat = gt_vy.flatten()
        est_vx_flat = est_vx.flatten()
        est_vy_flat = est_vy.flatten()

        if mask is not None:
            mask_flat = mask.flatten()
            gt_vx_flat = gt_vx_flat[mask_flat]
            gt_vy_flat = gt_vy_flat[mask_flat]
            est_vx_flat = est_vx_flat[mask_flat]
            est_vy_flat = est_vy_flat[mask_flat]

        gt_angle = np.arctan2(gt_vy_flat, gt_vx_flat)
        est_angle = np.arctan2(est_vy_flat, est_vx_flat)

        angle_diff = np.abs(gt_angle - est_angle)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

        mean_angle_error = np.mean(angle_diff)
        return float(np.degrees(mean_angle_error))

    def get_error_heatmap(
        self,
        ground_truth: np.ndarray,
        estimated: np.ndarray,
        mode: str = "combined"
    ) -> np.ndarray:
        """
        Generate an error heatmap.

        Args:
            ground_truth: Ground truth flow field
            estimated: Estimated flow field
            mode: Type of error - "velocity", "density", or "combined"

        Returns:
            Error heatmap array of shape (H, W)
        """
        if mode == "velocity":
            gt_vx = ground_truth[:, :, 1]
            gt_vy = ground_truth[:, :, 2]
            est_vx = estimated[:, :, 1]
            est_vy = estimated[:, :, 2]

            error = np.sqrt((gt_vx - est_vx)**2 + (gt_vy - est_vy)**2)

        elif mode == "density":
            gt_density = ground_truth[:, :, 0]
            est_density = estimated[:, :, 0]
            error = np.abs(gt_density - est_density)

        else:
            gt_vx = ground_truth[:, :, 1]
            gt_vy = ground_truth[:, :, 2]
            est_vx = estimated[:, :, 1]
            est_vy = estimated[:, :, 2]
            gt_density = ground_truth[:, :, 0]
            est_density = estimated[:, :, 0]

            vel_error = np.sqrt((gt_vx - est_vx)**2 + (gt_vy - est_vy)**2)
            dens_error = np.abs(gt_density - est_density)

            vel_max = np.max(vel_error) if np.max(vel_error) > 0 else 1
            dens_max = np.max(dens_error) if np.max(dens_error) > 0 else 1

            vel_error_norm = vel_error / vel_max
            dens_error_norm = dens_error / dens_max

            error = (
                self.config.velocity_weight * vel_error_norm +
                self.config.density_weight * dens_error_norm
            )

        self._error_heatmap_history.append(error.copy())
        if len(self._error_heatmap_history) > self._max_history:
            self._error_heatmap_history.pop(0)

        return error

    def generate_error_report(
        self,
        errors: Optional[Dict] = None
    ) -> str:
        """
        Generate a formatted error report.

        Args:
            errors: Optional pre-computed errors dictionary

        Returns:
            Formatted error report string
        """
        if errors is None:
            if not self._error_history:
                return "No error data available."
            errors = self._error_history[-1]

        if self.config.report_format == "json":
            return json.dumps(errors, indent=2)
        else:
            report_lines = [
                "=" * 50,
                "Flow Field Error Report",
                "=" * 50,
                f"Timestamp: {errors.get('timestamp', 'N/A')}",
                f"Evaluated Cells: {errors.get('num_evaluated_cells', 'N/A')}",
                "-" * 50,
                "Velocity Errors:",
                f"  MSE:  {errors.get('velocity_mse', 0):.6f}",
                f"  MAE:  {errors.get('velocity_mae', 0):.6f}",
                f"  RMSE: {errors.get('velocity_rmse', 0):.6f}",
                "-" * 50,
                "Density Errors:",
                f"  MSE:  {errors.get('density_mse', 0):.6f}",
                f"  MAE:  {errors.get('density_mae', 0):.6f}",
                f"  RMSE: {errors.get('density_rmse', 0):.6f}",
                "-" * 50,
                "Magnitude Errors:",
                f"  MSE:  {errors.get('magnitude_mse', 0):.6f}",
                f"  MAE:  {errors.get('magnitude_mae', 0):.6f}",
                "-" * 50,
                f"Direction Error: {errors.get('direction_error', 0):.2f} deg",
                "-" * 50,
                f"Combined Score: {errors.get('combined_score', 0):.6f}",
                "=" * 50
            ]
            return "\n".join(report_lines)

    def get_error_summary(self) -> Dict:
        """
        Get summary statistics across all error history.

        Returns:
            Dictionary with summary statistics
        """
        if not self._error_history:
            return {"message": "No error history available"}

        velocity_mse = [e["velocity_mse"] for e in self._error_history]
        density_mse = [e["density_mse"] for e in self._error_history]
        combined = [e["combined_score"] for e in self._error_history]

        return {
            "num_samples": len(self._error_history),
            "velocity_mse": {
                "mean": float(np.mean(velocity_mse)),
                "std": float(np.std(velocity_mse)),
                "min": float(np.min(velocity_mse)),
                "max": float(np.max(velocity_mse))
            },
            "density_mse": {
                "mean": float(np.mean(density_mse)),
                "std": float(np.std(density_mse)),
                "min": float(np.min(density_mse)),
                "max": float(np.max(density_mse))
            },
            "combined_score": {
                "mean": float(np.mean(combined)),
                "std": float(np.std(combined)),
                "min": float(np.min(combined)),
                "max": float(np.max(combined))
            }
        }

    def get_cumulative_error(self) -> np.ndarray:
        """
        Get cumulative error over time.

        Returns:
            Cumulative error array
        """
        if not self._error_history:
            return np.array([])

        combined_scores = [e["combined_score"] for e in self._error_history]
        return np.cumsum(combined_scores)

    def get_convergence_rate(self) -> Optional[float]:
        """
        Estimate convergence rate based on error history.

        Returns:
            Convergence rate estimate, or None if insufficient data
        """
        if len(self._error_history) < 10:
            return None

        first_half = np.mean([
            e["combined_score"] for e in self._error_history[:len(self._error_history)//2]
        ])
        second_half = np.mean([
            e["combined_score"] for e in self._error_history[len(self._error_history)//2:]
        ])

        if first_half == 0:
            return None

        convergence = (first_half - second_half) / first_half
        return float(convergence)

    def reset(self) -> None:
        """Reset the analyzer state."""
        self._error_history.clear()
        self._error_heatmap_history.clear()

    def export_errors(self, filepath: str) -> None:
        """
        Export error history to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        with open(filepath, 'w') as f:
            json.dump({
                "history": self._error_history,
                "summary": self.get_error_summary()
            }, f, indent=2)

    def get_peaks_and_valleys(self) -> Dict:
        """
        Identify peak and valley points in the error history.

        Returns:
            Dictionary with indices and values of peaks and valleys
        """
        if len(self._error_history) < 3:
            return {"peaks": [], "valleys": []}

        combined = np.array([e["combined_score"] for e in self._error_history])

        peaks = []
        valleys = []

        for i in range(1, len(combined) - 1):
            if combined[i] > combined[i-1] and combined[i] > combined[i+1]:
                peaks.append({"index": i, "value": float(combined[i])})
            elif combined[i] < combined[i-1] and combined[i] < combined[i+1]:
                valleys.append({"index": i, "value": float(combined[i])})

        return {"peaks": peaks, "valleys": valleys}
