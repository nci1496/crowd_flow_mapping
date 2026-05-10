# Crowd Flow Mapping

A simulation system for robot-based crowd flow field reconstruction in 2D environments.

## Overview

This project simulates a robot scanning a crowd environment and reconstructing the local flow field based on sensor measurements. It includes:

- **Crowd Simulation**: Realistic crowd movement with person-person interactions
- **Robot Sensing**: Ray-casting based sensor simulation with configurable noise
- **Flow Reconstruction**: Grid-based flow field mapping using various interpolation methods
- **Error Analysis**: Comprehensive error metrics and visualization
- **Visualization**: Static plots and animated visualizations

## Project Structure

```
crowd_flow_mapping/
├── data/               # Data storage directory
├── results/            # Output results directory
├── src/
│   ├── config.py           # Configuration management
│   ├── people_simulator.py  # Crowd movement simulation
│   ├── robot_scanner.py     # Robot sensor simulation
│   ├── flow_mapper.py       # Flow field reconstruction
│   ├── flow_analyzer.py     # Error computation
│   └── visualize.py         # Visualization engine
├── main.py             # Main entry point
├── requirements.txt   # Python dependencies
└── README.md
```

## Installation

```bash
# Create conda environment
conda create -n crowd_flow python=3.8
conda activate crowd_flow

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from main import SimulationRunner, create_test_config

# Create runner with test configuration
config = create_test_config()
runner = SimulationRunner(config)
runner.setup()

# Run simulation
results = runner.run(num_steps=100, verbose=True)

# View results
runner.plot_final_state()
```

## Configuration

All parameters are centralized in `src/config.py`. Use `GlobalConfig` to customize:

```python
from src.config import GlobalConfig

config = GlobalConfig()
config.simulator.num_people = 100
config.simulator.width = 30.0
config.scanner.scan_range = 15.0
config.mapper.update_method = "gaussian"
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `simulator.num_people` | 50 | Number of people in simulation |
| `simulator.width/height` | 20m | Environment dimensions |
| `scanner.scan_range` | 10m | Robot sensor range |
| `scanner.fov` | 120deg | Robot field of view |
| `mapper.update_method` | gaussian | Interpolation method |
| `simulation.timestep` | 0.1s | Simulation time step |

## Module Details

### People Simulator (`src/people_simulator.py`)

Simulates crowd movement with:
- Random velocity generation
- Boundary collision handling (reflect/bounce/wrap)
- Person-person repulsion forces
- Ground truth flow field computation

### Robot Scanner (`src/robot_scanner.py`)

Simulates robot sensor with:
- Ray-casting detection
- Configurable field of view
- Measurement noise injection
- Relative velocity estimation

### Flow Mapper (`src/flow_mapper.py`)

Reconstructs flow field using:
- **Nearest**: Direct cell assignment
- **Gaussian**: Kernel-weighted interpolation
- **Linear**: Bilinear interpolation
- Temporal smoothing

### Flow Analyzer (`src/flow_analyzer.py`)

Computes error metrics:
- MSE, MAE, RMSE for velocity and density
- Direction error
- Combined scoring
- Error heatmaps

### Visualizer (`src/visualize.py`)

Provides visualization:
- Quiver plots for flow fields
- Error heatmaps
- Animation generation
- Dashboard views

## ROS/Gazebo Integration

The modular architecture supports future ROS integration:
- Each module can be wrapped as a ROS node
- Topic interfaces for data exchange
- Service calls for configuration

Planned migration to C++ for core computation will maintain the same interface structure.

## Examples

### Basic Simulation

```python
from main import SimulationRunner, create_test_config
import matplotlib.pyplot as plt

config = create_test_config()
runner = SimulationRunner(config)
runner.setup()
results = runner.run(num_steps=50)
runner.plot_final_state()
plt.show()
```

### Animation Export

```python
animation = runner.create_animation()
runner.visualizer.save_animation(animation, "simulation.gif")
```

### Custom Configuration

```python
from src.config import GlobalConfig
from main import SimulationRunner

config = GlobalConfig()
config.simulator.num_people = 200
config.simulator.width = 40.0
config.simulator.height = 40.0
config.scanner.scan_range = 15.0
config.robot.patrol_mode = "grid"

runner = SimulationRunner(config)
runner.run(num_steps=200)
```

## License

MIT License
