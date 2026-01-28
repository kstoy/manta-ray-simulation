# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physics-based surface simulation system that models balls rolling on a fabric surface suspended by a grid of controllable rods. Uses catenary curves to model fabric between rod attachment points and XPBD (Extended Position Based Dynamics) for physics simulation with friction.

**Purpose**: Explore surface control strategies (weight-based, bang-bang, square patterns, cosine waves) to direct ball movement across a fabric surface.

## Running the Code

```bash
python run_simulation.py              # Run simulation with visualization
python scripts/performancetest.py     # Benchmark vs ball count
python scripts/profilesimulation.py   # Profile performance bottlenecks
```

## Project Structure

```
surface_sim/
├── src/                    # Source code package
│   ├── simulation.py       # Main simulation logic
│   ├── ballstate.py        # Ball positions, velocities, masses, radii
│   ├── rodstate.py         # Rod grid management, sensor data
│   ├── config.py           # SimConfig dataclass
│   ├── constants.py        # Direction constants (NE, NW, SW, SE)
│   ├── visualization.py    # GLTF export for 3D visualization
│   ├── physics/            # Physics modules
│   │   ├── simcorexpbd.py  # XPBD physics engine
│   │   ├── catenary.py     # Catenary curve math
│   │   └── catenarysurface.py  # Surface interpolation
│   └── controllers/        # Control strategies
│       ├── controller_base.py
│       ├── mass_sort_controller.py
│       ├── squarecontroller.py
│       └── squarecontroller_nonedeterministic_*.py
├── scripts/                # Entry point scripts
├── data/                   # Input data files
├── output/                 # Generated outputs (gitignored)
└── run_simulation.py       # Main entry point
```

## Architecture

### Data Flow
```
Controller → RodsState → Catenary Surface → BallsState → XPBD Physics → Visualization
```

### Main Simulation Loop (per timestep)
1. Sensor collection: aggregate ball weights onto rod corners
2. Control update: run controller logic for desired rod heights
3. Rod adjustment: P-control `rod_height += K * (desired - current)`
4. Physics substeps (2 per timestep): predict, detect collisions, XPBD corrections, friction
5. State recording for visualization

## Key Parameters (src/config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `D` | 1.0m | Rod spacing |
| `LF` | 1.45 | Fabric length factor (sag) |
| `GRIDSIZEX` / `GRIDSIZEY` | 20 / 4 | Rod grid dimensions |
| `DT` | 0.1s | Physics timestep |
| `MAXSIMULATIONSTEPS` | 200 | Simulation length (20 seconds) |
| `K` | 0.2 | Rod height P-control gain |
| `TARGET_WEIGHT` | 0.04 | Weight threshold for control |

## Extension Points

**New Controller**: Implement class in `src/controllers/` with `__init__(config)` and `update(i, j, timestep, sensors)` method returning desired rod height.

**Physics**: Modify `src/physics/simcorexpbd.py` for different constraint solvers or friction models.

## Dependencies

- `numpy` - Linear algebra
- `scipy` - Optimization
- `pygltflib` - GLTF export
