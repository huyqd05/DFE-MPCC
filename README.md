# DFE-MPCC AMR — Slosh- & Friction-Aware Model Predictive Contouring Control

Codebase for **DFE-MPCC** (Dynamic Friction- and Slosh-Aware Model Predictive Contouring Control) on a 4-wheeled mecanum Autonomous Mobile Robot (AMR) transporting liquid cargo, combining:
- **CasADi NMPC / MPCC** with ISO 8608 road roughness disturbance, 12-facet wrench zonotope feasibility, FF-RLS friction observer, and a two-stage safety filter
- **MuJoCo co-simulation** (`slosh3.xml`, `world.xml`–`world8.xml`, dynamic obstacles, LiDAR + occupancy grid + A* + H124 velocity tube)
- **2D WCSPH / Navier–Stokes CFD** fluid-structure interaction (FSI) cross-validation and offline Split Conformal checkpoint calibration
- **ROS 2 Jazzy** Software-in-the-Loop (SiL) node and WSL bridge

## Repository Structure

```text
├── controller/     # DFE-MPCC & baseline MPCC-NMPC AMR controllers, traction dynamics,
│                   # mecanum nullspace allocation, ISO 8608 road profile, FF-RLS observer, safety filter
├── solver/         # CasADi AMR dynamic model (robot_model_amr_casadi.py) & wrench zonotope (zonotope_wrench_set.py)
├── planning/       # A* planner (astar.py), dynamic obstacle injector, velocity tube H124
├── cfd/            # 2D WCSPH / SPH FSI verification, damping estimation & conformal calibration
├── simulation/     # MuJoCo closed-loop runner, co-simulation bridge, 3D/telemetry GUI,
│                   # WP01/model-consistency audits & controlled case-study benchmarks
├── config/         # Conformal calibration (conformal_calibration_q_eta.json), WP01/WP02 configs, RViz2/PlotJuggler layouts
├── xml/            # MuJoCo AMR + liquid-slosh models (slosh3.xml, slosh_h40/h60/h80.xml, slosh_rect.XML, world*.xml)
├── tests/          # Physics, traction, friction-observer & conformal unit tests (test_physics_and_conformal.py)
├── dfempcc_node.py # ROS 2 node running DFEMPCCAMRController + AStarPlanner + mecanum allocator
├── cosim_bridge.py # Root re-export of simulation.cosim_bridge
├── *.sh / test_ros2_wsl.py  # WSL ROS 2 Jazzy build, setup & SiL test scripts
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux / WSL:        source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Verification & Usage

### 1. Run Unit Tests & Audits
```bash
python tests/test_physics_and_conformal.py
python simulation/test_wp01_compliance_suite.py
python simulation/test_model_consistency.py
python simulation/test_independent_mujoco_vs_casadi.py
```

### 2. Controlled Case-Study Benchmarks
```bash
python simulation/benchmark_dfempcc_t1t2_vs_sph_iso.py
python simulation/benchmark_world6_complex_4dyn_obs.py
python simulation/benchmark_world8_3way.py
python simulation/benchmark_ablation_5cases.py --map slosh3.xml --mu 0.12
python simulation/benchmark_nmpc_checkpoint_vs_dfempcc.py
```

### 3. Interactive MuJoCo 3D & Telemetry Visualization
```bash
python simulation/visualize_live_slosh_and_forces_gui.py --map world7.xml
python simulation/visualize_world5_dynamic_obs_gui.py
python simulation/visualize_world6_dynamic_obs_gui.py
```

### 4. WCSPH / Conformal Calibration
```bash
python cfd/run_sph_benchmark_suite.py
python cfd/sph_fsi_verification.py
python cfd/test_sph_fsi_cross_validation.py
python cfd/calibrate_conformal_q_eta.py
```

### 5. ROS 2 Jazzy SiL (WSL / Linux)
See `setup_wsl_ros2.sh`, `build_controller.sh`, `test_bridge.sh`, and `test_full_system.sh`.

> **Note:** Local reports (`bao cao/`), binary artifacts, scratch scripts, legacy 8-DoF prototypes, and Monte Carlo experiment archives (`_frozen_legacy_monte_carlo/`, 30-seed batch runners) are excluded via `.gitignore`.
