# FSAE Self-Driving Sim

CARLA experiments for Formula SAE Driverless track generation, cone perception,
path planning, and vehicle control. The Python package is named `sim`.

## Experiments

| Command | Input and behavior |
| --- | --- |
| `python -m sim.run.preview --debug-road` | Generate an offline track preview and mesh. |
| `python -m sim.run.pathplanning` | Follow the generated, known centerline using a sampling lateral MPC and longitudinal PID. Draw Delaunay midpoint candidates for inspection. |
| `python -m sim.run.sensor_pathplanning` | Build cone landmarks from camera/LiDAR, plan with Delaunay, and apply MPC/PID control. Show separate camera and LiDAR windows. |

Driving experiments connect to `127.0.0.1:2000` and replace the current CARLA
world with a generated blank world. Run one driving experiment at a time.
Press `Ctrl+C` in its terminal to stop.

## Python Setup

Use Python 3.11 x64. From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Install the CARLA Python API from the same build as your server. For the Windows
source build described in [CARLA Windows Setup](CARLA_WINDOWS_SETUP.md):

```powershell
.\.venv\Scripts\python.exe -m pip install .\carla\PythonAPI\carla\dist\carla-0.9.16-cp311-cp311-win_amd64.whl
```

For an official CARLA 0.9.16 server with a matching published Python wheel, use
`python -m pip install -e ".[carla]"` in your environment instead.
`requirements.txt` delegates to the same package configuration.

Use `.\.venv\Scripts\python.exe` in place of `python` below, or activate the
environment with `.\.venv\Scripts\Activate.ps1`.
The viewer requires GUI-enabled `opencv-python`; the project pins version
4.10.0.84 with NumPy 1.x. Keep packages requiring a headless OpenCV distribution
in a separate environment.

Editable installation registers `sim` with the selected interpreter. After
installation, `python -m sim.run.pathplanning` works from any directory, and
source edits take effect without reinstalling. Use module commands for the
experiment entry points.

## CARLA Setup

Build CARLA and install the project cone assets using
[CARLA Windows Setup](CARLA_WINDOWS_SETUP.md). The current build uses
CARLA 0.9.16 and CARLA's Unreal Engine 4.26.2 fork.

From the repository root, start the built server:

```powershell
$carlaProject = (Resolve-Path .\carla\Unreal\CarlaUE4\CarlaUE4.uproject).Path
& .\UE4\Engine\Binaries\Win64\UE4Editor.exe $carlaProject -game -windowed -ResX=1280 -ResY=720 -quality-level=Low -carla-rpc-port=2000
```

Then run one experiment from another terminal:

```powershell
.\.venv\Scripts\python.exe -m sim.run.pathplanning
```

Or start the camera/LiDAR experiment:

```powershell
.\.venv\Scripts\python.exe -m sim.run.sensor_pathplanning
```

## Offline Preview

The track preview requires Python dependencies but no CARLA server or Python API:

```powershell
python -m sim.run.preview --debug-road --mission autocross
python -m sim.run.preview --debug-road --mission acceleration --track-name acceleration
python -m sim.run.preview --debug-road --mission skidpad --track-name skidpad
```

PNG previews and OBJ meshes are written to `data/debug/` relative to the working
directory. Use `--output-dir` to choose another location and `--help` to list
generation settings.

## Code Layout

```text
sim/
  run/           Runnable experiment modules
  env/           CARLA world and sensor wrappers
  roadgen/       Cone tracks, geometry, meshes, and CARLA spawning
  perception/    Cone fusion and lane detection interfaces
  planning/      Delaunay planning and waypoint helpers
  eval/          Evaluation scaffolding
  config/        Scenario configuration examples
  assets/cones/  Source cone models, textures, and UE4 assets
  setup/         CARLA Windows build compatibility patch
doc/
  papers/        Research references and original source links
```

Add runnable experiments as `sim/run/<name>.py` with a `main()` function and
an `if __name__ == "__main__": main()` guard. Run with `python -m sim.run.<name>`
and use package imports such as `from sim.roadgen import TrackGenerator`.
Automated assertions belong in a separate `tests/` directory when added.

## Current Scope

- Track generation supports acceleration, skidpad, and autocross layouts.
- The known-path experiment uses the generator's centerline as ground truth.
- Sensor perception uses LiDAR height filtering and Euclidean clustering, camera
  projection, HSV color classification, and a confidence-filtered online map.
  Vehicle pose comes from CARLA; cone positions are inferred from sensors.
- The sensor pipeline requires detection/projection tuning. It is experimental,
  not a validated autonomous lap-completion system.
- MPC samples steering values using a kinematic bicycle rollout. Speed targets
  account for curvature and acceleration/braking limits. Cone clearance is a
  soft cost, not a collision-free guarantee.
- `sim/eval` and `sim/config/scenarios.yaml` are scaffolding; driving demos do
  not yet load that YAML or compute full competition metrics.

See [reference papers](doc/papers/README.md) and the
[2026 Driverless rules](doc/FSAE_Driverless_2026_V1.pdf) for background.

## Repository Contents

Git tracks project code, configuration, documentation, source cone models,
project UE4 cone assets, and the CARLA build patch. `carla/` and `UE4/` are
external checkouts restored using the build guide. Local datasets, logs,
virtual environments, generated outputs, and downloaded paper PDFs are ignored.

Use upstream downloads for CARLA and authenticated access to the CARLA Unreal
Engine fork. Compiled redistributable simulator builds can be distributed
separately after checking their licenses. GitHub
[Release assets](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
must each be smaller than 2 GiB; regular Git blocks
[files larger than 100 MiB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Unreal Engine source and editor distribution is governed by the
[Unreal Engine license](https://www.unrealengine.com/eula/unreal).
