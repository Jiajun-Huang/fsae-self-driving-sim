# CARLA Windows Setup

This project uses the UE4 edition of CARLA. External source trees live beside
`sim/` and are excluded from this repository.

## Source Versions

| Component | Upstream | Revision |
| --- | --- | --- |
| CARLA 0.9.16 | [carla-simulator/carla](https://github.com/carla-simulator/carla), `ue4-dev` | `f06817e84189e59c3a81eb08c61a58e270d1d730` |
| UE4 4.26.2 | [CarlaUnreal/UnrealEngine](https://github.com/CarlaUnreal/UnrealEngine), `carla` | `e9d9e60c85f643e10eeb03f42f61554d18dcb30f` |
| CARLA content | CARLA `Util/ContentVersions.txt` | `20250912_2171890` |

Follow the [official Windows build guide](https://carla.readthedocs.io/en/0.9.16/build_windows/)
for prerequisites and asset installation. The modified engine requires a
GitHub account linked to Epic with the appropriate Unreal Engine access.

## Tools

- Windows x64 and Python 3.11 x64.
- Visual Studio 2022 C++ build tools; `.vsconfig` provides the component list.
- GNU Make 3.81, CMake 3.x, Git, 7-Zip, and Windows `tar`/`certutil`.
- NumPy 1.x for this Python API build; project dependencies are in `pyproject.toml`.

Use a short checkout path when building the engine. Commands below assume the
terminal is at the project root. Clone only when the destination folders do
not already exist.

```powershell
git clone --branch ue4-dev https://github.com/carla-simulator/carla.git carla
git -C carla checkout f06817e84189e59c3a81eb08c61a58e270d1d730
git clone --depth 1 --branch carla https://github.com/CarlaUnreal/UnrealEngine.git UE4
git -C UE4 fetch --depth 1 origin e9d9e60c85f643e10eeb03f42f61554d18dcb30f
git -C UE4 checkout e9d9e60c85f643e10eeb03f42f61554d18dcb30f
```

## Build Compatibility

[sim/setup/carla-windows.patch](sim/setup/carla-windows.patch) contains the Windows
build-script adjustments used here: quoted paths, archive extraction, Boost
checksum/extraction handling, and OSM2ODR/Xerces linking. Apply it once to the
pinned CARLA revision before building:

```powershell
git -C carla apply --check ../sim/setup/carla-windows.patch
git -C carla apply ../sim/setup/carla-windows.patch
```

An existing checkout may already contain these adjustments. This read-only check
succeeds when the patch is already applied:

```powershell
git -C carla apply --reverse --check ../sim/setup/carla-windows.patch
```

Select your local UE4 source build as the engine association for
`carla/Unreal/CarlaUE4/CarlaUE4.uproject` using the Unreal version selector.
For the OSM2ODR/StreetMap integration, enable the `StreetMap` plugin in that
project. The engine association is specific to each machine.

## Build

Open a Visual Studio x64 Native Tools command prompt. Ensure GNU Make 3.81,
CMake 3.x, and Python 3.11 are the executables found on `PATH`. From the
repository root in that command prompt:

```bat
set "UE4_ROOT=%CD%\UE4"
cd UE4
Setup.bat
GenerateProjectFiles.bat
Engine\Build\BatchFiles\Build.bat UE4Editor Win64 Development -WaitMutex
cd ..\carla
Update.bat
make setup
make PythonAPI
make launch
```

`Update.bat` obtains the content referenced by `Util/ContentVersions.txt`.
The content belongs in `carla/Unreal/CarlaUE4/Content/Carla`.
The expected editor executable is `UE4/Engine/Binaries/Win64/UE4Editor.exe`.

Install the produced Python API wheel into the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install .\carla\PythonAPI\carla\dist\carla-0.9.16-cp311-cp311-win_amd64.whl
```

## FSAE Cone Assets

The project includes four cone meshes, materials, textures, and a prop registry
under `sim/assets/cones/carla/FASE/`. With the editor and server closed, copy the
contents to the same asset path in CARLA:

```powershell
$coneDestination = '.\carla\Unreal\CarlaUE4\Content\Carla\FASE'
New-Item -ItemType Directory -Path $coneDestination -Force
Copy-Item .\sim\assets\cones\carla\FASE\* -Destination $coneDestination
```

Keep the directory name `FASE`: asset references and `FASE.Package.json` use
`/Game/Carla/FASE/`. Restart CARLA after installing the registry.
The required blueprint IDs are:

```text
static.prop.fsae_small_blue_single_white
static.prop.fsae_small_yellow_single_black
static.prop.fsae_small_orange_single_white
static.prop.fsae_large_orange_dual_white
```

Editable OBJ/MTL/PNG sources and dimensions are in
[sim/assets/cones](sim/assets/cones/README.md). For a different engine version,
reimport these sources, save the meshes under `/Game/Carla/FASE/`, retain their
names, and configure collision geometry so LiDAR rays can hit them.

## Run

From the repository root in PowerShell:

```powershell
$carlaProject = (Resolve-Path .\carla\Unreal\CarlaUE4\CarlaUE4.uproject).Path
& .\UE4\Engine\Binaries\Win64\UE4Editor.exe $carlaProject -game -windowed -ResX=1280 -ResY=720 -quality-level=Low -carla-rpc-port=2000
```

In another terminal:

```powershell
.\.venv\Scripts\python.exe -m sim.run.pathplanning
```

Use `sim.run.sensor_pathplanning` for the camera/LiDAR experiment. Both driving
experiments load a blank world and need the project cone assets installed.
