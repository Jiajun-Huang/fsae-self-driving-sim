# Simulation Package

Setup, run commands, current limitations, and the module layout are documented
in the [project README](../README.md).

Run experiment modules with the interpreter that has the project installed:

```powershell
python -m sim.run.preview --debug-road
python -m sim.run.pathplanning
python -m sim.run.sensor_pathplanning
```

The driving experiments require a running CARLA server and registered FSAE
cone assets. See [CARLA Windows Setup](../CARLA_WINDOWS_SETUP.md).
