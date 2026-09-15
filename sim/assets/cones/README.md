# FSAE Cone Models

These OBJ models are generated for the Formula SAE Driverless cone set.

OBJ units are centimeters, matching Unreal Engine's default world units.
The MTL file references solid-color PNG textures in `textures/` so UE4 imports visible colored materials instead of default gray slots.

| Cone | WEMAS part number | Dimensions | File |
| --- | --- | --- | --- |
| Small Blue, Single White Stripe | 400.000043.00.00 | 228 x 228 x 325 mm | fsae_small_blue_single_white.obj |
| Small Yellow, Single Black Stripe | 400.000013.01.10 | 228 x 228 x 325 mm | fsae_small_yellow_single_black.obj |
| Small Orange, Single White Stripe | 400.000013.00.00 | 228 x 228 x 325 mm | fsae_small_orange_single_white.obj |
| Large Orange, Dual White Stripes | 307.610500.00.00 | 285 x 285 x 505 mm | fsae_large_orange_dual_white.obj |

## CARLA / UE4 Import

The UE4 4.26.2 assets and prop registry are in `carla/FASE/`.
Copy their contents to `carla/Unreal/CarlaUE4/Content/Carla/FASE/` with CARLA closed,
then restart it. Keep the name `FASE` because the registry and asset references
use `/Game/Carla/FASE/`.

For a different engine version, import the OBJ files and keep the MTL and
`textures/` directory alongside them. Save the meshes with the names in
`carla/FASE/FASE.Package.json` and configure collision for LiDAR visibility.
See [CARLA Windows Setup](../../../CARLA_WINDOWS_SETUP.md) for full instructions.
