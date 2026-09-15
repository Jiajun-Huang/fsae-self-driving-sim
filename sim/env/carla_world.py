from __future__ import annotations

from typing import Any, Dict, Optional

import carla


class CarlaWorld:
    """Thin wrapper around the CARLA client and world."""

    def __init__(self, host: str = "127.0.0.1", port: int = 2000, timeout: int = 10):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.vehicle: Optional[carla.Actor] = None
        self.sensors: Dict[str, carla.Sensor] = {}

    def connect(self) -> "CarlaWorld":
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        return self

    def set_weather(self, weather_name: str = "ClearNoon") -> None:
        if self.world is None:
            raise RuntimeError("CARLA world is not connected.")

        weather_map = {
            "ClearNoon": carla.WeatherParameters.ClearNoon,
            "ClearSunset": carla.WeatherParameters.ClearSunset,
            "CloudyNoon": carla.WeatherParameters.CloudyNoon,
            "WetNoon": carla.WeatherParameters.WetNoon,
            "HardRainNoon": carla.WeatherParameters.HardRainNoon,
            "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
        }

        weather = weather_map.get(weather_name, carla.WeatherParameters.ClearNoon)
        self.world.set_weather(weather)

    def spawn_vehicle(
        self,
        blueprint_name: str = "vehicle.tesla.model3",
        spawn_point: Optional[carla.Transform] = None,
        role_name: str = "ego",
    ) -> carla.Actor:
        if self.world is None:
            raise RuntimeError("CARLA world is not connected.")

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(blueprint_name)[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", role_name)
        if spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = spawn_points[0] if spawn_points else carla.Transform()

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point, attach_to=None)
        self.vehicle.set_autopilot(False)
        return self.vehicle

    def attach_sensor(
        self,
        sensor_type: str,
        transform: Optional[carla.Transform] = None,
        attributes: Optional[Dict[str, str]] = None,
        sensor_name: str = "sensor",
    ) -> carla.Sensor:
        if self.world is None or self.vehicle is None:
            raise RuntimeError(
                "World and vehicle must be initialized before attaching sensors."
            )

        blueprint_library = self.world.get_blueprint_library()
        blueprint = blueprint_library.find(sensor_type)
        if attributes:
            for key, value in attributes.items():
                blueprint.set_attribute(key, value)

        if transform is None:
            transform = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0))

        sensor = self.world.spawn_actor(blueprint, transform, attach_to=self.vehicle)
        self.sensors[sensor_name] = sensor
        return sensor

    def tick(self, steps: int = 1) -> None:
        if self.world is None:
            raise RuntimeError("CARLA world is not connected.")

        for _ in range(steps):
            self.world.tick()

    def destroy(self) -> None:
        for sensor in self.sensors.values():
            sensor.destroy()

        if self.vehicle is not None:
            self.vehicle.destroy()

        self.sensors.clear()
        self.vehicle = None

    def get_world_snapshot(self) -> Dict[str, Any]:
        if self.world is None:
            return {}

        snapshot = self.world.get_snapshot()
        return {
            "frame": snapshot.frame,
            "weather": self.world.get_weather(),
        }
