from __future__ import annotations

from datetime import date

import carla


def build_blank_opendrive(
    name: str = "FSAEBlank",
    length_m: float = 320.0,
    lane_width_m: float = 140.0,
) -> str:
    half_length = length_m / 2.0
    half_width = lane_width_m / 2.0
    today = date.today().isoformat()
    return f"""<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="{name}" version="1.00" date="{today}" north="400" south="-400" east="400" west="-400" vendor="self-driving-sim"/>
  <road name="flat_anchor" length="{length_m:.3f}" id="1" junction="-1">
    <link/>
    <type s="0.0" type="town">
      <speed max="30" unit="m/s"/>
    </type>
    <planView>
      <geometry s="0.0" x="{-half_length:.3f}" y="0.0" hdg="0.0" length="{length_m:.3f}">
        <line/>
      </geometry>
    </planView>
    <elevationProfile>
      <elevation s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>
    </elevationProfile>
    <lateralProfile/>
    <lanes>
      <laneSection s="0.0">
        <center>
          <lane id="0" type="none" level="false">
            <roadMark sOffset="0.0" type="none" weight="standard" color="standard" width="0.0"/>
          </lane>
        </center>
        <left>
          <lane id="1" type="driving" level="false">
            <width sOffset="0.0" a="{half_width:.3f}" b="0.0" c="0.0" d="0.0"/>
            <roadMark sOffset="0.0" type="none" weight="standard" color="standard" width="0.0"/>
          </lane>
        </left>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0.0" a="{half_width:.3f}" b="0.0" c="0.0" d="0.0"/>
            <roadMark sOffset="0.0" type="none" weight="standard" color="standard" width="0.0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
    <objects/>
    <signals/>
  </road>
</OpenDRIVE>
"""


def generate_blank_world(
    client: carla.Client,
    name: str = "FSAEBlank",
    length_m: float = 320.0,
    lane_width_m: float = 140.0,
    additional_width_m: float = 20.0,
) -> carla.World:
    xodr = build_blank_opendrive(
        name=name,
        length_m=length_m,
        lane_width_m=lane_width_m,
    )
    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=50.0,
        wall_height=0.0,
        additional_width=additional_width_m,
        smooth_junctions=True,
        enable_mesh_visibility=True,
        enable_pedestrian_navigation=False,
    )
    return client.generate_opendrive_world(xodr, params)
