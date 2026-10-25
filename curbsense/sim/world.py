"""ScenarioSpec -> MuJoCo model: a 4-wheel skid-steer delivery robot on a sidewalk that drops to the road.

Coordinates: +x points from the sidewalk toward the road, the curb edge is at x = 0.
"""
from __future__ import annotations

import math

import mujoco

from curbsense.scenario import ScenarioSpec

WHEEL_R = 0.08
TRACK = 0.34      # left-right wheel distance
BASE = 0.36       # front-back wheel distance
CARGO_Z = 0.18    # tall cargo bay = high center of mass, like a real delivery robot
CARGO_MASS = 12.0
SLOPE_LEN = 3.0   # m of ramp for kind="slope"
FRONT_MU = 0.08   # front wheels act like swivel casters (low lateral grip) so the robot can actually steer
MOTOR_NM = 2.5    # per-wheel hub motor torque limit: sets how steep a hill or how tall a curb it can climb


def mjcf(spec: ScenarioSpec) -> str:
    h = spec.curb_height
    yaw = math.radians(spec.approach_angle_deg)
    x0 = -spec.start_dist
    y0 = spec.lateral_offset
    # spawn exactly at rest height: a 2 cm drop here made a fake ~3 g impact spike in every episode (found by the managed agent)
    z0 = (h if spec.kind == "drop" else 0.0) + WHEEL_R + 0.07 + 0.002
    qw, qz = math.cos(yaw / 2), math.sin(yaw / 2)
    mu = spec.friction
    wheels = ""
    for name, wx, wy in [("fl", BASE / 2, TRACK / 2), ("fr", BASE / 2, -TRACK / 2),
                         ("rl", -BASE / 2, TRACK / 2), ("rr", -BASE / 2, -TRACK / 2)]:
        wheels += f"""
      <body name="w_{name}" pos="{wx} {wy} -0.07">
        <joint name="j_{name}" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom type="cylinder" size="{WHEEL_R} 0.03" euler="1.5708 0 0" mass="0.6"
              friction="{FRONT_MU if name.startswith('f') else mu} 0.01 0.001" priority="{1 if name.startswith('f') else 0}" rgba=".1 .1 .1 1"/>
      </body>"""
    return f"""
<mujoco model="curb">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual><global offwidth="960" offheight="540"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="road" type="2d" builtin="checker" rgb1=".25 .25 .27" rgb2=".22 .22 .24" width="64" height="64"/>
    <material name="road" texture="road" texrepeat="20 20"/>
    <texture name="walk" type="2d" builtin="checker" rgb1=".72 .71 .68" rgb2=".66 .65 .62" width="64" height="64"/>
    <material name="walk" texture="walk" texrepeat="8 8"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".6 .75 .9" rgb2=".95 .97 1" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="2 -3 6" dir="-0.3 0.4 -1" directional="true" castshadow="true"/>
    <geom name="road" type="plane" size="20 20 0.1" material="road" friction="{mu} 0.01 0.001"/>
{terrain(spec)}
    <body name="robot" pos="{x0} {y0} {z0}" quat="{qw} 0 0 {qz}">
      <freejoint/>
      <geom name="chassis" type="box" size="0.26 0.15 0.06" mass="12" rgba=".95 .95 .97 1"/>
      <geom name="cargo" type="box" pos="-0.02 0 {CARGO_Z}" size="0.22 0.15 0.12" mass="{CARGO_MASS}" rgba=".2 .55 .95 1"/>
      <geom name="flag" type="capsule" fromto="-0.2 0.12 0.2 -0.2 0.12 0.55" size="0.008" mass="0.05" rgba="1 .4 .1 1"/>
      <camera name="front" pos="0.27 0 0.08" xyaxes="0 -1 0 0.15 0 1" fovy="75"/>
      <site name="imu" pos="0 0 0"/>{wheels}
    </body>
  </worldbody>
  <actuator>
    <velocity name="v_fl" joint="j_fl" kv="6" ctrlrange="-40 40" forcerange="-{MOTOR_NM} {MOTOR_NM}"/>
    <velocity name="v_fr" joint="j_fr" kv="6" ctrlrange="-40 40" forcerange="-{MOTOR_NM} {MOTOR_NM}"/>
    <velocity name="v_rl" joint="j_rl" kv="6" ctrlrange="-40 40" forcerange="-{MOTOR_NM} {MOTOR_NM}"/>
    <velocity name="v_rr" joint="j_rr" kv="6" ctrlrange="-40 40" forcerange="-{MOTOR_NM} {MOTOR_NM}"/>
  </actuator>
  <sensor>
    <accelerometer name="acc" site="imu"/>
    <gyro name="gyro" site="imu"/>
  </sensor>
</mujoco>"""


def terrain(spec: ScenarioSpec) -> str:
    h, mu = spec.curb_height, spec.friction
    fr = f'friction="{mu} 0.01 0.001"'
    if spec.kind == "obstacle":  # flat sidewalk with a bollard / pole / bin in the way
        return (f'<geom name="obstacle" type="cylinder" pos="{spec.obs_x} {spec.obs_y} 0.5" size="{spec.obs_r} 0.5" '
                f'rgba=".35 .35 .38 1" {fr}/>')
    if spec.kind == "pedestrian":  # flat sidewalk, a person crossing the robot's path (kinematic mocap body)
        return (f'<body name="ped" mocap="true" pos="{spec.obs_x} {spec.ped_start_y} 0">'
                f'<geom name="ped" type="capsule" fromto="0 0 0.25 0 0 1.5" size="0.22" rgba=".85 .55 .4 1"/></body>')
    if spec.kind == "climb":  # road x < 0, raised sidewalk x > 0
        return (f'<geom name="sidewalk" type="box" pos="5 0 {h / 2}" size="5 20 {h / 2}" material="walk" {fr}/>\n'
                f'    <geom name="curb_line" type="box" pos="-0.01 0 {h / 2}" size="0.012 20 {h / 2 + 0.001}" rgba=".9 .8 .2 1" contype="0" conaffinity="0"/>')
    if spec.kind == "slope":  # flat approach x < 0, ramp rising from x = 0, plateau at the top
        th = math.radians(spec.slope_deg)
        L, t = SLOPE_LEN, 0.05
        cx, cz = L / 2 * math.cos(th) + t * math.sin(th), L / 2 * math.sin(th) - t * math.cos(th)
        top = L * math.sin(th)
        return (f'<geom name="ramp" type="box" pos="{cx} 0 {cz}" size="{L / 2} 20 {t}" euler="0 {-th} 0" material="walk" {fr}/>\n'
                f'    <geom name="plateau" type="box" pos="{L * math.cos(th) + 5} 0 {top / 2}" size="5 20 {max(top / 2, 0.001)}" material="walk" {fr}/>')
    return (f'<geom name="sidewalk" type="box" pos="-5 0 {h / 2}" size="5 20 {h / 2}" material="walk" {fr}/>\n'
            f'    <geom name="curb_line" type="box" pos="0.01 0 {h / 2}" size="0.012 20 {h / 2 + 0.001}" rgba=".9 .8 .2 1" contype="0" conaffinity="0"/>')


ROBOT_R = 0.30  # m, radius of a circle around the robot footprint (for clearance)


def goal_x(spec: ScenarioSpec) -> float:
    if spec.kind in ("obstacle", "pedestrian"):
        return spec.obs_x + 1.5
    if spec.kind == "slope":
        return SLOPE_LEN * math.cos(math.radians(spec.slope_deg)) + 0.4
    return 1.2


def build(spec: ScenarioSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(mjcf(spec))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def obstacle_pos(spec: ScenarioSpec, t: float) -> tuple[float, float] | None:
    """World (x, y) of the obstacle / person at time t, or None for curb and slope scenarios."""
    if spec.kind == "obstacle":
        return spec.obs_x, spec.obs_y
    if spec.kind == "pedestrian":
        d = 1.0 if spec.ped_start_y <= 0 else -1.0
        y = spec.ped_start_y + d * spec.ped_speed * max(0.0, t - spec.ped_t0)
        y = min(y, -spec.ped_start_y) if d > 0 else max(y, -spec.ped_start_y)
        return spec.obs_x, y
    return None


def move_actors(spec: ScenarioSpec, data: mujoco.MjData, t: float) -> None:
    if spec.kind == "pedestrian":
        x, y = obstacle_pos(spec, t)
        data.mocap_pos[0] = [x, y, 0.0]


def set_drive(data: mujoco.MjData, v: float, w: float) -> None:
    """Skid-steer: forward speed v (m/s), yaw rate w (rad/s) -> wheel angular velocities."""
    left = (v - w * TRACK / 2) / WHEEL_R
    right = (v + w * TRACK / 2) / WHEEL_R
    data.ctrl[:] = [left, right, left, right]
