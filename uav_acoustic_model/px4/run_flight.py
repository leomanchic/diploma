"""Fly X500 through PX4/MAVSDK; Gazebo observer records physical truth separately."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import platform
import time
from pathlib import Path
from importlib import metadata

from .mavlink_offboard import DirectOffboard


def _latest_sample(path: Path) -> tuple[float, float, float, float]:
    """Return flushed Gazebo simulation time and world position, never wall time."""
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        lines = []
    if len(lines) < 2:
        raise ValueError("waiting for the X500 Gazebo observer to record state")
    fields = lines[-1].split(",")
    return tuple(float(fields[index]) for index in range(4))


async def wait_sample(path: Path, maximum_wall_wait_s: float) -> tuple[float, float, float, float]:
    deadline = time.monotonic() + maximum_wall_wait_s
    while time.monotonic() < deadline:
        try:
            return _latest_sample(path)
        except (OSError, ValueError, IndexError):
            await asyncio.sleep(0.1)
    raise TimeoutError("no X500 Gazebo state samples; check PX4 spawn and observer plugin")


async def wait_sim_time(path: Path, target_s: float, maximum_wall_wait_s: float) -> float:
    deadline = time.monotonic() + maximum_wall_wait_s
    while time.monotonic() < deadline:
        t = (await wait_sample(path, 2.0))[0]
        if t >= target_s:
            return t
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Gazebo simulation time did not reach {target_s:.3f} s")


def _phase(path: Path, name: str) -> None:
    temporary = path.with_suffix(".pending")
    temporary.write_text(name + "\n")
    temporary.replace(path)


def smooth_turn_velocity(elapsed_s: float, duration_s: float,
                         speed_mps: float, turn_angle_rad: float) -> tuple[float, float, float]:
    """Continuous commanded NED horizontal velocity and yaw (degrees)."""
    u = min(1.0, max(0.0, elapsed_s / duration_s))
    angle = turn_angle_rad * (3*u*u - 2*u*u*u)
    north = speed_mps * math.sin(angle)
    east = speed_mps * math.cos(angle)
    return north, east, 90.0 - math.degrees(angle)


def planned_turn_displacement(duration_s: float, speed_mps: float,
                              angle_rad: float, intervals: int = 1000) -> tuple[float, float]:
    """Trapezoidal integral of the command, used only for the planned route."""
    dt = duration_s / intervals
    north = east = 0.0
    for index in range(intervals + 1):
        vn, ve, _ = smooth_turn_velocity(index * dt, duration_s, speed_mps, angle_rad)
        weight = 0.5 if index in (0, intervals) else 1.0
        north += weight * vn * dt
        east += weight * ve * dt
    return north, east


def planned_route(plan: dict, phase_starts: dict[str, float],
                  landing_end_s: float) -> list[dict[str, float | str]]:
    """Requested spatial path and phase clock, kept separate from Gazebo truth."""
    vehicle = plan["vehicle"]
    x0, y0, z0 = vehicle["spawn_enu_m"]
    z = z0 + vehicle["takeoff_altitude_m"]
    speed = vehicle["horizontal_speed_mps"]
    straight = vehicle["straight_s"]
    turn = vehicle["turn_s"]
    angle = vehicle["turn_angle_rad"]
    north_turn, east_turn = planned_turn_displacement(turn, speed, angle)
    points: list[dict[str, float | str]] = []

    def add(t: float, x: float, y: float, height: float, phase: str) -> None:
        points.append({"sim_time_s": float(t), "x_m": x, "y_m": y,
                       "z_m": height, "flight_phase": phase})

    takeoff_start = phase_starts["takeoff"]
    hover_start = phase_starts["hover_before"]
    add(takeoff_start, x0, y0, z0, "takeoff")
    add(hover_start, x0, y0, z, "hover_before")
    straight_start = phase_starts["straight"]
    add(straight_start, x0, y0, z, "straight")
    add(phase_starts["turn"], x0 + speed*straight, y0, z, "turn")
    for index in range(1, 101):
        active = turn * index / 100
        # Integrate the original full-duration turn up to this time.
        steps = max(4, index * 10)
        dt = active / steps
        dn = de = 0.0
        for j in range(steps + 1):
            vn, ve, _ = smooth_turn_velocity(j*dt, turn, speed, angle)
            weight = 0.5 if j in (0, steps) else 1.0
            dn += weight*vn*dt
            de += weight*ve*dt
        add(phase_starts["turn"] + active, x0 + speed*straight + de,
            y0 + dn, z, "turn")
    hover_end = phase_starts["hover_after"] + vehicle["hover_after_s"]
    x_end, y_end = x0 + speed*straight + east_turn, y0 + north_turn
    add(hover_end, x_end, y_end, z, "hover_after")
    add(landing_end_s, x_end, y_end, z0, "land")
    return points


async def fly(directory: Path) -> None:
    from mavsdk import System

    directory = Path(directory).resolve()
    plan = json.loads((directory / "flight_plan.json").read_text())
    vehicle = plan["vehicle"]
    state_path = directory / "gazebo_state.csv"
    phase_path = directory / "flight_phase.txt"
    maximum_wait = float(vehicle["maximum_wall_wait_s"])
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    deadline = time.monotonic() + maximum_wait
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("MAVSDK did not connect to PX4 on UDP 14540")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("PX4 did not report healthy global/home position")
    await wait_sample(state_path, maximum_wait)
    # Offboard flight uses MAVLink setpoints, without an RC transmitter.
    await asyncio.wait_for(drone.param.set_param_int("COM_RC_IN_MODE", 4), 10)
    parameter_names = (
        "MPC_XY_VEL_MAX", "MPC_Z_VEL_MAX_UP", "MPC_ACC_HOR", "MPC_ACC_HOR_MAX",
        "MPC_TKO_SPEED", "MPC_LAND_SPEED",
    )
    parameters = {}
    for name in parameter_names:
        parameters[name] = await asyncio.wait_for(drone.param.get_param_float(name), 10)
    parameters["COM_RC_IN_MODE"] = await asyncio.wait_for(
        drone.param.get_param_int("COM_RC_IN_MODE"), 10)
    (directory / "autopilot_parameters.json").write_text(json.dumps({
        "schema_version": 1, "read_from": "PX4 MAVLink PARAM via MAVSDK before arming",
        "values": parameters,
    }, indent=2) + "\n")
    commands: list[dict[str, float | str]] = []
    phase_starts: dict[str, float] = {}
    offboard_attempts: list[str] = []
    direct_offboard: DirectOffboard | None = None

    def mark(name: str) -> float:
        t = _latest_sample(state_path)[0]
        _phase(phase_path, name)
        phase_starts[name] = t
        return t

    def log(phase: str, command: str, north: float = float("nan"),
            east: float = float("nan"), down: float = float("nan"),
            yaw: float = float("nan")) -> None:
        commands.append({"sim_time_s": _latest_sample(state_path)[0], "flight_phase": phase,
                         "command": command, "north": north, "east": east,
                         "down": down, "yaw_deg": yaw})

    try:
        await drone.action.set_takeoff_altitude(vehicle["takeoff_altitude_m"])
        await drone.action.arm()
        mark("takeoff")
        log("takeoff", "action.takeoff")
        await drone.action.takeoff()
        deadline = time.monotonic() + maximum_wait
        while time.monotonic() < deadline:
            t, _, _, height = await wait_sample(state_path, 2.0)
            if height >= vehicle["takeoff_reached_altitude_m"]:
                break
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("X500 did not reach takeoff altitude")
        direct_offboard = await DirectOffboard.connect(
            vehicle["offboard_setpoint_period_s"])
        log("takeoff", "mavlink.velocity_setpoint 0,0,0")
        await direct_offboard.start(drone)
        offboard_attempts.append("direct MAVLink: PX4 OFFBOARD confirmed")
        hover_start = mark("hover_before")
        log("hover_before", "mavlink.velocity_setpoint", 0, 0, 0, 90)
        await wait_sim_time(state_path, hover_start + vehicle["hover_before_s"], maximum_wait)
        straight_start = mark("straight")
        speed = vehicle["horizontal_speed_mps"]
        direct_offboard.set_velocity(0.0, speed, 0.0, 90.0)
        log("straight", "mavlink.velocity_setpoint", 0, speed, 0, 90)
        await wait_sim_time(state_path, straight_start + vehicle["straight_s"], maximum_wait)
        turn_start = mark("turn")
        turn_duration = vehicle["turn_s"]
        command_period = vehicle["offboard_setpoint_period_s"]
        next_command = turn_start
        while True:
            t = _latest_sample(state_path)[0]
            elapsed = t - turn_start
            if elapsed >= turn_duration:
                break
            if t >= next_command:
                vn, ve, yaw = smooth_turn_velocity(elapsed, turn_duration, speed,
                                                    vehicle["turn_angle_rad"])
                direct_offboard.set_velocity(vn, ve, 0.0, yaw)
                log("turn", "mavlink.velocity_setpoint", vn, ve, 0, yaw)
                next_command = t + command_period
            await asyncio.sleep(0.01)
        direct_offboard.set_velocity(0.0, 0.0, 0.0, 0.0)
        hover_end_start = mark("hover_after")
        log("hover_after", "mavlink.velocity_setpoint", 0, 0, 0, 0)
        await wait_sim_time(state_path, hover_end_start + vehicle["hover_after_s"], maximum_wait)
        mark("land")
        log("land", "action.land")
        await drone.action.land()
        await direct_offboard.stop()
        direct_offboard = None
        deadline = time.monotonic() + maximum_wait
        ground = plan["ground_height_m"]
        while time.monotonic() < deadline:
            t, _, _, height = await wait_sample(state_path, 2.0)
            if height <= ground + 0.5:
                break
            await asyncio.sleep(0.1)
        else:
            raise TimeoutError("X500 did not land")
        await wait_sim_time(state_path, t + 1.0, maximum_wait)
        landed_start = mark("landed")
        await wait_sim_time(state_path, landed_start + 0.2, maximum_wait)
    except Exception:
        _phase(phase_path, "flight_failed")
        try:
            await drone.action.land()
        except Exception:
            pass
        raise
    finally:
        if direct_offboard is not None:
            await direct_offboard.stop()
        if commands:
            with (directory / "flight_commands.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(commands[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(commands)
        if "landed" in phase_starts:
            route = planned_route(plan, phase_starts, phase_starts["landed"])
            with (directory / "planned_route.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(route[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(route)
        (directory / "flight_program_status.json").write_text(json.dumps({
            "schema_version": 1,
            "phase_command_start_sim_time_s": phase_starts,
            "offboard_start_attempts": offboard_attempts,
            "completed": "landed" in phase_starts,
            "control": "PX4 SITL v1.17.0 via MAVSDK-Python 3.10.0 actions/telemetry and PyMAVLink 2.4.49 Offboard setpoints",
            "python_version": platform.python_version(),
            "mavsdk_version": metadata.version("mavsdk"),
            "pymavlink_version": metadata.version("pymavlink"),
            "grpcio_version": metadata.version("grpcio"),
            "protobuf_version": metadata.version("protobuf"),
        }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_directory", type=Path)
    args = parser.parse_args()
    asyncio.run(fly(args.recording_directory))
