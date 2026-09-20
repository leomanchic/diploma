"""Small MAVLink setpoint transport for PX4 SITL Offboard mode.

MAVSDK remains the action, telemetry and parameter client. This transport
sends only velocity/yaw setpoints and the mode command over PX4's GCS MAVLink
endpoint, because the pinned MAVSDK server returns NO_SETPOINT_SET even after
acknowledging both of its Offboard setpoint RPCs on this Linux installation.
"""

from __future__ import annotations

import asyncio
import contextlib
import math


class DirectOffboard:
    # SET_POSITION_TARGET_LOCAL_NED: positions, accelerations and yaw rate are
    # ignored; vx, vy, vz and yaw are used. Values are NED and radians.
    VELOCITY_YAW_MASK = 0x007 | 0x1C0 | 0x800

    def __init__(self, connection: object, period_s: float):
        self.connection = connection
        self.period_s = period_s
        self.velocity_ned_yaw = (0.0, 0.0, 0.0, 90.0)
        self._task: asyncio.Task | None = None

    @classmethod
    async def connect(cls, period_s: float, timeout_s: float = 30.0) -> "DirectOffboard":
        from pymavlink import mavutil

        # PX4's normal MAVLink instance sends from UDP 18570 to GCS UDP 14550.
        connection = mavutil.mavlink_connection(
            "udpin:127.0.0.1:14550", source_system=245, source_component=190)
        heartbeat = await asyncio.to_thread(connection.wait_heartbeat, timeout=timeout_s)
        if heartbeat is None or heartbeat.get_srcSystem() != 1:
            raise TimeoutError("no PX4 heartbeat on GCS MAVLink UDP 14550")
        return cls(connection, period_s)

    def set_velocity(self, north: float, east: float, down: float, yaw_deg: float) -> None:
        self.velocity_ned_yaw = (float(north), float(east), float(down), float(yaw_deg))

    def _send_setpoint(self) -> None:
        from pymavlink import mavutil

        north, east, down, yaw_deg = self.velocity_ned_yaw
        self.connection.mav.set_position_target_local_ned_send(
            0, 1, 1, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            self.VELOCITY_YAW_MASK, 0, 0, 0,
            north, east, down, 0, 0, 0, math.radians(yaw_deg), 0)

    async def _stream(self) -> None:
        while True:
            self._send_setpoint()
            await asyncio.sleep(self.period_s)

    async def start(self, drone: object, timeout_s: float = 10.0) -> None:
        from pymavlink import mavutil
        from mavsdk.telemetry import FlightMode

        if self._task is not None:
            raise RuntimeError("Offboard setpoint stream already started")
        self._task = asyncio.create_task(self._stream())
        await asyncio.sleep(1.25)  # PX4 requires >1 s of setpoints at >2 Hz.
        if self._task.done():
            self._task.result()
        self.connection.mav.command_long_send(
            1, 1, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            6, 0, 0, 0, 0, 0)  # PX4 custom main mode 6 = OFFBOARD.

        async def await_mode() -> None:
            async for mode in drone.telemetry.flight_mode():
                if mode == FlightMode.OFFBOARD:
                    return

        try:
            await asyncio.wait_for(await_mode(), timeout_s)
        except TimeoutError as exc:
            raise TimeoutError("PX4 did not confirm OFFBOARD flight mode") from exc

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self.connection.close()
