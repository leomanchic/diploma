"""Portable PX4 pilot contract checks; actual SITL flight is a local system gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from px4.coordinates import (
    BODY_FRD_FROM_FLU, WORLD_NED_FROM_ENU, enu_to_ned, flu_to_frd,
    gazebo_wxyz_to_px4_ned_frd_matrix, ned_to_enu,
)
from px4.create_scene import create_scene
from px4.create_probe import create_probe
from px4.finalize_recording import finalize, phase_intervals
from px4.launch_gazebo import launch
from px4.mavlink_offboard import DirectOffboard
from px4.run_flight import planned_turn_displacement, smooth_turn_velocity
from simulation.gazebo_offline import load_gazebo_recording
from validation.gazebo_experiment import create_experiment, initialize_experiment, verify_results
from validation.gazebo_offline_run import process_recording
from visualization.gazebo_offline_view import create_viewer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_enu_ned_body_axes_and_quaternion_order() -> None:
    east, north, up = np.eye(3)
    np.testing.assert_array_equal(enu_to_ned(east), [0, 1, 0])
    np.testing.assert_array_equal(enu_to_ned(north), [1, 0, 0])
    np.testing.assert_array_equal(enu_to_ned(up), [0, 0, -1])
    np.testing.assert_array_equal(ned_to_enu(enu_to_ned([11, -4, 7])), [11, -4, 7])
    np.testing.assert_array_equal(flu_to_frd([1, 2, 3]), [1, -2, -3])
    assert np.linalg.det(WORLD_NED_FROM_ENU) == 1
    assert np.linalg.det(BODY_FRD_FROM_FLU) == 1
    # Gazebo identity: body forward points East, i.e. PX4 yaw +90 degrees.
    identity = gazebo_wxyz_to_px4_ned_frd_matrix([1, 0, 0, 0])
    np.testing.assert_allclose(identity[:, 0], [0, 1, 0], atol=1e-15)
    # Gazebo ENU +90-degree yaw: body forward points North, PX4 yaw 0.
    north_facing = gazebo_wxyz_to_px4_ned_frd_matrix([
        math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)])
    np.testing.assert_allclose(north_facing[:, 0], [1, 0, 0], atol=1e-15)
    with pytest.raises(ValueError, match="unit w,x,y,z"):
        gazebo_wxyz_to_px4_ned_frd_matrix([0, 0, 0, 2])


def test_turn_command_is_continuous_and_uses_ned() -> None:
    beginning = smooth_turn_velocity(0, 5, 3, math.pi/2)
    end = smooth_turn_velocity(5, 5, 3, math.pi/2)
    np.testing.assert_allclose(beginning, [0, 3, 90], atol=1e-12)
    np.testing.assert_allclose(end, [3, 0, 0], atol=1e-12)
    north, east = planned_turn_displacement(5, 3, math.pi/2)
    assert 8 < north < 12 and 8 < east < 12


def test_direct_mavlink_velocity_setpoint_has_ned_axes_and_yaw_radians(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    link = SimpleNamespace(mav=SimpleNamespace(
        set_position_target_local_ned_send=lambda *args: calls.append(args)))
    monkeypatch.setitem(sys.modules, "pymavlink", SimpleNamespace(
        mavutil=SimpleNamespace(mavlink=SimpleNamespace(MAV_FRAME_LOCAL_NED=1))))
    sender = DirectOffboard(link, 0.05)
    sender.set_velocity(1.0, 2.0, -0.5, 90.0)
    sender._send_setpoint()
    assert calls[0][3:5] == (1, 0x9C7)
    assert calls[0][8:11] == (1.0, 2.0, -0.5)
    assert calls[0][14] == pytest.approx(math.pi/2)


def test_gazebo_launcher_records_applied_seed_and_protects_recording(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from px4 import launch_gazebo as module

    root = tmp_path / "acoustic"
    observer = root / "build/px4-observer/libgazebo_px4_observer.so"
    observer.parent.mkdir(parents=True)
    observer.touch()
    px4_root = tmp_path / "PX4-Autopilot"
    (px4_root / "Tools/simulation/gz/models").mkdir(parents=True)
    (px4_root / "build/px4_sitl_default/src/modules/simulation/gz_plugins").mkdir(parents=True)
    directory = tmp_path / "new_recording"
    directory.mkdir()
    (directory / "scene.sdf").write_text("<sdf/>")
    (directory / "flight_plan.json").write_text(json.dumps({"gazebo": {"seed": 20260920}}))
    calls = []
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs: (
        calls.append((argv, kwargs)), SimpleNamespace(returncode=0))[1])
    assert launch(directory, px4_root=px4_root, headless=True) == 0
    saved = json.loads((directory / "gazebo_launch.json").read_text())
    assert saved["seed_applied"] == 20260920
    assert saved["argv"][-3:-1] == ["20260920", "-s"]
    assert calls[0][1]["env"]["GZ_SIM_RESOURCE_PATH"] == str(
        px4_root / "Tools/simulation/gz/models")
    with pytest.raises(FileExistsError, match="already started"):
        launch(directory, px4_root=px4_root)


def test_scene_has_separate_observer_and_protects_new_directory(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from px4 import create_scene as module

    fake = tmp_path / "px4"
    world = fake / "Tools/simulation/gz/worlds/default.sdf"
    model = fake / "Tools/simulation/gz/models/x500/model.sdf"
    server = fake / "src/modules/simulation/gz_bridge/server.config"
    for path in (world, model, server):
        path.parent.mkdir(parents=True, exist_ok=True)
    world.write_text("<sdf version='1.9'><world name='default'><physics>"
                     "<max_step_size>0.004</max_step_size><real_time_update_rate>250</real_time_update_rate>"
                     "</physics></world></sdf>")
    model.write_text("<sdf version='1.9'><model name='x500'/></sdf>")
    server.write_text("<server_config><plugins><plugin entity_name='*' entity_type='world' "
                      "filename='gz-sim-physics-system' name='gz::sim::systems::Physics'/>"
                      "</plugins></server_config>")
    monkeypatch.setattr(module, "_git_sha", lambda path: "test-sha")
    monkeypatch.setattr(module.subprocess, "check_output", lambda args, **kwargs: " test-sha path\n")
    destination = tmp_path / "new"
    scene = create_scene(destination, px4_root=fake)
    content = scene.read_text()
    assert content.count('name="S0"') == 1
    assert content.count('name="S1"') == 1
    assert content.count('name="S2"') == 1
    assert 'name="acoustic::X500Observer"' in content
    assert 'entity_name=' not in content
    assert 'name="x500"' not in content  # PX4 spawns the vehicle.
    with pytest.raises(FileExistsError, match="must be empty"):
        create_scene(destination, px4_root=fake)


def _flight_fixture(directory: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from px4 import finalize_recording as module

    directory.mkdir()
    plan = json.loads((Path(__file__).resolve().parents[1] / "px4/flight_plan.json").read_text())
    plan["vehicle"].update(hover_before_s=1.2, straight_s=1.2,
                           turn_s=1.2, hover_after_s=1.2)
    (directory / "flight_plan.json").write_text(json.dumps(plan))
    (directory / "scene.sdf").write_text("<sdf version='1.9'><world name='test'/></sdf>\n")
    (directory / "flight_commands.csv").write_text("sim_time_s,flight_phase,command\n1,takeoff,action.takeoff\n")
    (directory / "planned_route.csv").write_text("sim_time_s,x_m,y_m,z_m,flight_phase\n0,35,35,0,takeoff\n4,40,35,16,turn\n")
    (directory / "flight_program_status.json").write_text(json.dumps({
        "completed": True, "python_version": "3.12.3", "mavsdk_version": "3.10.0",
        "grpcio_version": "1.84.0", "protobuf_version": "7.36.2",
    }))
    (directory / "autopilot_parameters.json").write_text(json.dumps({"values": {"MPC_XY_VEL_MAX": 12}}))
    (directory / "preflight.json").write_text(json.dumps({
        "flight_plan_sha256": _sha(directory / "flight_plan.json"),
        "scene_sdf_sha256": _sha(directory / "scene.sdf"),
        "px4_git_sha": "test-px4-sha",
        "px4_gz_submodule_sha": "test-gz-sha",
        "px4_submodule_status": [" test-gz-sha Tools/simulation/gz"],
        "px4_x500_model_sdf_sha256": "test-model-sha",
        "observer_source_sha256": _sha(Path(__file__).resolve().parents[1] / "px4/observe_x500.cc"),
        "pilot_source_sha256": {"run_flight.py": _sha(Path(__file__).resolve().parents[1] / "px4/run_flight.py")},
        "acoustic_git_sha_at_creation": "test-acoustic-sha",
    }))
    (directory / "flight_phase.txt").write_text("landed\n")
    boundaries = [(0, "preflight"), (0.5, "takeoff"), (2, "hover_before"),
                  (3.2, "straight"), (4.4, "turn"), (5.6, "hover_after"),
                  (6.8, "land"), (8, "landed")]
    with (directory / "gazebo_state.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("sim_time_s", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz",
                         "vx_mps", "vy_mps", "vz_mps", "flight_phase"))
        for index in range(1, 421):
            t = round(index * 0.02, 8)
            phase = max((name for start, name in boundaries if t >= start),
                        key=lambda name: [p for _, p in boundaries].index(name))
            z = min(16, max(0, (t-0.5)*11)) if t < 6.8 else max(0, 16-(t-6.8)*14)
            x = 35 + 3*min(max(t-3.2, 0), 1.2)
            writer.writerow((t, x, 35, z, 1, 0, 0, 0, 0, 0, 0, phase))
    monkeypatch.setattr(module.subprocess, "check_output",
                        lambda args, **kwargs: "test-px4-sha" if args[0] == "git" else "8.15.0")
    return directory


def test_flight_time_validation_and_full_short_offline_path(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    directory = _flight_fixture(tmp_path / "flight", monkeypatch)
    manifest = finalize(directory, px4_root=tmp_path)
    assert manifest["kind"] == "px4_flight"
    assert [item["name"] for item in manifest["phase_intervals"]][-6:] == [
        "hover_before", "straight", "turn", "hover_after", "land", "landed"]
    recording = load_gazebo_recording(directory)
    assert recording.world_velocities_mps.shape == (420, 3)
    assert recording.flight_phases[0] == "preflight"
    probe = create_probe(directory, tmp_path / "probe")
    assert json.loads(probe.read_text())["processing"]["audio"]["duration_s"] == 1.0
    assert _sha(probe.parent / "gazebo_state.csv") == _sha(directory / "gazebo_state.csv")
    # The short acoustic probe is an explicit separate processing config,
    # frozen before any result exists. It is not a claimed physical flight.
    short = json.loads((directory / "processing_config.json").read_text())
    short["processing"]["audio"].update(reception_start_s=3.0, duration_s=0.35)
    config_path = directory / "short_processing_config.json"
    config_path.write_text(json.dumps(short))
    experiment = initialize_experiment(directory, config_path)
    assert experiment["recording"]["state_sha256"] == _sha(directory / "gazebo_state.csv")
    summary = process_recording(directory)
    assert summary["recording_kind"] == "px4_flight"
    assert set(summary["methods"][next(iter(summary["methods"]))]["phase_metrics"]) == {
        "hover_before", "straight", "turn", "hover_after"}
    assert verify_results(directory)["run_id"] == experiment["run_id"]
    viewer = create_viewer(directory)
    assert "заданный маршрут" in viewer.read_text()
    assert "Фазы записанного полёта" in viewer.read_text()
    variant = create_experiment(directory, tmp_path / "variant", snr_db=5.0)
    assert variant["run_id"] != experiment["run_id"]
    assert load_gazebo_recording(tmp_path / "variant").csv_sha256 == recording.csv_sha256


def test_flight_gap_duplicate_and_reset_rejected(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    directory = _flight_fixture(tmp_path / "flight", monkeypatch)
    manifest = finalize(directory, px4_root=tmp_path)
    state = directory / "gazebo_state.csv"
    original = state.read_text().splitlines(keepends=True)
    for replacement, reason in ((original[51], "duplicate"), ("0,"+original[51].split(",", 1)[1], "reset")):
        changed = original.copy()
        changed[52] = replacement
        state.write_text("".join(changed))
        with pytest.raises(ValueError, match="duplicate timestamp or simulation time reset"):
            load_gazebo_recording(directory)
    state.write_text("".join(original[:51] + original[52:]))
    with pytest.raises(ValueError, match="missing Gazebo export sample"):
        load_gazebo_recording(directory)
    state.write_text("".join(original))
    (directory / "planned_route.csv").write_text("tampered\n")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_gazebo_recording(directory)


def test_init_rejects_phase_labels_that_disagree_with_recording(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = _flight_fixture(tmp_path / "flight", monkeypatch)
    finalize(directory, px4_root=tmp_path)
    config = json.loads((directory / "processing_config.json").read_text())
    config["processing"]["evaluation_phases"][2]["start_s"] += 0.2
    wrong = directory / "wrong_config.json"
    wrong.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="flight phases differ"):
        initialize_experiment(directory, wrong)
    assert not (directory / "experiment.json").exists()


def test_phase_order_must_complete() -> None:
    with pytest.raises(ValueError, match="incomplete or out of order"):
        phase_intervals([{"sim_time_s": "0", "flight_phase": "hover_before"}], 0.02)
