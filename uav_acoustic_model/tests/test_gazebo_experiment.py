"""Frozen Gazebo replay identity and artifact-consistency checks."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from simulation import gazebo_offline
from gazebo.create_scene import create_scene
from validation import gazebo_offline_run
from validation import gazebo_experiment
from validation.gazebo_experiment import (
    create_experiment, finalize_experiment, initialize_experiment, load_experiment, sha256,
    verify_results, write_results_manifest,
)
from visualization.gazebo_offline_view import create_viewer

RESULTS = Path(__file__).resolve().parents[1] / "results" / "gazebo_offline"
STRAIGHT = RESULTS / "constant_velocity"
TURN = RESULTS / "smooth_turn"
PROCESSING_CONFIG = Path(__file__).resolve().parents[1] / "gazebo" / "processing_config.json"


def _inputs_only(source: Path, target: Path) -> Path:
    target.mkdir()
    for name in ("gazebo_state.csv", "manifest.json", "experiment.json"):
        shutil.copy2(source / name, target / name)
    return target


def _current_code_inputs(source: Path, target: Path, config_path: Path) -> Path:
    """Create a new experiment for current code from archived recording bytes."""
    target.mkdir()
    for name in ("gazebo_state.csv", "manifest.json"):
        shutil.copy2(source / name, target / name)
    archived = load_experiment(source)
    config_path.write_text(json.dumps({"schema_version": 1, "processing": archived["processing"]}))
    initialize_experiment(target, config_path)
    return target


def test_replay_uses_saved_audio_configuration_after_scene_json_changes(tmp_path, monkeypatch):
    directory = _current_code_inputs(STRAIGHT, tmp_path / "copy", tmp_path / "frozen-config.json")
    config = json.loads(gazebo_offline.CONFIG_PATH.read_text())
    config["snr_db"], config["seed"] = -17, 123456
    changed_scene = tmp_path / "scene.json"
    changed_scene.write_text(json.dumps(config))
    monkeypatch.setattr(gazebo_offline, "CONFIG_PATH", changed_scene)
    expected = load_experiment(directory)["processing"]["audio"]

    class ReachedAudio(Exception):
        pass

    def observe_audio(*args, **kwargs):
        assert kwargs["snr_db"] == expected["snr_db"]
        assert kwargs["seed"] == expected["base_seed"]
        assert kwargs["sampling_rate_hz"] == expected["sampling_rate_hz"]
        raise ReachedAudio

    monkeypatch.setattr(gazebo_offline_run, "synthesize_multistation_audio", observe_audio)
    with pytest.raises(ReachedAudio):
        gazebo_offline_run.process_recording(directory)


def test_run_id_is_portable_and_json_key_order_independent(tmp_path):
    directory = _inputs_only(STRAIGHT, tmp_path / "moved")
    experiment = load_experiment(directory)
    reordered = dict(reversed(list(json.loads(json.dumps(experiment)).items())))
    reordered["processing"] = dict(reversed(list(experiment["processing"].items())))
    reordered["recording"]["manifest"] = dict(reversed(list(experiment["recording"]["manifest"].items())))
    reordered["recording"]["manifest_sha256"] = "0" * 64
    assert finalize_experiment(reordered)["run_id"] == experiment["run_id"]
    reordered["recording"]["manifest_sha256"] = experiment["recording"]["manifest_sha256"]
    (directory / "experiment.json").write_text(json.dumps(reordered, indent=4))
    assert load_experiment(directory)["run_id"] == experiment["run_id"]
    assert finalize_experiment(reordered)["run_id"] == experiment["run_id"]


def test_code_hash_uses_posix_names_and_checks_source_bytes(tmp_path, monkeypatch):
    relative_windows = gazebo_experiment._source_name(
        PureWindowsPath(r"C:\checkout\validation\example.py"),
        PureWindowsPath(r"C:\checkout"))
    relative_posix = gazebo_experiment._source_name(
        PurePosixPath("/checkout/validation/example.py"),
        PurePosixPath("/checkout"))
    assert relative_windows == relative_posix == "validation/example.py"
    (tmp_path / "validation").mkdir()
    source = tmp_path / "validation" / "example.py"
    source.write_text("value = 1\n")
    monkeypatch.setattr(gazebo_experiment, "ROOT", tmp_path)
    digest = gazebo_experiment.code_sha256()
    source.write_text("value = 2\n")
    assert gazebo_experiment.code_sha256() != digest
    directory = _inputs_only(STRAIGHT, tmp_path / "code-check")
    with pytest.raises(ValueError, match="code SHA-256"):
        load_experiment(directory, check_code=True)


def test_fresh_gazebo_recording_init_process_viewer(tmp_path):
    directory = tmp_path / "fresh-gazebo"
    directory.mkdir()
    for name in ("gazebo_state.csv", "manifest.json"):
        shutil.copy2(STRAIGHT / name, directory / name)
    recorded_sha = sha256(directory / "gazebo_state.csv")
    assert not (directory / "summary.json").exists()
    assert not (directory / "bearing_results.csv").exists()
    config = json.loads(PROCESSING_CONFIG.read_text())
    config["processing"]["audio"]["duration_s"] = 0.3
    config["processing"]["audio"]["base_seed"] = 91234
    config_path = tmp_path / "explicit-processing.json"
    config_path.write_text(json.dumps(config))
    command = [sys.executable, "-m", "validation.gazebo_offline_run", "init",
               str(directory), "--processing-config", str(config_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    frozen = load_experiment(directory, check_code=True)
    assert frozen["processing"]["audio"]["base_seed"] == 91234
    assert frozen["processing"]["audio"]["duration_s"] == 0.3
    assert frozen["provenance"]["processing_config_sha256"] == sha256(config_path)
    assert not (directory / "summary.json").exists()
    config_path.write_text("{}")
    summary = gazebo_offline_run.process_recording(directory)
    assert summary["run_id"] == frozen["run_id"]
    assert summary["seed"] == 91234
    assert all(item["first_confirmation_time_s"] is None
               for item in summary["methods"].values())
    assert verify_results(directory)["run_id"] == frozen["run_id"]
    assert create_viewer(directory).exists()
    assert sha256(directory / "gazebo_state.csv") == recorded_sha
    with pytest.raises(FileExistsError, match="experiment.json"):
        initialize_experiment(directory, PROCESSING_CONFIG)


def test_scene_generation_refuses_existing_recording(tmp_path):
    directory = tmp_path / "accepted"
    directory.mkdir()
    (directory / "gazebo_state.csv").write_text("accepted recording\n")
    with pytest.raises(FileExistsError, match="not empty"):
        create_scene("constant_velocity", directory)
    assert (directory / "gazebo_state.csv").read_text() == "accepted recording\n"


def test_init_rejects_geometry_mismatch_before_freezing(tmp_path):
    directory = tmp_path / "fresh"
    directory.mkdir()
    for name in ("gazebo_state.csv", "manifest.json"):
        shutil.copy2(STRAIGHT / name, directory / name)
    config = json.loads(PROCESSING_CONFIG.read_text())
    config["processing"]["stations"][1]["position_m"][0] += 1
    config_path = tmp_path / "wrong-stations.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="recorded Gazebo geometry"):
        initialize_experiment(directory, config_path)
    assert not (directory / "experiment.json").exists()


def test_code_refresh_preserves_recording_and_prior_run_provenance(tmp_path, monkeypatch):
    directory = _inputs_only(STRAIGHT, tmp_path / "refresh")
    previous = load_experiment(directory)
    for name in json.loads((STRAIGHT / "results_manifest.json").read_text())["files"]:
        shutil.copy2(STRAIGHT / name, directory / name)
    shutil.copy2(STRAIGHT / "results_manifest.json", directory / "results_manifest.json")
    monkeypatch.setattr(gazebo_experiment, "code_sha256", lambda: "1" * 64)
    refreshed = gazebo_experiment.refresh_code_identity(directory)
    assert refreshed["run_id"] != previous["run_id"]
    assert refreshed["recording"] == previous["recording"]
    assert refreshed["processing"] == previous["processing"]
    assert refreshed["provenance"]["code_refresh_history"][-1]["previous_run_id"] == previous["run_id"]
    assert load_experiment(directory, check_code=True)["run_id"] == refreshed["run_id"]
    with pytest.raises(ValueError, match="run process first"):
        verify_results(directory)


def test_run_id_changes_for_material_inputs(tmp_path):
    source = _current_code_inputs(STRAIGHT, tmp_path / "source", tmp_path / "source-config.json")
    original = load_experiment(source, check_code=True)
    for change in ("recording", "seed", "snr", "calibration"):
        modified = json.loads(json.dumps(original))
        if change == "recording":
            modified["recording"]["state_sha256"] = "0" * 64
        elif change == "seed":
            modified["processing"]["audio"]["base_seed"] += 1
        elif change == "snr":
            modified["processing"]["audio"]["snr_db"] += 1
        else:
            modified["processing"]["calibration"]["values"][0]["bias_rad"][0] += 1e-5
        assert finalize_experiment(modified)["run_id"] != original["run_id"]
    destination = tmp_path / "new"
    created = create_experiment(source, destination, seed=23, snr_db=5.0,
                                comparison_group_id="paired-example")
    assert created["run_id"] != original["run_id"]
    assert created["comparison_group_id"] == "paired-example"
    assert sha256(destination / "gazebo_state.csv") == sha256(STRAIGHT / "gazebo_state.csv")
    with pytest.raises(FileExistsError):
        create_experiment(source, destination, seed=24)


def test_straight_and_turn_event_ids_do_not_intersect():
    ids = []
    for directory in (STRAIGHT, TURN):
        experiment = load_experiment(directory)
        verify_results(directory, experiment)
        events = set()
        for method in experiment["processing"]["frontend"]["methods"]:
            with (directory / f"updates_{method}.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            assert rows and all(row["event_id"].startswith(experiment["run_id"] + "|") for row in rows)
            events.update(row["event_id"] for row in rows)
        ids.append(events)
    assert ids[0].isdisjoint(ids[1])


def test_changed_recording_and_mixed_results_are_rejected(tmp_path):
    directory = _inputs_only(STRAIGHT, tmp_path / "tampered")
    with (directory / "gazebo_state.csv").open("a") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="recording SHA-256"):
        load_experiment(directory)
    with pytest.raises(ValueError, match="recording SHA-256"):
        create_viewer(directory)

    directory = _inputs_only(STRAIGHT, tmp_path / "mixed")
    for name in ("summary.json", "bearing_results.csv", "results_manifest.json",
                 "tracking_all_6_equal_gcc_wls.csv", "tracking_equal_weight_srp_phat.csv",
                 "updates_all_6_equal_gcc_wls.csv", "updates_equal_weight_srp_phat.csv"):
        shutil.copy2(STRAIGHT / name, directory / name)
    shutil.copy2(TURN / "summary.json", directory / "summary.json")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        create_viewer(directory)
    # Even a newly calculated file hash cannot make a foreign summary valid.
    manifest_path = directory / "results_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["summary.json"] = sha256(directory / "summary.json")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different experiment"):
        create_viewer(directory)
    shutil.copy2(STRAIGHT / "summary.json", directory / "summary.json")
    shutil.copy2(TURN / "tracking_all_6_equal_gcc_wls.csv",
                 directory / "tracking_all_6_equal_gcc_wls.csv")
    manifest["files"]["summary.json"] = sha256(directory / "summary.json")
    manifest["files"]["tracking_all_6_equal_gcc_wls.csv"] = sha256(
        directory / "tracking_all_6_equal_gcc_wls.csv")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mixed experiment IDs"):
        create_viewer(directory)


def test_empty_update_log_replaces_stale_content_and_verifies(tmp_path):
    directory = _inputs_only(STRAIGHT, tmp_path / "empty")
    experiment = load_experiment(directory)
    run_id = experiment["run_id"]
    filenames = ["summary.json", "bearing_results.csv"]
    summary = {"run_id": run_id, "config_sha256": experiment["config_sha256"],
               "gazebo_state_sha256": experiment["recording"]["state_sha256"]}
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "bearing_results.csv").write_text("run_id,sequence_id\n" + run_id + "," + run_id + "\n")
    for method in experiment["processing"]["frontend"]["methods"]:
        tracking = f"tracking_{method}.csv"
        updates = f"updates_{method}.csv"
        (directory / tracking).write_text("run_id,sequence_id\n" + run_id + "," + run_id + "\n")
        (directory / updates).write_text("old,stale,data\n")
        gazebo_offline_run._write_result_csv(directory / updates, [],
                                             empty_columns=("run_id", "sequence_id", "event_id"))
        assert (directory / updates).read_text() == "run_id,sequence_id,event_id\n"
        filenames.extend((tracking, updates))
    write_results_manifest(directory, experiment, filenames)
    assert verify_results(directory)["run_id"] == run_id


def test_legacy_requires_explicit_migration(tmp_path):
    directory = tmp_path / "legacy"
    directory.mkdir()
    with pytest.raises(ValueError, match="migrate"):
        load_experiment(directory)


def test_replay_cli_rejects_silent_parameter_override():
    result = subprocess.run([sys.executable, "-m", "validation.gazebo_offline_run",
                             "process", str(STRAIGHT), "--seed", "99"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "require `new`" in result.stderr
