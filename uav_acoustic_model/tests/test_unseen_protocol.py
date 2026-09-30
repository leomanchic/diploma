"""Frozen pre-evaluation contract and flight-program shape checks."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from px4.finalize_recording import phase_intervals
from px4.run_flight import planned_command_route

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pre_evaluation_protocol_manifest_is_frozen_and_complete():
    manifest = json.loads((ROOT / "UNSEEN_MANOEUVRES_PROTOCOL_MANIFEST.json").read_text())
    assert manifest["base_commit_sha"] == "a2f8fd052cd38f7da5a21a0d5b9238f9bb90892a"
    assert _sha(ROOT / "UNSEEN_MANOEUVRES_PROTOCOL.md") == manifest["protocol_sha256"]
    assert manifest["audio_stream_count"] == 24
    assert manifest["tracker_run_count"] == 96
    assert len(manifest["matrix"]) == 24
    assert [spec["index"] for spec in manifest["matrix"]] == list(range(24))
    assert len({(spec["trajectory"], spec["source_class"], spec["distance_m"],
                 spec["replicate"]) for spec in manifest["matrix"]}) == 24
    assert manifest["smoke_index"] == 8
    assert manifest["matrix"][8]["source_class"] == "nonstationary_harmonic"
    assert manifest["matrix"][8]["distance_m"] == 700
    assert manifest["fixed_background"]["reference_snr_db"] == 10.0
    assert manifest["fixed_background"]["no_per_distance_normalization"]
    assert manifest["recorded_source_decision"]["replacement"] == "nonstationary_harmonic"
    assert manifest["processing"]["tracker"]["recovery"]["maximum_batch_optimizations_per_generation"] == 128
    assert manifest["variants"]["baseline"]["confirmation_station_count"] == 2
    assert manifest["variants"]["three_station_confirmation"]["confirmation_station_count"] == 3
    for item in manifest["flight_plans"].values():
        assert _sha(ROOT / item["path"]) == item["sha256"]
    for name, digest in manifest["flight_code_sha256"].items():
        assert _sha(ROOT / name) == digest


def test_paired_noise_and_source_seeds_are_disjoint_from_prior():
    manifest = json.loads((ROOT / "UNSEEN_MANOEUVRES_PROTOCOL_MANIFEST.json").read_text())
    old = json.loads((ROOT / "results/robust_track_confirmation/evaluation_manifest.json").read_text())
    specs = manifest["matrix"]
    for ti in (0, 1):
        for replicate in (0, 1):
            chosen = [s for s in specs if s["trajectory_index"] == ti and s["replicate"] == replicate]
            assert len({s["noise_seed"] for s in chosen}) == 1
    for ti in (0, 1):
        for source in (0, 1):
            chosen = [s for s in specs if s["trajectory_index"] == ti and s["source_index"] == source]
            assert len({s["source_seed"] for s in chosen}) == 1
    assert not ({s["noise_seed"] for s in specs} & {s["noise_seed"] for s in old["specs"]})
    assert not ({s["source_seed"] for s in specs} & {s["source_seed"] for s in old["specs"]})
    assert len({s["source_seed"] for s in specs}) == 4
    assert len({s["noise_seed"] for s in specs}) == 4
    for spec in specs:
        assert spec["source_seed"] == int(np.random.SeedSequence([
            20260930, 0, spec["trajectory_index"], spec["source_index"]
        ]).generate_state(1, dtype=np.uint64)[0])
        assert spec["noise_seed"] == int(np.random.SeedSequence([
            20260930, 1, spec["trajectory_index"], spec["replicate"]
        ]).generate_state(1, dtype=np.uint64)[0])


def test_new_flight_phases_have_distinct_order_and_planned_route_is_commands_only():
    for profile, phases in (
        ("spatial_manoeuvre", ["preflight", "takeoff", "hover_before", "climbing_turn",
                               "descending_turn", "exit", "hover_after", "land", "landed"]),
        ("radial_approach_depart", ["preflight", "takeoff", "hover_before", "approach",
                                    "radial_pause", "depart", "lateral_turn", "lateral",
                                    "hover_after", "land", "landed"]),
    ):
        rows = [{"flight_phase": name, "sim_time_s": float(index)}
                for index, name in enumerate(phases)]
        assert [row["name"] for row in phase_intervals(rows, 0.02, profile)] == phases
    plan = json.loads((ROOT / "px4/flight_plan_spatial_manoeuvre.json").read_text())
    commands = [
        {"command": "mavlink.velocity_setpoint", "sim_time_s": 10.0,
         "north": 0, "east": 2, "down": -0.5, "flight_phase": "climbing_turn"},
        {"command": "mavlink.velocity_setpoint", "sim_time_s": 12.0,
         "north": 1, "east": 0, "down": 0.5, "flight_phase": "descending_turn"},
    ]
    route = planned_command_route(plan, commands, 20.0)
    assert route[1]["x_m"] == 39.0
    assert route[1]["z_m"] == 17.0
    assert route[-1]["z_m"] == 0.0


def test_frozen_protocol_checkout_preserves_exact_bytes_on_windows():
    result = subprocess.run(
        ['git', 'check-attr', 'text', '--', 'UNSEEN_MANOEUVRES_PROTOCOL.md'],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip().endswith('text: unset')
