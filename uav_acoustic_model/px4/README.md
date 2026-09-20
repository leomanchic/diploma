# PX4 X500 → Gazebo state → three-station acoustic pilot

This pilot uses PX4 v1.17.0 to control an X500. Gazebo Harmonic supplies the
physical motion; a read-only world plugin observes the X500 `base_link` origin.
The acoustic source is still the existing synthetic broadband approximation.
Rotor sound from motor RPM, field detection range, wind and reflections are not
modelled. No ROS 2 or prescribed-motion plugin controls this aircraft.

## Tested environment and setup

Ubuntu 24.04.5 under WSL2, Gazebo Harmonic `gz sim 8.15.0`, PX4 tag `v1.17.0`
at `d6f12ad1c4f70ad3230afd7d86e971421e02fef4` with Gazebo model
submodule `b6127f4ec20de867e215fb5f78ae88b80f371909`. Each recording's
`preflight.json` contains the complete recursive submodule status and hashes
of the exact world and X500 model. The checked tag's 39 submodule SHAs are
also in [`submodules-v1.17.0.txt`](submodules-v1.17.0.txt). The flight client uses
MAVSDK-Python `3.10.0` for actions, telemetry and parameters, and PyMAVLink
`2.4.49` for the Offboard setpoint stream and mode command. PX4 confirms
Offboard mode before straight flight begins. The pinned MAVSDK server returned
`NO_SETPOINT_SET` for both of its Offboard setpoint RPCs in two separate local
attempts; their failed, unfinalized recordings remain locally as `pilot_001`
and `pilot_002`. The accepted `pilot_003` uses direct MAVLink setpoints. The
acoustic `.venv` does not need PX4 flight dependencies.

The PX4 v1.17.0 `Tools/setup/ubuntu.sh` installs firmware toolchains by
default. This SITL pilot needs no NuttX or hardware flashing tools. Install
the system build dependency once:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends libopencv-dev
```

Prepare the separate PX4 tree and Python environment:

```bash
cd ~/projects/PX4-Autopilot
git checkout v1.17.0
git submodule update --init --recursive
PX4_PATCH=~/projects/diploma-gazebo/uav_acoustic_model/px4/px4-v1.17.0-minimal-sitl.patch
if ! git apply --reverse --check "$PX4_PATCH" 2>/dev/null; then
  git apply "$PX4_PATCH"
fi
python3 -m venv .venv-px4
.venv-px4/bin/python -m pip install -r Tools/setup/requirements.txt ninja
.venv-px4/bin/python -m pip install -r ~/projects/diploma-gazebo/uav_acoustic_model/px4/requirements-flight.txt
.venv-px4/bin/python -m pip check
PATH="$PWD/.venv-px4/bin:$PATH" HEADLESS=1 make px4_sitl gz_x500
```

The patch removes only the unused uXRCE-DDS client and optical-flow Gazebo
plugin from the local SITL build: this environment cannot fetch their external
sources from GitHub. The exact patch SHA and applied status are recorded in
the flight manifest. The upstream PX4 Git commit remains v1.17.0. The last
command is PX4's standard X500 target with a headless Gazebo client. Check
`Gazebo world is ready`, `gz_bridge` for `x500_0`, and
`Startup script returned successfully`. The default world may warn about
unused camera/optical-flow plugins. End with Ctrl+C before the custom scene.
To see the default scene, omit `HEADLESS=1` if WSLg works.

## New recording: three terminals

Set a **new, empty** `RUN_DIR` for every flight. `create_scene.py` refuses a
nonempty directory. The scene is based on PX4's tagged default world: flat
ground `z=0`; array centres S0 `(0,0,5)`, S1 `(100,0,10)`, S2 `(10,90,3)` m;
X500 spawns at `(35,35,0)` m away from all stations. These positions and the
flight settings live in [`flight_plan.json`](flight_plan.json). Edit a copy of
that file **before** scene creation for another experiment.

First, from the acoustic project root:

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
export RUN_DIR="$PWD/results/px4_flight/pilot_004"
export PX4_ROOT="$HOME/projects/PX4-Autopilot"
cmake -S px4 -B build/px4-observer
cmake --build build/px4-observer -j2
.venv/bin/python -m px4.create_scene "$RUN_DIR" --px4-root "$PX4_ROOT"
```

Terminal A starts Gazebo **with its GUI** and passes the seed from the frozen
flight plan. The launcher writes `gazebo_launch.json` with the exact arguments
and environment before starting Gazebo. Add `--headless` for server-only mode:

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
export RUN_DIR="$PWD/results/px4_flight/pilot_004"
export PX4_ROOT="$HOME/projects/PX4-Autopilot"
.venv/bin/python -m px4.launch_gazebo "$RUN_DIR" --px4-root "$PX4_ROOT"
```

Terminal B starts the **already built** PX4 autopilot and spawns X500 in that
existing world. `-d` turns off the interactive PX4 shell, keeping the log
small:

```bash
cd ~/projects/PX4-Autopilot
GZ_IP=127.0.0.1 PX4_SIM_MODEL=gz_x500 \
PX4_GZ_STANDALONE=1 PX4_GZ_WORLD=px4_acoustic \
PX4_GZ_MODEL_POSE=35,35,0,0,0,0 \
build/px4_sitl_default/bin/px4 -d
```

Terminal C flies through MAVLink after PX4 reports healthy position:

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
export RUN_DIR="$PWD/results/px4_flight/pilot_004"
~/projects/PX4-Autopilot/.venv-px4/bin/python -m px4.run_flight "$RUN_DIR"
```

The fixed program arms and takes off to 16 m, stabilizes, flies East at 3 m/s
for 5 s, rotates its velocity smoothly through approximately 90° over 5 s,
holds again and lands. Hover periods are 3 s each.
The program explicitly sets PX4 `COM_RC_IN_MODE=4` for MAVLink control without
an RC transmitter and records the read-back value plus position-control speed
and acceleration limits in `autopilot_parameters.json`. MAVSDK arms, takes off,
reads PX4 telemetry and lands. PyMAVLink sends only the Offboard velocity/yaw
setpoints and mode command on PX4's GCS UDP 14550 port. The observer writes
`gazebo_state.csv` from post-physics Gazebo state at 50 Hz; it does not command
pose. `flight_commands.csv` and `planned_route.csv` preserve requests separately.
The flight program writes phase markers; the observer attaches each marker to
the next simulation-time sample. Stop PX4 and Gazebo with Ctrl+C after landing.

Then freeze the recording and process it **offline**, from the acoustic root:

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
export RUN_DIR="$PWD/results/px4_flight/pilot_004"
.venv/bin/python -m px4.finalize_recording "$RUN_DIR"
.venv/bin/python -m px4.assess_recording "$RUN_DIR"
.venv/bin/python -m px4.create_probe "$RUN_DIR" "${RUN_DIR}_probe"
.venv/bin/python -m validation.gazebo_offline_run init "${RUN_DIR}_probe" --processing-config "${RUN_DIR}_probe/probe_processing_config.json"
/usr/bin/time -v .venv/bin/python -m validation.gazebo_offline_run process "${RUN_DIR}_probe"
.venv/bin/python -m validation.gazebo_offline_run init "$RUN_DIR" --processing-config "$RUN_DIR/processing_config.json"
.venv/bin/python -m validation.gazebo_offline_run process "$RUN_DIR"
.venv/bin/python -m visualization.gazebo_offline_view "$RUN_DIR"
xdg-open "$RUN_DIR/viewer.html"
```

`finalize_recording` checks phase order, takeoff/landing, pose, quaternion,
velocity, duplicates, time resets and gaps. It selects the reception interval
from the **predeclared flight phases**, before localization output exists: from
0.5 s before the first hover to 0.5 s after the second. Expected audio is
about 17 s at 48 kHz, with one fixed source/noise seed and AWGN +10 dB.
The previous GCC/SRP, calibration and tracker settings are copied from
`gazebo/processing_config.json` and frozen by `init`. Do not use the old
2.5–3.5 s manoeuvre labels for this flight. The viewer checks all input/result
hashes and shows the planned and observed paths, acoustic estimate, missing
intervals, phase boundaries and per-phase descriptive statistics.

## Coordinates and repeatability

Gazebo state is world ENU in metres and seconds, quaternion `w,x,y,z`, and
observed world velocity at the X500 canonical `base_link` origin. The source
has zero offset from that link. PX4 uses world NED and body FRD. For world
vectors `(N,E,D)=(y,x,-z)`; body `(F,R,D)=(x,-y,-z)` from Gazebo FLU.
`px4/coordinates.py` and its tests cover axis, sign and quaternion mapping.
All exported timestamps are simulation time. Pause or playback rate cannot
alter the physical timestamps. Reprocessing a saved recording checks its SHA
and reproduces the acoustic result; a second physical flight is not claimed
to be byte-identical.

## Accepted one-flight result

[`pilot_003`](../results/px4_flight/pilot_003) is the completed local PX4
flight. Its CSV SHA-256 is
`6f882b300b24c9204ffdd9b8a2df11e68954dfa02a36a59c8442ab1e8edf12ea`;
its acoustic `run_id` is `gzrun-f0aa319adf342003c9d9f535`. The observer
recorded 3954 post-step samples over simulation time 10.964–90.024 s with
maximum gap 0.02000000000001 s. X500 reached 16.079 m and landed at 0.227 m
above world `z=0` (its `base_link` origin). The straight segment moved 12.98 m
East; the turn moved 6.99 m North and ended with observed North velocity above
3 m/s. A 50 Hz cubic spline's velocity differs from the independent Gazebo
velocity by 0.00683 m/s RMS; downsampling this same flight to 25 Hz raises
that to 0.00907 m/s RMS. This is a cadence check, not a second flight.
The `pilot_003` Gazebo process was started directly before this launcher
existed: `gazebo_seed_requested` is `20260920`, while
`gazebo_seed_applied` is `null`. The original CSV is unchanged; corrected
metadata and a repeated acoustic processing run yielded zero numeric change
in all 9564 bearing rows, 300 tracking rows and 280 update rows. New runs
through Terminal A record the applied seed.
An independent `init → process → viewer` from a copy of the corrected frozen
inputs under `/tmp` kept the **same** `run_id`, method metrics and all
non-runtime numerical fields (maximum difference 0, tolerance `1e-12`).
The per-frame measured wall runtimes changed, as expected. The audit is in
[`portable_replay_comparison.json`](../results/px4_flight/pilot_003/portable_replay_comparison.json).

The frozen audio interval is 17.02 s at 48 kHz, with seed `20260920`, AWGN
`+10 dB`, and the unchanged GCC/SRP/calibration/tracker settings. The one-second
copy probe took 4.98 s and 180 MB peak resident memory; the main processing
took 54.81 s and 1.11 GB. Both completed successfully.

| Method | Valid publications | Confirmation (sim s) | Conditional RMSE | Accepted / rejected updates | Track losses |
|---|---:|---:|---:|---:|---:|
| GCC-PHAT / WLS | 141/150 (94%) | 47.0193125 | 0.2488 m | 138 / 2 | 0 |
| SRP-PHAT | 141/150 (94%) | 47.0193125 | 0.2488 m | 138 / 2 | 0 |

Both methods had 100% valid publications in the straight, turn and second
hover phases. In the first hover, each had 24/27; the viewer preserves those
missing estimates and shows the initialization failures. Both rejected updates
were `emission_outside_history`. Open the completed result with:

```bash
cd ~/projects/diploma-gazebo/uav_acoustic_model
xdg-open results/px4_flight/pilot_003/viewer.html
```

This HTML visualizes the saved flight; opening `scene.sdf` alone starts an
empty scene until PX4 spawns X500 and Terminal C flies it. The pilot uses a
synthetic broadband acoustic source and one seed. These numbers are an
integration result, not a field accuracy or population coverage estimate.

Portable CI runs Python tests with a synthetic flight-shaped fixture and a
hash/physics check of this committed recording. The PX4 build and physical
takeoff/landing are local Linux system gates. Run a short acoustic probe in a
**copy** of each new recording before its main processing. Never overwrite an
accepted run.
