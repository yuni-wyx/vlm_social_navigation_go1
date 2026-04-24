# Motion Control — Unitree Go1

> Real-time subsumption-based velocity control with LiDAR safety, hallway centering, and stuck-recovery behaviors. Uses a split-architecture: laptop runs the control logic over ROS, NX1 runs the SDK bridge to the motor controller.

## Architecture

```
 ┌──────────────────────────────────┐           ┌──────────────────────────────┐
 │  LAPTOP (192.168.123.67)         │  UDP:9900 │  NX1 (192.168.123.15)        │
 │                                  │ ────────> │                              │
 │  subsumption_controller.py       │  (vx,vy,  │  sdk_udp_bridge.py           │
 │  ├─ Subscribes /scan_odom        │   wz)     │  ├─ Receives UDP packets     │
 │  ├─ Runs P1/P2/P3/P4 logic      │           │  ├─ Sets SDK HighCmd         │
 │  └─ Sends velocity via UDP       │           │  └─ Sends to MCU via SDK     │
 │                                  │           │                              │
 │  ROS_MASTER=Pi:11311             │           │  MCU (192.168.12.1:8082)     │
 │  Reads: /scan_odom (LaserScan)   │           │  Unitree SDK (Python 3.7)    │
 └──────────────────────────────────┘           └──────────────────────────────┘
```

## Quick Start

### 1. Start LiDAR + Relay (on laptop)
```bash
cd ~/dog_ws/streaming
bash launch.sh lidar    # Starts LiDAR on NX1
bash launch.sh relay    # Relays /scan → /scan_odom with frame fixes
```

### 2. Start SDK Bridge (on NX1)

```bash
# One-time: add route to MCU subnet
sudo ip route add 192.168.12.0/24 dev eth0

# Start the bridge
python3.7 /path/to/sdk_udp_bridge.py
```

Or from the laptop via SSH:
```bash
export NX1_HOST=unitree@192.168.123.15

ssh "$NX1_HOST" "sudo ip route add 192.168.12.0/24 dev eth0"
scp scripts/sdk_udp_bridge.py "$NX1_HOST":/tmp/
ssh "$NX1_HOST" \
    "nohup python3.7 /tmp/sdk_udp_bridge.py > /tmp/bridge.log 2>&1 &"
```

### 3. Start Controller (on laptop)
```bash
export ROS_MASTER_URI=http://192.168.123.161:11311
export ROS_IP=192.168.123.67
python3 scripts/subsumption_controller.py
```

**Ctrl+C stops the robot immediately.**

## Social Navigation MVP

This repo now also includes a small `social_nav_controller.py` that reuses the
same laptop-side ROS subscription path and NX1 UDP bridge as the existing
controller.

### What it does

- `baseline` mode:
  - reads `/scan_odom`
  - computes a robust front obstacle distance from the front LiDAR sector
  - drives forward unless that distance is below `baseline_stop_dist`
- `human_aware` mode:
  - reads `/scan_odom` and `/camera_face/left/image_raw`
  - runs a lightweight OpenCV person detector on the front camera
  - if a person is centered in front, switches to `social_stop_dist`
  - otherwise keeps `baseline_stop_dist`

This is intentionally a simple early-stop behavior:
- no map
- no planner
- no multi-human tracking
- no trajectory prediction
- no low-level control changes on NX1

## Offline Evaluation Pipeline

For the final-project stop-and-think evaluation, the repo now includes an
offline pipeline that compares:

- geometry-only baseline
- single-image VLM baseline
- sequence-image VLM method

The pieces are intentionally separated from the live ROS controller:

- `streaming/scripts/extract_social_nav_data.py`
  - extracts front-camera frames plus optional front-distance metadata from one bag
- `motion_control/scripts/social_nav_eval.py`
  - builds samples, writes label templates, runs predictions, and computes metrics
- `motion_control/eval/sample_labels_template.csv`
  - lightweight CSV example for manual ground-truth annotation

The offline evaluator now distinguishes:

- primary benchmark methods
  - `geometry`
  - `internvl_single_image_navigation`
  - `internvl_sequence_image_navigation`
  - `qwen_single_image_navigation`
  - `qwen_sequence_image_navigation`
- legacy/debug methods
  - kept available only when explicitly requested for ablation/debugging

This change is intentional: the InternVL legacy `/analyze` and
`structured_localization` prompts produced unstable no-person false positives,
so they are excluded from the default formal benchmark suite.

Note for the current 13-scenario rosbag set:

- these bags were recorded without LiDAR enabled
- the extracted datasets therefore do not contain the `/scan_odom`-derived
  front distance used by the geometry baseline
- geometry is still kept in the default suite for API consistency, but its
  predictions will be marked as unavailable with `missing_front_dist`
  rather than treated as a meaningful score

For rosbag-based experiments, the repo also includes a scenario manifest:

- `motion_control/eval/scenario_manifest.json`
  - maps each benchmark bag to scenario metadata such as expected action,
    human presence, motion, notes, whether it is a primary case, and the
    standard extracted dataset directory for that bag

This lets the evaluator batch over the full rosbag scenario set while keeping
the original single-bag / extracted-data flow intact.

Example flow:

```bash
# 1. Extract images + metadata from a fixed bag
python3 streaming/scripts/extract_social_nav_data.py \
  streaming/bags/myrun_fixed.bag \
  /tmp/social_nav_eval

# 2. Build single-image and sequence sample sets
python3 motion_control/scripts/social_nav_eval.py build-samples \
  /tmp/social_nav_eval/frames.jsonl \
  /tmp/social_nav_eval/single_samples.jsonl \
  --input-type single

python3 motion_control/scripts/social_nav_eval.py build-samples \
  /tmp/social_nav_eval/frames.jsonl \
  /tmp/social_nav_eval/sequence_samples.jsonl \
  --input-type sequence \
  --sequence-length 5

# 3. Generate a label template, fill in ground_truth_action later
python3 motion_control/scripts/social_nav_eval.py write-label-template \
  /tmp/social_nav_eval/sequence_samples.jsonl \
  /tmp/social_nav_eval/labels.csv

# 4. Run the default primary benchmark suite
#    Assumes InternVL wrapper at http://localhost:8100
#    and Qwen wrapper at http://localhost:8101
python3 motion_control/scripts/social_nav_eval.py run-benchmark \
  --single-samples-jsonl /tmp/social_nav_eval/single_samples.jsonl \
  --sequence-samples-jsonl /tmp/social_nav_eval/sequence_samples.jsonl \
  --output-dir /tmp/social_nav_eval/predictions \
  --geometry-stop-threshold 1.0

# 5. Compare the primary methods only (default reporting behavior)
python3 motion_control/scripts/social_nav_eval.py evaluate \
  --labels-csv /tmp/social_nav_eval/labels.csv \
  --predictions-jsonl \
    /tmp/social_nav_eval/predictions/geometry.jsonl \
    /tmp/social_nav_eval/predictions/internvl_single_image_navigation.jsonl \
    /tmp/social_nav_eval/predictions/internvl_sequence_image_navigation.jsonl \
    /tmp/social_nav_eval/predictions/qwen_single_image_navigation.jsonl \
    /tmp/social_nav_eval/predictions/qwen_sequence_image_navigation.jsonl \
  --output-dir /tmp/social_nav_eval/results \
  --safety-distance-threshold 1.0

# Optional: include legacy/debug prompts for ablation
python3 motion_control/scripts/social_nav_eval.py run-benchmark \
  --single-samples-jsonl /tmp/social_nav_eval/single_samples.jsonl \
  --sequence-samples-jsonl /tmp/social_nav_eval/sequence_samples.jsonl \
  --output-dir /tmp/social_nav_eval/predictions_all \
  --geometry-stop-threshold 1.0 \
  --include-legacy-prompts

python3 motion_control/scripts/social_nav_eval.py evaluate \
  --labels-csv /tmp/social_nav_eval/labels.csv \
  --predictions-jsonl /tmp/social_nav_eval/predictions_all/*.jsonl \
  --output-dir /tmp/social_nav_eval/results_all \
  --safety-distance-threshold 1.0 \
  --include-legacy-prompts
```

Manifest-driven rosbag batch run:

```bash
python3 motion_control/scripts/run_social_nav_benchmark.py \
  --run-name go1_social_nav_apr18 \
  --sequence-length 5 \
  --geometry-stop-threshold 1.0
```

This will create a reproducible run directory under:

- `benchmark_runs/go1_social_nav_apr18/`

and, for each scenario in the manifest:
- extract front-camera frames and optional front distance from the listed bag
- build single-image and sequence samples
- run the default primary benchmark suite
- save per-scenario artifacts under `benchmark_runs/<run_name>/<bag_id>/`
- write aggregate results to:
  - `aggregate_results.json`
  - `aggregate_results.csv`
  - `primary_cases_summary.csv`
  - `review_cases_summary.csv`

Use `--include-legacy-prompts` if you want the legacy/debug methods added to
the batch run as well:

```bash
python3 motion_control/scripts/run_social_nav_benchmark.py \
  --run-name go1_social_nav_apr18_all \
  --sequence-length 5 \
  --geometry-stop-threshold 1.0 \
  --include-legacy-prompts
```

Fully decoupled extracted-data workflow:

1. Extract all bags once in a ROS environment:

```bash
python3 motion_control/scripts/prepare_social_nav_extracted.py \
  --manifest motion_control/eval/scenario_manifest.json \
  --output-root extracted_social_nav
```

This creates one dataset per bag:

- `extracted_social_nav/<bag_id>/`
  - `images/`
  - `frames.jsonl`
  - `extraction_summary.json`

Because the current bag set has no LiDAR scan topic, those extracted datasets
will support the VLM methods but not produce a usable geometry baseline.

2. Run the full benchmark later without ROS, using only extracted data:

```bash
python3 motion_control/scripts/run_social_nav_benchmark.py \
  --run-name go1_social_nav_extracted_apr18 \
  --input-mode extracted \
  --sequence-length 5 \
  --geometry-stop-threshold 1.0
```

If you prefer not to store `extracted_dir` in the manifest, you can point the
runner at a root directory containing `<root>/<bag_id>/` datasets:

```bash
python3 motion_control/scripts/run_social_nav_benchmark.py \
  --run-name go1_social_nav_extracted_apr18 \
  --input-mode extracted \
  --extracted-root extracted_social_nav \
  --sequence-length 5 \
  --geometry-stop-threshold 1.0
```

Current wrapper ports for the default benchmark:

- InternVL wrapper: `http://localhost:8100`
- Qwen wrapper: `http://localhost:8101`

The runner assumes the current wrapper ports:
- InternVL wrapper: `http://localhost:8100`
- Qwen wrapper: `http://localhost:8101`

The VLM prompt templates used for offline evaluation are defined centrally in
`social_nav_eval_prompts.py`, and `vlm_wrapper.py` now exposes a separate
`/analyze_navigation` endpoint for single-image and multi-image offline runs
without changing the existing `/analyze` ROS-facing behavior.

The `predict` subcommand still supports the older `single_vlm` and
`sequence_vlm` aliases for backward compatibility, but the formal benchmark
should use `run-benchmark` so the method set and reporting categories stay
consistent.

### Topics reused

| Topic | Type | Role |
|-------|------|------|
| `/scan_odom` | `sensor_msgs/LaserScan` | Front obstacle distance |
| `/camera_face/left/image_raw` | `sensor_msgs/Image` | Person detection in `human_aware` mode |
| `UDP :9900` to NX1 | packed `(vx, vy, wz)` floats | Same motion path as the existing controller |

### Run it

Start sensors and relay first:

```bash
cd streaming
./launch.sh lidar
./launch.sh relay
./launch.sh cameras   # needed for human_aware mode
```

Make sure the NX1 bridge is running:

```bash
python3.7 motion_control/scripts/sdk_udp_bridge.py
```

Then, on the laptop:

```bash
cd motion_control
./launch_social_nav.sh baseline
./launch_social_nav.sh human_aware
```

### Parameters

These are ROS private params accepted by `social_nav_controller.py`:

| Param | Default | Meaning |
|-------|---------|---------|
| `~mode` | `baseline` | `baseline` or `human_aware` |
| `~baseline_stop_dist` | `0.6` | Default stop threshold |
| `~social_stop_dist` | `1.2` | Larger stop threshold when a person is in front |
| `~front_sector_half_angle_deg` | `10.0` | Half-width of the front LiDAR sector |
| `~front_distance_percentile` | `20.0` | Robust front distance percentile over valid front ranges |
| `~person_center_fraction` | `0.4` | Width of the "front" image region as a fraction of image width |

Example with overrides:

```bash
./launch_social_nav.sh human_aware \
  _baseline_stop_dist:=0.7 \
  _social_stop_dist:=1.4 \
  _front_sector_half_angle_deg:=12.0
```

Logs are written to `/tmp/social_nav.log` with:
- timestamp
- mode
- front distance
- active stop threshold
- person detection flags
- published action (`FORWARD` or `STOP`)

## Behavior Stack (Subsumption)

| Priority | Behavior | Trigger | Action |
|----------|----------|---------|--------|
| **P1** | Safety Stop | Front (±10°) < 1.5m for 3+ frames | Stop forward, keep angular. If stuck >8 frames: rotate in place at 0.15 rad/s to find open direction |
| **P2** | Wall Avoid | Either wall < 0.7m | Steer away at ±0.12 rad/s, slow to 70% speed |
| **P3** | Center | Both walls < 3.0m, imbalance > 0.1m | Proportional centering (gain=0.10, max vz=0.10) |
| **P4** | Forward | Default | Drive forward at 0.2 m/s |

## Challenges & Solutions

### 1. Cannot Use SDK Directly from Laptop
**Problem:** The Unitree SDK communicates with the MCU at `192.168.12.1` via UDP. The laptop is on `192.168.123.x` — different subnet, no route.

**Solution:** Run the SDK bridge on NX1 (`192.168.123.15`), which is on the same physical Ethernet as the MCU. The laptop sends commands via a lightweight UDP socket to NX1.

### 2. NX1 Cannot Reach MCU (192.168.12.1)
**Problem:** NX1 has no default route to the `192.168.12.0/24` subnet even though it's on the same Ethernet.

**Solution:** Add a static route: `sudo ip route add 192.168.12.0/24 dev eth0`. This must be re-added after every reboot/battery-swap.

### 3. NX1 Python Version Incompatibility
**Problem:** NX1 runs Ubuntu 18.04 with Python 3.6.9 as system python. The Unitree SDK .so files are compiled for Python 3.7/3.8.

**Solution:** Use `python3.7` explicitly (installed separately on NX1 at `/usr/bin/python3.7`).

### 4. /cmd_vel Doesn't Move the Robot
**Problem:** The built-in `ros2udp_motion_mode_adv` node subscribes to `/cmd_vel` but publishing Twist messages to it had no effect.

**Solution:** Bypass ROS entirely for motor commands. The SDK bridge sends `HighCmd` directly to the MCU with `mode=2` (walk) and `gaitType=1` (trot).

### 5. SSH Compound Commands Fail (Exit 255)
**Problem:** SSH commands with semicolons and `nohup` fail with exit code 255 when too complex.

**Solution:** Either (a) use `scp` to deploy a shell script and run it, or (b) break compound commands into separate SSH calls.

### 6. Battery Swap Clears /tmp on NX1
**Problem:** After a battery swap (power cycle), all files in `/tmp` on NX1 are deleted.

**Solution:** Re-deploy scripts via `scp` after each reboot. The launch sequence handles this.

### 7. Front Sector Too Wide → False P1 Stops
**Problem:** With ±30° front sector, the diagonal projection of side walls (at ~0.7m) appeared within the front sector at ~1.1m, triggering false safety stops in narrow hallways.

**Evolution:**
- ±30° → ±20° → ±10° (final). At ±10°, only truly frontal obstacles trigger P1.

### 8. P1 Safety Stop Freezes Robot Completely
**Problem:** When P1 fired, the robot stopped all motion (vx=0, vz=0) — it couldn't steer away from the obstacle.

**Solution:** P1 now only stops forward motion but preserves angular corrections from P2. The robot turns away from the wall while stationary. After 8 frames stuck, it rotates in place to scan for the open direction.

### 9. P2 Drift Correction Causes Circling
**Problem:** The original "drift toward right wall" logic (from the hallway-following controller) caused the robot to steer hard right in open spaces (R=3-10m), resulting in circles.

**Evolution:**
- First fix: cap drift to HALLWAY_MAX=2.0m (only drift if R < 2m)
- Second fix: reduce VZ_MAX from 0.30 to 0.12
- Final fix: remove drift entirely, replace with centering (equalize R and L)

### 10. Multiple Controllers Running Simultaneously
**Problem:** Launching a new controller without killing the old one resulted in two processes sending conflicting commands (one sending vz=-0.30, the other vz=0.00).

**Solution:** Always `pkill -f` the old controller before launching a new one. The current script uses a unique node name (`subsumption_controller`) for easy cleanup.

### 11. LaserScan Noise / Transient Readings
**Problem:** Raw LiDAR readings contain transient spikes and noise that cause erratic behavior.

**Solution:** Apply a median filter (kernel size 5) to smooth the scan data. Invalid readings (inf/nan) are replaced with 10.0m.

### 12. Robot Still Drifts Right
**Known issue:** Even with centering (gain=0.10, max vz=0.10), the robot still drifts slightly right over time. Possible causes:
- Mechanical bias in the Go1's gait
- LiDAR mounting offset
- Centering gain still too conservative

**Next steps:** Increase centering gain, add integral term to correct persistent drift, or use odometry feedback.

## Files

| File | Runs on | Description |
|------|---------|-------------|
| `scripts/subsumption_controller.py` | Laptop | Main controller: reads /scan_odom, runs behavior stack, sends UDP |
| `scripts/sdk_udp_bridge.py` | NX1 | Receives UDP velocity commands, forwards to Go1 MCU via SDK |
| `scripts/go1_navigate.py` | NX1 | Simple file-based navigator for open-loop commands (rotate, forward) |

## Network Reference

| Host | IP | Role |
|------|----|------|
| Laptop | 192.168.123.67 | Controller, ROS subscriber |
| Pi (rosmaster) | 192.168.123.161 | ROS master, runs ros2udp |
| NX1 | 192.168.123.15 | LiDAR, SDK bridge |
| NX2 | 192.168.123.14 | Cameras |
| NX3 | 192.168.123.13 | Cameras |
| MCU | 192.168.12.1 | Motor controller (SDK target) |

## ROS Topics

| Topic | Type | Source | Used by |
|-------|------|--------|---------|
| `/scan` | LaserScan | NX1 (LiDAR) | sensor_relay.py |
| `/scan_odom` | LaserScan | sensor_relay.py | subsumption_controller.py |
| `/ros2udp/odom` | Odometry | Pi (built-in) | sensor_relay.py |
