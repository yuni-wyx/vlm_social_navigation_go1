# LLM Subsumption Controller

3-tier Subsumption Architecture for LLM-guided navigation on the Unitree Go1.

## Architecture

```
Priority 1 (HIGHEST): Hard Safety Stop   — obstacle < 1.0m → full stop
Priority 2:           Hallway Keeper      — right-biased wall following
Priority 3 (LOWEST):  LLM Intent         — EXPLORE_FORWARD, TAKE_NEXT_LEFT, etc.
```

## Topics

| Direction | Topic | Type | Notes |
|-----------|-------|------|-------|
| Sub | `/scan_odom` | LaserScan | 2D LiDAR (configurable via `~scan_topic`) |
| Sub | `/odom_fixed` | Odometry | Relative distance only (configurable via `~odom_topic`) |
| Sub | `/incoming_semantic_node` | String | JSON objects from detection pipeline |
| Sub | `/llm_intent` | String | `EXPLORE_FORWARD`, `TAKE_NEXT_LEFT`, `TAKE_NEXT_RIGHT`, `STOP_AND_SCAN` |
| Pub | `/cmd_vel` | Twist | Kinematic output at 30 Hz |
| Srv | `/get_local_graph` | Trigger | Returns JSON of semantic trajectory, clears buffer |

## Usage

```bash
# With default topics (Go1 streaming)
rosrun llm_controller llm_subsumption_controller.py

# With custom topics
rosrun llm_controller llm_subsumption_controller.py \
    _scan_topic:=/scan _odom_topic:=/odom

# Send intent
rostopic pub /llm_intent std_msgs/String "data: 'EXPLORE_FORWARD'"

# Query local graph
rosservice call /get_local_graph
```

## Go1 Topic Mapping

The Go1 doesn't publish standard `/scan` or `/odom` — it uses:
- `/scan_odom` — 2D LaserScan from Velodyne (via `sensor_relay.py`)
- `/odom_fixed` — Odometry with fixed covariance (via `sensor_relay.py`)
- `/cmd_vel` — Standard Twist commands (published to Pi at 192.168.123.161)
