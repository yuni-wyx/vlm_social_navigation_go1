#!/usr/bin/env python3
"""
LLM Subsumption Controller for Unitree Go1
===========================================

A 3-tier Subsumption Architecture ROS node that bridges a high-level LLM
reasoning engine to low-level kinematic control on the Unitree Go1.

Architecture:
  Priority 1 (HIGHEST): Hard Safety Stop — Overrides everything if obstacle < 1.0m
  Priority 2:           Right-Biased Hallway Keeper — Wall-following correction
  Priority 3 (LOWEST):  LLM Intent Execution — EXPLORE_FORWARD, TAKE_NEXT_LEFT, etc.

Topics are adapted for the Go1's actual sensor stack:
  - /scan_odom (or /scan) for 2D LaserScan
  - /odom_fixed (or /odom) for odometry
  - /cmd_vel for motion commands

Author: Cairo (auto-generated)
ROS: Noetic (Python 3)
"""

import collections
import json
import math

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

# Optional: scipy for fast median filter, fall back to pure numpy
try:
    from scipy.ndimage import median_filter as scipy_median_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class LLMSubsumptionController:
    """
    Subsumption-based controller that fuses LiDAR safety, hallway-keeping,
    and LLM intent into a single /cmd_vel output at 30 Hz.
    """

    # ─── Physical Constants ─────────────────────────────────────────────
    SAFETY_DISTANCE = 1.5       # Hard stop if obstacle closer than this (meters)
    WALL_TARGET     = 0.6       # Ideal distance from the right wall (meters)
    WALL_DRIFT_MAX  = 0.8       # Start correcting toward wall if farther than this
    HALLWAY_CORR_VZ = 0.3       # Angular velocity for hallway corrections (rad/s)
    EXPLORE_VX      = 0.3       # Forward speed during EXPLORE_FORWARD (m/s)
    TURN_VZ         = 0.5       # Angular velocity during 90° turns (rad/s)
    OPENING_THRESH  = 2.5       # Distance threshold to detect a hallway opening (m)
    THREAT_FRAMES   = 3         # Consecutive frames required to confirm a threat
    SANITIZE_MAX    = 10.0      # Replacement value for inf/NaN readings

    # ─── Sector Angle Bounds (degrees → radians) ───────────────────────
    FRONT_MIN_DEG, FRONT_MAX_DEG = -30.0,  30.0
    LEFT_MIN_DEG,  LEFT_MAX_DEG  =  75.0, 105.0
    RIGHT_MIN_DEG, RIGHT_MAX_DEG = -105.0, -75.0

    def __init__(self):
        rospy.init_node("llm_subsumption_controller", anonymous=False)

        # ─── Parameters (configurable via rosparam) ─────────────────────
        self.scan_topic = rospy.get_param("~scan_topic", "/scan_odom")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom_fixed")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.control_rate = rospy.get_param("~control_rate", 30.0)
        self.dry_run = rospy.get_param("~dry_run", True)  # Safe default: don't move

        if self.dry_run:
            rospy.logwarn("[Subsumption] DRY RUN MODE — cmd_vel will be PRINTED, not published.")
            rospy.logwarn("[Subsumption] Set _dry_run:=false to enable actual robot motion.")

        rospy.loginfo(f"[Subsumption] scan={self.scan_topic}, odom={self.odom_topic}, "
                      f"cmd_vel={self.cmd_vel_topic}, rate={self.control_rate}Hz, "
                      f"dry_run={self.dry_run}")

        # Throttle dry-run prints to 2Hz (not 30Hz)
        self._dry_run_print_counter = 0
        self._dry_run_print_interval = int(self.control_rate / 2)  # Every 15 frames

        # ─── State Variables ────────────────────────────────────────────
        self.current_intent = "STOP_AND_SCAN"   # Safe default
        self.threat_counter = 0                  # Temporal debounce for P1

        # Filtered sector arrays (updated by scan callback)
        self.front_sector = np.array([self.SANITIZE_MAX])
        self.left_sector  = np.array([self.SANITIZE_MAX])
        self.right_sector = np.array([self.SANITIZE_MAX])
        self.scan_received = False

        # Turn state machine for TAKE_NEXT_LEFT / TAKE_NEXT_RIGHT
        self._turn_phase = None       # None | "seeking_opening" | "rotating"
        self._turn_direction = 0      # +1 for left, -1 for right
        self._turn_start_yaw = None
        self._current_yaw = 0.0

        # ─── Graph Memory Buffer (deque, max 50 nodes) ─────────────────
        self.local_graph = collections.deque(maxlen=50)
        self.distance_since_last_node = 0.0
        self._prev_odom_pos = None    # (x, y) tuple for delta computation

        # ─── Publishers ─────────────────────────────────────────────────
        self.cmd_pub = rospy.Publisher(
            self.cmd_vel_topic, Twist, queue_size=1
        )

        # ─── Subscribers ────────────────────────────────────────────────
        rospy.Subscriber(
            self.scan_topic, LaserScan, self._scan_cb, queue_size=1
        )
        rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_cb, queue_size=1
        )
        rospy.Subscriber(
            "/incoming_semantic_node", String, self._semantic_cb, queue_size=10
        )
        rospy.Subscriber(
            "/llm_intent", String, self._intent_cb, queue_size=1
        )

        # ─── Services ──────────────────────────────────────────────────
        rospy.Service(
            "/get_local_graph", Trigger, self._get_local_graph_srv
        )

        # ─── Control Timer (30 Hz) ─────────────────────────────────────
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate), self._control_loop
        )

        rospy.loginfo("[Subsumption] Controller initialized. Waiting for scan data...")

    # ═══════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════════

    def _scan_cb(self, msg: LaserScan):
        """
        LiDAR Filtering Pipeline:
          1. Sanitize inf/NaN
          2. 1D median filter (despeckle)
          3. Sector slicing (front, left, right)
        """
        # --- Step 1: Sanitize ---
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, self.SANITIZE_MAX)

        # --- Step 2: Spatial Filter (Despeckle) ---
        if HAS_SCIPY:
            ranges = scipy_median_filter(ranges, size=5).astype(np.float32)
        else:
            # Pure numpy fallback: pad + rolling median (window=5)
            padded = np.pad(ranges, 2, mode='edge')
            ranges = np.median(
                np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1
            ).astype(np.float32)

        # --- Step 3: Sector Slicing ---
        n = len(ranges)
        if n == 0:
            return

        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        # Precompute index boundaries for each sector
        def deg_to_idx(deg):
            rad = math.radians(deg)
            idx = int((rad - angle_min) / angle_inc)
            return max(0, min(n - 1, idx))

        fi0, fi1 = deg_to_idx(self.FRONT_MIN_DEG), deg_to_idx(self.FRONT_MAX_DEG)
        li0, li1 = deg_to_idx(self.LEFT_MIN_DEG),  deg_to_idx(self.LEFT_MAX_DEG)
        ri0, ri1 = deg_to_idx(self.RIGHT_MIN_DEG), deg_to_idx(self.RIGHT_MAX_DEG)

        self.front_sector = ranges[min(fi0, fi1):max(fi0, fi1) + 1]
        self.left_sector  = ranges[min(li0, li1):max(li0, li1) + 1]
        self.right_sector = ranges[min(ri0, ri1):max(ri0, ri1) + 1]

        # Extract yaw from the scan header timestamp for turn tracking
        self.scan_received = True

    def _odom_cb(self, msg: Odometry):
        """
        Integrate Euclidean distance traveled for geodesic distance tracking.
        Only uses relative deltas — no absolute positioning.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self._prev_odom_pos is not None:
            dx = x - self._prev_odom_pos[0]
            dy = y - self._prev_odom_pos[1]
            self.distance_since_last_node += math.sqrt(dx * dx + dy * dy)

        self._prev_odom_pos = (x, y)

        # Extract yaw from quaternion (used for 90° turn execution)
        q = msg.pose.pose.orientation
        # Yaw from quaternion: atan2(2(wz + xy), 1 - 2(y² + z²))
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._current_yaw = math.atan2(siny, cosy)

    def _semantic_cb(self, msg: String):
        """
        Accumulate a semantic node into the local graph buffer.
        Attaches the geodesic walking distance from the previous node.
        """
        try:
            node_data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            rospy.logwarn(f"[Subsumption] Bad JSON on /incoming_semantic_node: {e}")
            return

        entry = {
            "node_data": node_data,
            "geodesic_distance_from_prev": round(self.distance_since_last_node, 3),
            "timestamp": rospy.get_time(),
        }
        self.local_graph.append(entry)
        rospy.loginfo(
            f"[Graph] Node added (d={self.distance_since_last_node:.2f}m). "
            f"Buffer: {len(self.local_graph)}/50"
        )

        # Reset distance counter for next segment
        self.distance_since_last_node = 0.0

    def _intent_cb(self, msg: String):
        """
        Receive a state command from the LLM.
        Valid intents: EXPLORE_FORWARD, TAKE_NEXT_LEFT, TAKE_NEXT_RIGHT,
                       STOP_AND_SCAN
        """
        intent = msg.data.strip().upper()
        valid = {"EXPLORE_FORWARD", "TAKE_NEXT_LEFT", "TAKE_NEXT_RIGHT", "STOP_AND_SCAN"}
        if intent not in valid:
            rospy.logwarn(f"[Subsumption] Unknown intent '{msg.data}', ignoring.")
            return

        rospy.loginfo(f"[Subsumption] Intent: {self.current_intent} → {intent}")
        self.current_intent = intent

        # Initialize turn state machine if needed
        if intent in ("TAKE_NEXT_LEFT", "TAKE_NEXT_RIGHT"):
            self._turn_phase = "seeking_opening"
            self._turn_direction = 1 if intent == "TAKE_NEXT_LEFT" else -1
            self._turn_start_yaw = None

    # ═══════════════════════════════════════════════════════════════════
    # SERVICE
    # ═══════════════════════════════════════════════════════════════════

    def _get_local_graph_srv(self, req):
        """
        Returns the accumulated semantic trajectory as JSON,
        then clears the buffer for the next query cycle.
        """
        graph_list = list(self.local_graph)
        response = TriggerResponse()
        response.success = True
        response.message = json.dumps(graph_list)

        rospy.loginfo(
            f"[Graph] Served {len(graph_list)} nodes. Clearing buffer."
        )
        self.local_graph.clear()
        self.distance_since_last_node = 0.0

        return response

    # ═══════════════════════════════════════════════════════════════════
    # CONTROL LOOP (30 Hz Timer)
    # ═══════════════════════════════════════════════════════════════════

    def _control_loop(self, event):
        """
        3-tier Subsumption evaluator. Runs at 30 Hz.
        Publishes exactly one Twist message per invocation.
        """
        cmd = Twist()
        decision_reason = "WAITING_FOR_SCAN"

        # Don't command anything until we have real scan data
        if not self.scan_received:
            self._publish_or_print(cmd, decision_reason)
            return

        # ── Priority 1: Hard Safety Stop ────────────────────────────────
        front_min = float(np.min(self.front_sector))

        if front_min < self.SAFETY_DISTANCE:
            self.threat_counter += 1
            if self.threat_counter >= self.THREAT_FRAMES:
                # HARD STOP — absolute override, ignore LLM entirely
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                decision_reason = f"P1_SAFETY_STOP (obstacle={front_min:.2f}m)"
                self._publish_or_print(cmd, decision_reason)
                if self.threat_counter == self.THREAT_FRAMES:
                    rospy.logwarn(
                        f"[P1 SAFETY] Hard stop! Obstacle at {front_min:.2f}m "
                        f"(threshold={self.SAFETY_DISTANCE}m)"
                    )
                return
        else:
            self.threat_counter = 0

        # ── Priority 2: Right-Biased Hallway Keeper (with deadband) ─────
        dr = float(np.median(self.right_sector))  # Median right distance
        dl = float(np.median(self.left_sector))    # Median left distance

        hallway_override = False
        #  Deadband: [WALL_TARGET, WALL_DRIFT_MAX] = [0.6, 0.8] → no correction
        #  Below 0.6  → proportional veer LEFT  (too close to wall)
        #  Above 0.8  → proportional tuck RIGHT (drifting to center)
        if dr < self.WALL_TARGET:
            # Proportional: the closer, the stronger the correction
            error = self.WALL_TARGET - dr  # positive, 0→0.6 range
            vz = min(self.HALLWAY_CORR_VZ, 0.15 + error * 0.5)
            cmd.angular.z = vz
            cmd.linear.x = self.EXPLORE_VX * max(0.3, 1.0 - error * 2)
            decision_reason = f"P2_WALL_CLOSE (R={dr:.2f}m, vz=+{vz:.2f})"
            hallway_override = True
        elif dr > self.WALL_DRIFT_MAX and dl > 1.0:
            # Proportional: the farther, the stronger the tuck-back
            error = dr - self.WALL_DRIFT_MAX  # positive
            vz = min(self.HALLWAY_CORR_VZ, 0.1 + error * 0.3)
            cmd.angular.z = -vz
            cmd.linear.x = self.EXPLORE_VX * 0.8
            decision_reason = f"P2_WALL_DRIFT (R={dr:.2f}m, vz=-{vz:.2f})"
            hallway_override = True
        # else: dr in [0.6, 0.8] → DEADBAND, no P2 override

        if hallway_override:
            self._publish_or_print(cmd, decision_reason)
            return

        # ── Priority 3: Execute LLM Intent ──────────────────────────────
        intent = self.current_intent

        if intent == "STOP_AND_SCAN":
            decision_reason = "P3_STOP_AND_SCAN"
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        elif intent == "EXPLORE_FORWARD":
            decision_reason = "P3_EXPLORE_FORWARD"
            cmd.linear.x = self.EXPLORE_VX
            cmd.angular.z = 0.0

        elif intent in ("TAKE_NEXT_LEFT", "TAKE_NEXT_RIGHT"):
            decision_reason = f"P3_TURN_{self._turn_phase or 'init'}"
            cmd = self._execute_turn(cmd, dl if intent == "TAKE_NEXT_LEFT" else dr)

        self._publish_or_print(cmd, decision_reason)

    # ─── Publish or Print (Dry-Run Support) ───────────────────────────

    def _publish_or_print(self, cmd: Twist, reason: str):
        """
        In live mode: publish cmd to /cmd_vel.
        In dry-run mode: print a throttled summary of the decision to terminal.
        """
        if not self.dry_run:
            self.cmd_pub.publish(cmd)
            return

        # Throttle prints to ~2 Hz (every 15th frame at 30Hz)
        self._dry_run_print_counter += 1
        if self._dry_run_print_counter < self._dry_run_print_interval:
            return
        self._dry_run_print_counter = 0

        # Compute sector summaries
        f_min = float(np.min(self.front_sector))
        r_med = float(np.median(self.right_sector))
        l_med = float(np.median(self.left_sector))

        # ANSI colors for terminal readability
        RED = "\033[91m"
        YEL = "\033[93m"
        GRN = "\033[92m"
        CYN = "\033[96m"
        RST = "\033[0m"

        # Color the decision based on priority
        if "P1" in reason:
            color = RED
        elif "P2" in reason:
            color = YEL
        else:
            color = GRN

        print(
            f"{CYN}[DRY]{RST} "
            f"F={f_min:5.2f}m  R={r_med:5.2f}m  L={l_med:5.2f}m  │  "
            f"vx={cmd.linear.x:+.2f}  vz={cmd.angular.z:+.2f}  │  "
            f"{color}{reason}{RST}  │  "
            f"intent={self.current_intent}  dist={self.distance_since_last_node:.1f}m"
        )

    # ─── Turn State Machine ─────────────────────────────────────────────

    def _execute_turn(self, cmd: Twist, sector_dist: float) -> Twist:
        """
        Two-phase turn execution:
          Phase 1 ("seeking_opening"): Drive forward until the target side
                  shows an opening (distance > OPENING_THRESH).
          Phase 2 ("rotating"): Execute a 90° yaw rotation in place.
        """
        if self._turn_phase == "seeking_opening":
            if sector_dist > self.OPENING_THRESH:
                # Opening detected — begin rotation
                self._turn_phase = "rotating"
                self._turn_start_yaw = self._current_yaw
                rospy.loginfo(
                    f"[P3 Turn] Opening detected ({sector_dist:.1f}m). "
                    f"Starting 90° rotation."
                )
            else:
                # Keep driving forward, haven't reached the opening yet
                cmd.linear.x = self.EXPLORE_VX
                cmd.angular.z = 0.0
                return cmd

        if self._turn_phase == "rotating":
            if self._turn_start_yaw is None:
                self._turn_start_yaw = self._current_yaw

            # Calculate how far we've rotated
            delta_yaw = self._normalize_angle(
                self._current_yaw - self._turn_start_yaw
            )

            if abs(delta_yaw) < math.radians(85):
                # Still turning — rotate in place
                cmd.linear.x = 0.0
                cmd.angular.z = self.TURN_VZ * self._turn_direction
            else:
                # Turn complete — reset to EXPLORE_FORWARD
                rospy.loginfo("[P3 Turn] 90° turn complete. Resuming EXPLORE_FORWARD.")
                self._turn_phase = None
                self._turn_start_yaw = None
                self.current_intent = "EXPLORE_FORWARD"
                cmd.linear.x = self.EXPLORE_VX
                cmd.angular.z = 0.0

        return cmd

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # ═══════════════════════════════════════════════════════════════════
    # SHUTDOWN
    # ═══════════════════════════════════════════════════════════════════

    def shutdown(self):
        """Send a zero-velocity command on shutdown to stop the robot."""
        rospy.loginfo("[Subsumption] Shutting down — sending stop command.")
        self.cmd_pub.publish(Twist())


def main():
    controller = LLMSubsumptionController()
    rospy.on_shutdown(controller.shutdown)

    rospy.loginfo("[Subsumption] Running. Ctrl+C to stop.")
    rospy.spin()


if __name__ == "__main__":
    main()
