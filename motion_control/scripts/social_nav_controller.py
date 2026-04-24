#!/usr/bin/env python3
"""
Minimal social navigation controller for Unitree Go1.

This node reuses the existing Go1 control path:
    /scan_odom + /camera_face/left/image_raw -> laptop-side controller
    -> UDP :9900 -> sdk_udp_bridge.py on NX1 -> Unitree SDK -> Go1

Modes:
    baseline:
        Stop only when a frontal LiDAR obstacle is closer than
        baseline_stop_dist.

    human_aware:
        Detect people in the front camera. If a person is centered in front
        of the robot, use the larger social_stop_dist. Otherwise keep the
        baseline_stop_dist.

Notes:
    - This is intentionally a small MVP controller: forward or stop only.
    - It does not replace the existing subsumption controller.
    - Person detection uses OpenCV's built-in HOG pedestrian detector so the
      module stays self-contained and easy to swap later.
"""

import math
import json
import csv
import os
import base64
import threading
import signal
import socket
import struct
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import rospy
import requests
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from social_nav_policy import project_realtime_action

try:
    from cv_bridge import CvBridge, CvBridgeError
except ImportError:  # pragma: no cover - depends on ROS environment
    CvBridge = None
    CvBridgeError = Exception


class SocialNavController:
    def __init__(self):
        rospy.init_node("social_nav_controller", anonymous=False)

        self.mode = rospy.get_param("~mode", "baseline").strip().lower()
        if self.mode not in ("baseline", "human_aware", "social_fsm"):
            rospy.logwarn("Unknown mode '%s'; falling back to 'baseline'", self.mode)
            self.mode = "baseline"

        self.scan_topic = rospy.get_param("~scan_topic", "/scan_odom")
        self.image_topic = rospy.get_param("~image_topic", "/camera_face/left/image_raw")
        self.nx1_ip = rospy.get_param("~nx1_ip", "192.168.123.15")
        self.nx1_port = int(rospy.get_param("~nx1_port", 9900))
        self.forward_speed = float(rospy.get_param("~forward_speed", 0.2))
        self.baseline_stop_dist = float(rospy.get_param("~baseline_stop_dist", 0.6))
        self.social_stop_dist = float(rospy.get_param("~social_stop_dist", 1.2))
        self.front_sector_half_angle_deg = float(
            rospy.get_param("~front_sector_half_angle_deg", 10.0)
        )
        self.front_distance_percentile = float(
            rospy.get_param("~front_distance_percentile", 20.0)
        )
        self.person_center_fraction = float(
            rospy.get_param("~person_center_fraction", 0.4)
        )
        self.person_stride = int(rospy.get_param("~person_stride", 8))
        self.person_padding = int(rospy.get_param("~person_padding", 8))
        self.person_scale = float(rospy.get_param("~person_scale", 1.05))
        self.person_hit_threshold = float(
            rospy.get_param("~person_hit_threshold", 0.0)
        )
        self.max_range_fallback = float(rospy.get_param("~max_range_fallback", 10.0))
        self.scan_timeout = float(rospy.get_param("~scan_timeout", 1.0))
        self.image_timeout = float(rospy.get_param("~image_timeout", 2.0))
        legacy_vlm_api_url = rospy.get_param("~vlm_api_url", None)
        self.vlm_wrapper_url = rospy.get_param(
            "~vlm_wrapper_url",
            legacy_vlm_api_url if legacy_vlm_api_url is not None else "http://192.168.50.86:8000/analyze",
        )
        self.vlm_api_url = self.vlm_wrapper_url
        self.vlm_connect_timeout = float(rospy.get_param("~vlm_connect_timeout", 3.0))
        self.vlm_read_timeout = float(rospy.get_param("~vlm_read_timeout", 60.0))
        self.vision_query_interval = float(rospy.get_param("~vision_query_interval", 2.0))
        self.vision_stale_timeout = float(rospy.get_param("~vision_stale_timeout", 6.0))
        self.hold_until_first_vision = bool(rospy.get_param("~hold_until_first_vision", False))
        self.vlm_only_debug = bool(rospy.get_param("~vlm_only_debug", False))
        self.use_vlm_wrapper = bool(rospy.get_param("~use_vlm_wrapper", True))
        self.log_path = rospy.get_param("~log_path", "/tmp/social_nav.log")
        self.log_every_n = max(1, int(rospy.get_param("~log_every_n", 1)))
        self.debug_log_interval = float(rospy.get_param("~debug_log_interval", 1.0))
        self.debug_state_topic = rospy.get_param("~debug_state_topic", "/social_nav/debug_state")
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "/social_nav/debug_image")
        self.csv_log_interval = float(rospy.get_param("~csv_log_interval", 0.5))

        # ---------- Social-FSM parameters (used only when mode=social_fsm) ----------
        # These live alongside the existing params so baseline and human_aware modes
        # remain bit-for-bit unchanged.
        self.fsm_seq_prompt_name = rospy.get_param(
            "~fsm_seq_prompt_name", "sequence_image_navigation"
        )
        # Wrapper endpoint for the sequence-mode prompt. Must be the /analyze_navigation
        # endpoint of a wrapper whose social_nav_eval_prompts.py contains the
        # new-schema crossing-aware prompt.
        self.fsm_nav_wrapper_url = rospy.get_param(
            "~fsm_nav_wrapper_url", "http://192.168.50.86:8000/analyze_navigation"
        )
        # How many frames to stash in the ring buffer and how many to submit per decision.
        self.fsm_ring_size = int(rospy.get_param("~fsm_ring_size", 30))
        self.fsm_seq_length = int(rospy.get_param("~fsm_seq_length", 5))
        self.fsm_seq_span_sec = float(rospy.get_param("~fsm_seq_span_sec", 2.0))
        # When fsm_seq_length == 2 the controller sends a past+current pair
        # to the VLM. The past frame is taken from the ring buffer such that
        # its age lies in [fsm_seq_past_min_sec, fsm_seq_past_max_sec].
        self.fsm_seq_past_min_sec = float(
            rospy.get_param("~fsm_seq_past_min_sec", 1.0)
        )
        self.fsm_seq_past_max_sec = float(
            rospy.get_param("~fsm_seq_past_max_sec", 2.0)
        )
        # Triggers: a person appearing ahead or LiDAR coming closer than this
        # should force a transition from CRUISE -> SOCIAL_STOP.
        self.fsm_person_trigger_dist = float(
            rospy.get_param("~fsm_person_trigger_dist", 1.8)
        )
        self.fsm_hard_stop_dist = float(rospy.get_param("~fsm_hard_stop_dist", 0.6))
        # Dwell time after SOCIAL_STOP to let perception stabilize before DECIDE.
        self.fsm_social_stop_settle_sec = float(
            rospy.get_param("~fsm_social_stop_settle_sec", 0.8)
        )
        # Execute maneuvers: short, guarded bursts.
        # LEFT/RIGHT can be executed in one of three modes, chosen by
        # ``~fsm_lateral_mode``:
        #   - "turn_creep" (default): positive/negative yaw burst followed by
        #                             a short forward creep. Most conservative
        #                             because it never commands a lateral
        #                             velocity the platform has not validated.
        #   - "strafe":              pure lateral vy sidestep (wz=0, vx=0).
        #                             True "move parallel" avoidance. Requires
        #                             the platform to honour non-zero vy.
        #   - "blend":               small lateral vy + small forward vx for a
        #                             diagonal sidestep.
        self.fsm_lateral_mode = rospy.get_param(
            "~fsm_lateral_mode", "turn_creep"
        ).strip().lower()
        if self.fsm_lateral_mode not in ("turn_creep", "strafe", "blend", "turn_only"):
            rospy.logwarn(
                "[social_fsm] unknown fsm_lateral_mode=%r; falling back to turn_creep",
                self.fsm_lateral_mode,
            )
            self.fsm_lateral_mode = "turn_creep"
        self.fsm_forward_burst_sec = float(
            rospy.get_param("~fsm_forward_burst_sec", 0.6)
        )
        self.fsm_yaw_burst_sec = float(rospy.get_param("~fsm_yaw_burst_sec", 0.6))
        self.fsm_yaw_burst_speed = float(rospy.get_param("~fsm_yaw_burst_speed", 0.6))
        self.fsm_creep_after_turn_sec = float(
            rospy.get_param("~fsm_creep_after_turn_sec", 0.6)
        )
        self.fsm_creep_speed = float(rospy.get_param("~fsm_creep_speed", 0.15))
        # Strafe / blend parameters.
        self.fsm_strafe_sec = float(rospy.get_param("~fsm_strafe_sec", 0.8))
        self.fsm_strafe_speed = float(rospy.get_param("~fsm_strafe_speed", 0.15))
        self.fsm_blend_forward_speed = float(
            rospy.get_param("~fsm_blend_forward_speed", 0.1)
        )
        # Lateral chaining: if LiDAR still reports trigger-range obstacle
        # right after an EXECUTE LEFT/RIGHT primitive, re-issue the same
        # primitive up to ``fsm_lateral_chain_max`` additional times (same
        # direction) before going back to DECIDE. This keeps the robot
        # moving parallel instead of being pinned by LiDAR STOPs.
        self.fsm_lateral_chain_max = int(
            rospy.get_param("~fsm_lateral_chain_max", 3)
        )
        # Optional backward-recovery burst. When ``fsm_backup_on_hard_stop``
        # is true and the FSM would otherwise freeze at HARD_STOP, the
        # controller emits a short reverse burst (vx=-fsm_backup_speed for
        # fsm_backup_sec) to regain distance. Off by default: this is
        # open-loop backward motion with no rear sensing.
        self.fsm_backup_on_hard_stop = bool(
            rospy.get_param("~fsm_backup_on_hard_stop", False)
        )
        self.fsm_backup_speed = float(rospy.get_param("~fsm_backup_speed", 0.1))
        self.fsm_backup_sec = float(rospy.get_param("~fsm_backup_sec", 0.4))
        self.fsm_lateral_chain_count = 0
        self.fsm_last_lateral_action = None
        self.fsm_last_backup_time = 0.0
        # VLM-clearance latch: after the VLM commits a non-STOP action with
        # ``blocked=False``, suppress the LiDAR person-trigger in CRUISE for
        # this many seconds so the VLM's decision dominates over a
        # close-but-not-colliding LiDAR return. The hard-stop override
        # (front_dist < fsm_hard_stop_dist) is NEVER latched: it always
        # fires. Setting this to 0 disables the latch entirely.
        self.fsm_vlm_latch_sec = float(rospy.get_param("~fsm_vlm_latch_sec", 2.0))
        self._fsm_latch_until = 0.0
        self._fsm_latch_reason = None
        # "Turn off the LiDAR" controls. There are three levels, each a
        # separate knob so the operator can trade safety for autonomy
        # explicitly:
        #   - fsm_disable_lidar_trigger  : CRUISE no longer transitions to
        #     SOCIAL_STOP when front_dist < person_trigger. LiDAR no longer
        #     influences high-level decisions. The hard-stop safety override
        #     is still honoured unless also disabled.
        #   - fsm_disable_hard_stop      : hard-stop override is disabled
        #     too. VERY DANGEROUS: nothing will automatically halt the robot
        #     near a close obstacle. Use only with ESTOP reachable.
        #   - fsm_cruise_decide_interval_sec : when >0 AND lidar trigger is
        #     disabled, CRUISE forces a DECIDE every N seconds so the VLM
        #     still gets to influence motion even without a LiDAR trigger.
        self.fsm_disable_lidar_trigger = bool(
            rospy.get_param("~fsm_disable_lidar_trigger", False)
        )
        self.fsm_disable_hard_stop = bool(
            rospy.get_param("~fsm_disable_hard_stop", False)
        )
        self.fsm_cruise_decide_interval_sec = float(
            rospy.get_param("~fsm_cruise_decide_interval_sec", 0.0)
        )
        # Post-lateral DECIDE: after a LEFT/RIGHT EXECUTE primitive
        # completes, jump straight to DECIDE (bypassing the decide
        # cooldown) so the VLM is re-queried immediately. This makes the
        # robot keep strafing until the model says FORWARD.
        self.fsm_force_decide_after_lateral = bool(
            rospy.get_param("~fsm_force_decide_after_lateral", False)
        )
        # Safety gate: VLM LEFT/RIGHT outputs are advisory only by default.
        # Keep the suggested side in the logs, but do not execute a lateral
        # primitive unless an operator explicitly opts in.
        self.fsm_allow_vlm_lateral = bool(
            rospy.get_param("~fsm_allow_vlm_lateral", False)
        )
        # Safety cap: maximum number of consecutive lateral primitives
        # chosen by the VLM without an intervening FORWARD/STOP. Prevents
        # an infinite "keep strafing" loop when the scene is irresolvable.
        self.fsm_lateral_streak_max = int(
            rospy.get_param("~fsm_lateral_streak_max", 8)
        )
        self._fsm_lateral_streak = 0
        if self.fsm_disable_hard_stop:
            rospy.logwarn(
                "[social_fsm] HARD-STOP IS DISABLED (fsm_disable_hard_stop=true). "
                "The LiDAR emergency override will not halt the robot near obstacles. "
                "Keep the operator ESTOP reachable at all times."
            )
        if self.fsm_disable_lidar_trigger:
            rospy.logwarn(
                "[social_fsm] LiDAR person-trigger is disabled. CRUISE will not "
                "transition to SOCIAL_STOP from LiDAR alone. periodic_decide=%s",
                self.fsm_cruise_decide_interval_sec,
            )
        # HARD_STOP escape-strafe: when true, hitting the hard-stop threshold
        # triggers one short lateral primitive (direction = cached VLM
        # recommended_avoidance_side, else ``fsm_escape_default_side``).
        # After the primitive completes the FSM returns to CRUISE so the
        # robot can continue forward once the corridor opens.
        # Rising-edge triggered: only fires on the tick that newly enters the
        # hard-stop zone. If the escape does not clear the zone, subsequent
        # ticks freeze normally (no infinite escape loop).
        self.fsm_escape_on_hard_stop = bool(
            rospy.get_param("~fsm_escape_on_hard_stop", False)
        )
        self.fsm_escape_default_side = rospy.get_param(
            "~fsm_escape_default_side", "left"
        ).strip().lower()
        if self.fsm_escape_default_side not in ("left", "right"):
            self.fsm_escape_default_side = "left"
        self._fsm_prev_hard_stop_ok = None   # True if last tick had front >= hard_stop
        self._fsm_in_hard_stop_escape = False

        # Per-tick overhead controls. Each of these was running every 10 Hz
        # tick and could each add 1-30 ms. Turn them off if control latency
        # matters more than diagnostic bandwidth.
        self.fsm_disable_debug_image = bool(
            rospy.get_param("~fsm_disable_debug_image", False)
        )
        self.fsm_disable_per_tick_csv = bool(
            rospy.get_param("~fsm_disable_per_tick_csv", False)
        )
        self.fsm_disable_text_log = bool(
            rospy.get_param("~fsm_disable_text_log", False)
        )
        # Profiling: print max/avg/p95 loop time every ``fsm_profile_interval``
        # seconds when true. Useful for confirming which paths to disable.
        self.fsm_profile_loop = bool(rospy.get_param("~fsm_profile_loop", False))
        self.fsm_profile_interval = float(
            rospy.get_param("~fsm_profile_interval", 5.0)
        )
        self._fsm_loop_samples = []
        self._fsm_profile_last = time.time()
        # Cache the text-log handle so we do not open/close a file per tick.
        self._log_file_handle = None
        if not self.fsm_disable_text_log:
            try:
                self._log_file_handle = open(self.log_path, "a", buffering=1)
            except Exception as exc:
                rospy.logwarn("could not open text log: %r", exc)
                self._log_file_handle = None
        # Cooldowns / hysteresis so the robot does not thrash.
        self.fsm_decide_cooldown_sec = float(
            rospy.get_param("~fsm_decide_cooldown_sec", 1.5)
        )
        self.fsm_review_hold_sec = float(rospy.get_param("~fsm_review_hold_sec", 4.0))
        self.fsm_review_retries = int(rospy.get_param("~fsm_review_retries", 2))
        # Option-B: LiDAR-gated auto-recovery from HOLD_REVIEW.
        # After retries are exhausted the FSM normally waits for an operator,
        # but if LiDAR reports a clear corridor (front_dist >= clear_dist) for
        # ``clear_sec`` continuous seconds, reset and re-enter CRUISE. This
        # means "scene cleared" auto-unsticks without ever overriding the
        # VLM's decision to stop in a genuinely ambiguous scene.
        self.fsm_hold_review_clear_dist = float(
            rospy.get_param("~fsm_hold_review_clear_dist", self.fsm_person_trigger_dist)
        )
        self.fsm_hold_review_clear_sec = float(
            rospy.get_param("~fsm_hold_review_clear_sec", 3.0)
        )
        # Timestamp of the first consecutive tick where LiDAR was clear while
        # in HOLD_REVIEW. None when not currently clear.
        self._hold_review_clear_since = None
        # Safety overrides.
        self.fsm_request_timeout_sec = float(
            rospy.get_param("~fsm_request_timeout_sec", 8.0)
        )

        # FSM state (mutated only in the run loop thread).
        self.fsm_state = "CRUISE"
        self.fsm_state_entered = 0.0
        self.fsm_last_decision_time = 0.0
        self.fsm_last_action = "STOP"
        self.fsm_exec_until = 0.0
        self.fsm_exec_primitive = None
        self.fsm_review_count = 0
        # Ring buffer of (stamp_sec, BGR ndarray).
        self.fsm_frames = deque(maxlen=self.fsm_ring_size)
        self.fsm_frames_lock = threading.Lock()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.nx_addr = (self.nx1_ip, self.nx1_port)

        self.latest_scan = None
        self.latest_scan_stamp = None
        self.person_detected = False
        self.person_in_front = False
        self.last_detection_stamp = None
        self.last_valid_detection_stamp = None
        self.latest_image = None
        self.latest_image_stamp = None
        self.vision_source = "none"
        self.vision_latency = None
        self.last_vision_update_time = None
        self.vision_frequency = 0.0
        self.vision_valid = False
        self.vision_thread = None
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.last_stale_log_time = 0.0
        self.has_successful_vision_result = False
        self.last_wait_for_vision_log_time = 0.0
        self.loop_count = 0
        self.last_log_time = 0.0
        self.last_log_signature = None
        self.last_csv_time = 0.0
        self.last_csv_action = None

        self.logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.logs_dir, f"social_nav_{timestamp}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp",
            "mode",
            "vision_source",
            "vision_frequency",
            "vision_reference_time",
            "vision_age",
            "vision_stale",
            "front_dist",
            "person_detected",
            "person_in_front",
            "active_threshold",
            "threshold_source",
            "action",
        ])
        self.csv_file.flush()

        # Dedicated VLM-decision log. Captures every /analyze_navigation
        # response with full reasoning (motion, crossing_direction, avoidance
        # side, risk, uncertainty_reason, raw action) plus the FSM state that
        # acted on it and the LiDAR front-distance at the time.
        self.vlm_log_path = os.path.join(
            self.logs_dir, f"social_nav_vlm_{timestamp}.jsonl"
        )
        self.vlm_log_csv_path = os.path.join(
            self.logs_dir, f"social_nav_vlm_{timestamp}.csv"
        )
        self.vlm_log_file = open(self.vlm_log_path, "w")
        self.vlm_log_csv = open(self.vlm_log_csv_path, "w", newline="")
        self.vlm_csv_writer = csv.writer(self.vlm_log_csv)
        self.vlm_csv_writer.writerow([
            "timestamp",
            "fsm_state_entered_from",
            "front_dist",
            "num_images",
            "wrapper_url",
            "prompt_name",
            "action_normalized",
            "action_raw",
            "motion",
            "crossing_direction",
            "safer_lateral_side",
            "recommended_avoidance_side",
            "risk_level",
            "path_blocked_latest_frame",
            "uncertainty_reason",
            "wrapper_ok",
            "wrapper_error",
            "fsm_next_state",
            "robot_actuation",
            "latency_sec",
        ])
        self.vlm_log_csv.flush()
        self.fsm_last_front_dist = None
        # Edge-triggered UDP output state.
        self._last_udp_kind = None
        self._last_udp_vec = None
        self._last_udp_time = 0.0
        # Live-debug topic dedicated to the social FSM; emits one short JSON
        # message per 10 Hz tick tagged with who decided the action (LIDAR
        # vs VLM vs HARD_SAFETY) and the latest VLM reasoning that was used.
        self.fsm_debug_topic = rospy.get_param(
            "~fsm_debug_topic", "/social_nav/fsm_debug"
        )
        self.fsm_debug_pub = rospy.Publisher(
            self.fsm_debug_topic, String, queue_size=1
        )
        # Cache the most recent VLM decision so every tick's debug line can
        # show the reasoning currently driving motion, not only the tick
        # where the call happened.
        self.fsm_last_vlm = {
            "action": None,
            "motion": None,
            "crossing_direction": None,
            "safer_lateral_side": None,
            "recommended_avoidance_side": None,
            "risk_level": None,
            "path_blocked_latest_frame": None,
            "uncertainty_reason": None,
            "latency_sec": None,
            "timestamp": None,
        }

        self.debug_state_pub = rospy.Publisher(self.debug_state_topic, String, queue_size=1)
        self.debug_image_pub = None

        self.bridge = None
        self.hog = None
        # social_fsm wires the image stream + cv_bridge but does not start the
        # background HOG / legacy wrapper loop; decisions are on-demand from
        # the state machine.
        if self.mode == "social_fsm":
            if CvBridge is None:
                rospy.logerr(
                    "cv_bridge not available; social_fsm requires it. "
                    "Falling back to baseline."
                )
                self.mode = "baseline"
            else:
                self.bridge = CvBridge()
                rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        if self.mode == "human_aware":
            if CvBridge is None:
                rospy.logwarn("cv_bridge not available; human_aware mode will behave like baseline.")
            else:
                self.bridge = CvBridge()
                if not self.use_vlm_wrapper:
                    self.hog = cv2.HOGDescriptor()
                    self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
                rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
                if self.use_vlm_wrapper:
                    self.vision_thread = threading.Thread(
                        target=self.vision_update_loop,
                        name="social_nav_vision_worker",
                        daemon=True,
                    )
                    self.vision_thread.start()

        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_cb, queue_size=1)
        rospy.on_shutdown(self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        with open(self.log_path, "w") as log_file:
            log_file.write("social_nav_controller start\n")

        rospy.loginfo(
            "social_nav_controller: mode=%s scan=%s image=%s baseline=%.2f social=%.2f vlm_api=%s use_vlm_wrapper=%s vision_query_interval=%.2f vision_stale_timeout=%.2f hold_until_first_vision=%s vlm_only_debug=%s",
            self.mode,
            self.scan_topic,
            self.image_topic,
            self.baseline_stop_dist,
            self.social_stop_dist,
            self.vlm_api_url,
            self.use_vlm_wrapper,
            self.vision_query_interval,
            self.vision_stale_timeout,
            self.hold_until_first_vision,
            self.vlm_only_debug,
        )
        print(f"Logging to CSV: {self.csv_path}")

    def send_velocity(self, vx, vy, wz):
        self.sock.sendto(struct.pack("fff", vx, vy, wz), self.nx_addr)

    def send_stop(self):
        for _ in range(5):
            self.sock.sendto(b"STOP", self.nx_addr)

    def shutdown(self, *_args):
        self.stop_event.set()
        try:
            self.send_stop()
        finally:
            if self.vision_thread is not None and self.vision_thread.is_alive():
                self.vision_thread.join(timeout=1.0)
            for attr in ("csv_file", "vlm_log_file", "vlm_log_csv", "_log_file_handle"):
                try:
                    h = getattr(self, attr)
                    if h is not None:
                        h.close()
                except Exception:
                    pass
            try:
                self.sock.close()
            except Exception:
                pass

    def scan_cb(self, msg):
        self.latest_scan = msg
        self.latest_scan_stamp = rospy.Time.now()

    def image_cb(self, msg):
        if self.bridge is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, "cv_bridge conversion failed: %s", exc)
            return

        with self.state_lock:
            self.latest_image = frame
            self.latest_image_stamp = rospy.Time.now()
        if self.mode == "social_fsm":
            # Push a lightweight copy into the ring buffer (bounded by fsm_ring_size).
            with self.fsm_frames_lock:
                self.fsm_frames.append((time.time(), frame.copy()))
            return
        if self.use_vlm_wrapper:
            return
        else:
            if self.hog is None:
                return
            detected, in_front = self.detect_person(frame)
            with self.state_lock:
                self.vision_source = "hog_local"
                self.vision_latency = 0.0
                self.vision_frequency = 0.0
                self.vision_valid = True
                self.last_vision_update_time = rospy.Time.now()
                self.person_detected = detected
                self.person_in_front = in_front
                self.last_detection_stamp = rospy.Time.now()
                self.last_valid_detection_stamp = self.last_detection_stamp

    def vision_update_loop(self):
        while not rospy.is_shutdown() and not self.stop_event.is_set():
            self.update_cached_vision()
            self.stop_event.wait(self.vision_query_interval)

    def update_cached_vision(self):
        with self.state_lock:
            frame = None if self.latest_image is None else self.latest_image.copy()
            image_stamp = self.latest_image_stamp

        if frame is None or image_stamp is None:
            return

        start = time.time()
        detected, in_front, source, is_valid = self.detect_person_vlm(frame)
        latency = time.time() - start

        with self.state_lock:
            previous_update_time = self.last_vision_update_time
            self.vision_latency = latency
            self.last_vision_update_time = rospy.Time.now()
            if previous_update_time is not None:
                dt = (self.last_vision_update_time - previous_update_time).to_sec()
                self.vision_frequency = 1.0 / dt if dt > 1e-6 else 0.0
            if is_valid:
                self.person_detected = detected
                self.person_in_front = in_front
                self.vision_source = source
                self.vision_valid = True
                self.last_detection_stamp = rospy.Time.now()
                self.last_valid_detection_stamp = self.last_detection_stamp
                self.has_successful_vision_result = True

        rospy.loginfo(
            "[social_nav] vision_update source=%s valid=%s person_detected=%s person_in_front=%s latency=%.2f frequency=%.3f reference_time=%.3f",
            source,
            is_valid,
            detected,
            in_front,
            latency,
            self.vision_frequency,
            self.last_vision_update_time.to_sec() if self.last_vision_update_time is not None else 0.0,
        )

    def detect_person_vlm(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            rospy.logwarn_throttle(5.0, "JPEG encode failed for VLM wrapper request")
            return False, False, "vlm_wrapper_encode_error", False

        payload = {
            "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii")
        }

        try:
            response = requests.post(
                self.vlm_api_url,
                json=payload,
                timeout=(self.vlm_connect_timeout, self.vlm_read_timeout),
            )
            response.raise_for_status()
            data = response.json()
        except requests.Timeout as exc:
            rospy.logwarn(
                "VLM wrapper timeout url=%s error=%r",
                self.vlm_api_url,
                exc,
            )
            return False, False, "vlm_wrapper_timeout", False
        except requests.ConnectionError as exc:
            rospy.logwarn(
                "VLM wrapper connection error url=%s error=%r",
                self.vlm_api_url,
                exc,
            )
            return False, False, "vlm_wrapper_connection_error", False
        except requests.HTTPError as exc:
            body = ""
            if exc.response is not None:
                body = exc.response.text[:300]
            rospy.logwarn(
                "VLM wrapper HTTP error url=%s error=%r status=%s body=%s",
                self.vlm_api_url,
                exc,
                getattr(exc.response, "status_code", "unknown"),
                body,
            )
            return False, False, "vlm_wrapper_http_error", False
        except requests.RequestException as exc:
            body = getattr(getattr(exc, "response", None), "text", "")[:300]
            rospy.logwarn(
                "VLM wrapper request failed url=%s error=%r status=%s body=%s",
                self.vlm_api_url,
                exc,
                getattr(getattr(exc, "response", None), "status_code", "unknown"),
                body,
            )
            return False, False, "vlm_wrapper_request_error", False
        except ValueError as exc:
            rospy.logwarn("VLM wrapper JSON decode failed url=%s error=%r", self.vlm_api_url, exc)
            return False, False, "vlm_wrapper_json_error", False

        if not data.get("ok", False):
            rospy.logwarn(
                "VLM wrapper returned ok=false url=%s error=%s raw_text=%s",
                self.vlm_api_url,
                data.get("error", "unknown"),
                str(data.get("raw_text", ""))[:300],
            )
            return False, False, "vlm_wrapper_ok_false", False

        return bool(data.get("person_detected", False)), bool(data.get("person_in_front", False)), "vlm_cache", True

    def detect_person(self, frame):
        height, width = frame.shape[:2]
        work = frame

        if width > 640:
            scale = 640.0 / float(width)
            work = cv2.resize(frame, (640, int(height * scale)))
        else:
            scale = 1.0

        boxes, _weights = self.hog.detectMultiScale(
            work,
            winStride=(self.person_stride, self.person_stride),
            padding=(self.person_padding, self.person_padding),
            scale=self.person_scale,
            hitThreshold=self.person_hit_threshold,
        )

        if len(boxes) == 0:
            return False, False

        center_left = width * (0.5 - self.person_center_fraction / 2.0)
        center_right = width * (0.5 + self.person_center_fraction / 2.0)

        for (x, y, w, h) in boxes:
            center_x = (x + 0.5 * w) / scale
            if center_left <= center_x <= center_right:
                return True, True

        return True, False

    @staticmethod
    def get_sector(ranges, angle_min, angle_inc, n, deg_start, deg_end):
        def deg_to_idx(deg):
            angle = math.radians(deg)
            raw = int((angle - angle_min) / angle_inc)
            return max(0, min(n - 1, raw))

        i0 = deg_to_idx(deg_start)
        i1 = deg_to_idx(deg_end)
        lo, hi = min(i0, i1), max(i0, i1)
        return ranges[lo:hi + 1]

    def compute_front_distance(self, scan_msg):
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        valid = np.isfinite(ranges) & (ranges > 0.05)
        ranges = ranges[valid]

        if ranges.size == 0:
            return self.max_range_fallback

        sanitized = np.array(scan_msg.ranges, dtype=np.float32)
        sanitized = np.where(
            np.isfinite(sanitized) & (sanitized > 0.05),
            sanitized,
            self.max_range_fallback,
        )

        front = self.get_sector(
            sanitized,
            scan_msg.angle_min,
            scan_msg.angle_increment,
            len(sanitized),
            -self.front_sector_half_angle_deg,
            self.front_sector_half_angle_deg,
        )

        if front.size == 0:
            return self.max_range_fallback

        front_valid = front[np.isfinite(front) & (front > 0.05)]
        if front_valid.size == 0:
            return self.max_range_fallback

        percentile = np.clip(self.front_distance_percentile, 0.0, 100.0)
        return float(np.percentile(front_valid, percentile))

    def publish_debug_image(self, state):
        if self.bridge is None or self.debug_image_pub is None or self.latest_image is None:
            return

        overlay = self.latest_image.copy()
        color = (0, 255, 0) if state["action"] == "FORWARD" else (0, 0, 255)
        lines = [
            f"mode: {state['mode']}",
            f"vision_source: {state['vision_source']}",
            f"vision_frequency: {state['vision_frequency']:.3f}",
            f"vision_ref_time: {state['vision_reference_time']:.2f}",
            f"vision_age: {state['vision_age']:.2f}",
            f"vision_stale: {state['vision_stale']}",
            f"front_dist: {state['front_dist']:.2f}",
            f"person_detected: {state['person_detected']}",
            f"person_in_front: {state['person_in_front']}",
            f"active_threshold: {state['active_threshold']:.2f}",
            f"threshold_source: {state['threshold_source']}",
            f"action: {state['action']}",
        ]

        y = 30
        for line in lines:
            cv2.putText(
                overlay,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            y += 28

        try:
            msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            msg.header.stamp = rospy.Time.now()
            self.debug_image_pub.publish(msg)
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, "debug image publish failed: %s", exc)

    def build_state_line(self, state):
        return (
            f"[social_nav] mode={state['mode']} "
            f"vision_source={state['vision_source']} "
            f"vision_frequency={state['vision_frequency']:.3f} "
            f"vision_reference_time={state['vision_reference_time']:.3f} "
            f"vision_age={state['vision_age']:.1f} "
            f"vision_stale={state['vision_stale']} "
            f"front_dist={state['front_dist']:.2f} "
            f"person_detected={state['person_detected']} "
            f"person_in_front={state['person_in_front']} "
            f"active_threshold={state['active_threshold']:.2f} "
            f"threshold_source={state['threshold_source']} "
            f"action={state['action']}"
        )

    def maybe_log_state(self, state):
        now_sec = rospy.Time.now().to_sec()
        signature = (
            state["mode"],
            state["vision_source"],
            round(state["vision_frequency"], 3),
            round(state["vision_reference_time"], 3),
            round(state["vision_age"], 1),
            state["vision_stale"],
            state["person_detected"],
            state["person_in_front"],
            round(state["front_dist"], 2),
            round(state["active_threshold"], 2),
            state["threshold_source"],
            state["action"],
        )

        should_log = (
            self.last_log_signature != signature
            or (now_sec - self.last_log_time) >= self.debug_log_interval
        )
        if not should_log:
            return

        line = self.build_state_line(state)
        rospy.loginfo(line)
        if not self.fsm_disable_text_log and self._log_file_handle is not None:
            try:
                self._log_file_handle.write(
                    f"{datetime.now().isoformat(timespec='seconds')} {line}\n"
                )
            except Exception:
                pass
        self.last_log_signature = signature
        self.last_log_time = now_sec

    def maybe_log_csv(self, state):
        now_sec = time.time()
        if (
            self.last_csv_action == state["action"]
            and (now_sec - self.last_csv_time) < self.csv_log_interval
        ):
            return

        self.csv_writer.writerow([
            now_sec,
            state["mode"],
            state["vision_source"],
            f"{state['vision_frequency']:.3f}",
            f"{state['vision_reference_time']:.3f}",
            f"{state['vision_age']:.3f}",
            state["vision_stale"],
            f"{state['front_dist']:.3f}",
            state["person_detected"],
            state["person_in_front"],
            f"{state['active_threshold']:.3f}",
            state["threshold_source"],
            state["action"],
        ])
        self.csv_file.flush()
        self.last_csv_time = now_sec
        self.last_csv_action = state["action"]

    # ------------------------------------------------------------------ FSM helpers

    def fsm_pick_sequence(self):
        """Pick the frames to send to the VLM.

        Special-case: when ``fsm_seq_length == 2``, return a "past + current"
        pair. The newest frame in the ring buffer is the current frame; the
        paired past frame is the first frame in the buffer whose age falls in
        [fsm_seq_past_min_sec, fsm_seq_past_max_sec] seconds. This gives the
        VLM a meaningful temporal baseline with only two images (so each
        sequence call pays the cost of a 2-image request instead of N).

        General case: keep the original policy of ``fsm_seq_length`` frames
        evenly spaced across the last ``fsm_seq_span_sec`` seconds.

        Returns a list of BGR ndarrays, newest last. Returns None if the
        buffer does not yet hold suitable frames.
        """
        with self.fsm_frames_lock:
            snapshot = list(self.fsm_frames)
        if self.fsm_seq_length <= 0 or not snapshot:
            return None

        if self.fsm_seq_length == 2:
            now = time.time()
            newest_t, newest_f = snapshot[-1]
            past_min = self.fsm_seq_past_min_sec
            past_max = self.fsm_seq_past_max_sec
            # Walk from oldest to newest looking for a frame whose age (from
            # the current wall clock) falls in the target window.
            past_f = None
            for (t, f) in snapshot:
                age = now - t
                if past_min <= age <= past_max:
                    past_f = f
                    break
            if past_f is None:
                # Not enough history yet: fall back to the oldest available
                # if its age >= past_min, else use whatever is oldest.
                oldest_t, oldest_f = snapshot[0]
                if (now - oldest_t) >= past_min:
                    past_f = oldest_f
                else:
                    # Buffer is too fresh to provide a meaningful past frame.
                    # Caller should wait one more tick.
                    return None
            return [past_f, newest_f]

        # General N-frame path (unchanged from prior behaviour).
        if len(snapshot) < self.fsm_seq_length:
            return None
        now = time.time()
        window = [(t, f) for (t, f) in snapshot if now - t <= self.fsm_seq_span_sec]
        if len(window) < self.fsm_seq_length:
            window = snapshot[-self.fsm_seq_length :]
        n = len(window)
        if n == self.fsm_seq_length:
            picks = window
        else:
            last = n - 1
            idx = [round(i * last / (self.fsm_seq_length - 1)) for i in range(self.fsm_seq_length)]
            seen = set(); out = []
            for k in idx:
                if k not in seen:
                    seen.add(k); out.append(window[k])
            while len(out) < self.fsm_seq_length and out:
                out.append(window[-1])
            picks = out[: self.fsm_seq_length]
        return [f for (_t, f) in picks]

    @staticmethod
    def _jpeg_b64(frame):
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            return None
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def fsm_call_sequence_wrapper(self, frames):
        """POST a short image sequence to /analyze_navigation with the
        sequence prompt. Returns ``(action, response_json, envelope, latency)``
        where ``action`` is normalized to STOP/FORWARD/LEFT/RIGHT/REVIEW or
        None on failure. ``envelope`` is the full wrapper response (including
        `ok` / `error`) so callers can log it verbatim."""
        images_b64 = []
        for f in frames:
            b = self._jpeg_b64(f)
            if b is None:
                return None, {"error": "jpeg_encode_failed"}, {"error": "jpeg_encode_failed"}, 0.0
            images_b64.append(b)
        payload = {
            "prompt_name": self.fsm_seq_prompt_name,
            "images_base64": images_b64,
        }
        t_start = time.time()
        try:
            resp = requests.post(
                self.fsm_nav_wrapper_url,
                json=payload,
                timeout=(self.vlm_connect_timeout, self.fsm_request_timeout_sec),
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            latency = time.time() - t_start
            rospy.logwarn("[social_fsm] wrapper request failed: %r", exc)
            return None, {"error": f"request_failed: {exc}"}, {"ok": False, "error": f"request_failed: {exc}"}, latency
        except ValueError as exc:
            latency = time.time() - t_start
            rospy.logwarn("[social_fsm] wrapper JSON decode failed: %r", exc)
            return None, {"error": f"json_decode_failed: {exc}"}, {"ok": False, "error": f"json_decode_failed: {exc}"}, latency
        latency = time.time() - t_start
        if not data.get("ok"):
            rospy.logwarn("[social_fsm] wrapper ok=false err=%s", data.get("error"))
            return None, data.get("response_json") or {}, data, latency
        rj = data.get("response_json") or {}
        raw_action = str(rj.get("recommended_action", "")).strip().upper()
        action = raw_action
        # Normalize synonyms conservatively — do NOT allow unknown labels to
        # unlock motion.
        if action in ("GO_LEFT", "STEP_LEFT", "SIDESTEP_LEFT", "TURN_LEFT"):
            action = "LEFT"
        elif action in ("GO_RIGHT", "STEP_RIGHT", "SIDESTEP_RIGHT", "TURN_RIGHT"):
            action = "RIGHT"
        elif action in ("HALT", "YIELD"):
            action = "STOP"
        elif action in ("CONTINUE", "GO", "PROCEED"):
            action = "FORWARD"
        elif action in ("WAIT", "HOLD", "UNKNOWN", "UNSURE", "UNCERTAIN", "DEFER"):
            action = "REVIEW"
        if action not in ("STOP", "FORWARD", "LEFT", "RIGHT", "REVIEW"):
            return None, rj, data, latency
        return action, rj, data, latency

    def log_vlm_decision(self, *, state_entered_from, front_dist, num_images,
                         action_normalized, action_raw, response_json, envelope,
                         latency_sec, fsm_next_state, robot_actuation):
        """Persist a single VLM decision to both JSONL (full detail) and
        CSV (tabular summary). Every DECIDE cycle emits one row, including
        wrapper failures."""
        now = time.time()
        rj = response_json or {}
        record = {
            "timestamp": now,
            "datetime": datetime.utcfromtimestamp(now).isoformat() + "Z",
            "fsm_state_entered_from": state_entered_from,
            "front_dist": front_dist,
            "num_images": num_images,
            "wrapper_url": self.fsm_nav_wrapper_url,
            "prompt_name": self.fsm_seq_prompt_name,
            "latency_sec": latency_sec,
            "wrapper_ok": bool((envelope or {}).get("ok", False)),
            "wrapper_error": (envelope or {}).get("error", ""),
            "action_normalized": action_normalized,
            "action_raw": action_raw,
            "response_json": rj,
            "raw_text_snippet": str((envelope or {}).get("raw_text", ""))[:400],
            "fsm_next_state": fsm_next_state,
            "robot_actuation": robot_actuation,
        }
        try:
            self.vlm_log_file.write(json.dumps(record, sort_keys=True) + "\n")
            self.vlm_log_file.flush()
        except Exception:
            pass
        try:
            self.vlm_csv_writer.writerow([
                f"{now:.3f}",
                state_entered_from,
                f"{front_dist:.3f}" if front_dist is not None else "",
                num_images,
                self.fsm_nav_wrapper_url,
                self.fsm_seq_prompt_name,
                action_normalized or "",
                action_raw or "",
                rj.get("motion", ""),
                rj.get("crossing_direction", ""),
                rj.get("safer_lateral_side", ""),
                rj.get("recommended_avoidance_side", ""),
                rj.get("risk_level", ""),
                rj.get("path_blocked_latest_frame", ""),
                (rj.get("uncertainty_reason", "") or "").replace("\n", " ")[:200],
                bool((envelope or {}).get("ok", False)),
                str((envelope or {}).get("error", ""))[:200],
                fsm_next_state,
                robot_actuation,
                f"{latency_sec:.3f}",
            ])
            self.vlm_log_csv.flush()
        except Exception:
            pass

    def fsm_enter(self, new_state):
        if self.fsm_state != new_state:
            rospy.loginfo(
                "[social_fsm] %s -> %s (t=%.2f)",
                self.fsm_state,
                new_state,
                time.time() - self.fsm_state_entered,
            )
        self.fsm_state = new_state
        self.fsm_state_entered = time.time()

    def fsm_lateral_primitive_duration(self):
        """Total time the LEFT/RIGHT primitive runs for under the current mode."""
        if self.fsm_lateral_mode == "strafe":
            return self.fsm_strafe_sec
        if self.fsm_lateral_mode == "blend":
            return self.fsm_strafe_sec
        if self.fsm_lateral_mode == "turn_only":
            return self.fsm_yaw_burst_sec
        # turn_creep
        return self.fsm_yaw_burst_sec + self.fsm_creep_after_turn_sec

    def fsm_begin_primitive(self, action):
        """Kick off the short guarded maneuver for the chosen action."""
        now = time.time()
        self.fsm_exec_primitive = action
        self.fsm_exec_started = now
        if action == "FORWARD":
            self.fsm_exec_until = now + self.fsm_forward_burst_sec
        elif action in ("LEFT", "RIGHT"):
            self.fsm_exec_until = now + self.fsm_lateral_primitive_duration()
        else:
            # STOP / REVIEW: execute primitive is no-op; the driver sends STOP.
            self.fsm_exec_until = now

    def fsm_tick_primitive(self):
        """Returns (vx, vy, wz) for the currently-active primitive, or None
        when the primitive has ended."""
        now = time.time()
        if now >= self.fsm_exec_until or self.fsm_exec_primitive is None:
            return None
        action = self.fsm_exec_primitive
        if action == "FORWARD":
            return (self.forward_speed, 0.0, 0.0)
        if action in ("LEFT", "RIGHT"):
            sign = 1.0 if action == "LEFT" else -1.0
            if self.fsm_lateral_mode == "strafe":
                # Pure lateral vy sidestep. Positive vy is the robot's left
                # on the Unitree high-level controller.
                return (0.0, sign * self.fsm_strafe_speed, 0.0)
            if self.fsm_lateral_mode == "blend":
                # Diagonal: small forward + lateral simultaneously.
                return (self.fsm_blend_forward_speed,
                        sign * self.fsm_strafe_speed,
                        0.0)
            if self.fsm_lateral_mode == "turn_only":
                # Pure yaw in place: rotates the robot's front camera to
                # look in a new direction without translating. Paired with
                # fsm_force_decide_after_lateral so the VLM re-examines the
                # new view and commits FORWARD when it sees a clear path.
                return (0.0, 0.0, sign * self.fsm_yaw_burst_speed)
            # turn_creep (default)
            elapsed = now - self.fsm_exec_started
            if elapsed < self.fsm_yaw_burst_sec:
                return (0.0, 0.0, sign * self.fsm_yaw_burst_speed)
            return (self.fsm_creep_speed, 0.0, 0.0)
        return None

    def fsm_describe_actuation(self, action):
        """Human-readable description of the primitive that will be executed
        for ``action`` under the current lateral mode. Used for the VLM log."""
        if action == "FORWARD":
            return f"vx={self.forward_speed:.2f} for {self.fsm_forward_burst_sec:.2f}s"
        if action in ("LEFT", "RIGHT"):
            sign_str = "+" if action == "LEFT" else "-"
            if self.fsm_lateral_mode == "strafe":
                return f"vy={sign_str}{self.fsm_strafe_speed:.2f} for {self.fsm_strafe_sec:.2f}s (pure strafe)"
            if self.fsm_lateral_mode == "blend":
                return (f"vx={self.fsm_blend_forward_speed:.2f} vy={sign_str}{self.fsm_strafe_speed:.2f} "
                        f"for {self.fsm_strafe_sec:.2f}s (blend diagonal)")
            if self.fsm_lateral_mode == "turn_only":
                return f"wz={sign_str}{self.fsm_yaw_burst_speed:.2f} for {self.fsm_yaw_burst_sec:.2f}s (turn in place)"
            return (f"wz={sign_str}{self.fsm_yaw_burst_speed:.2f} for {self.fsm_yaw_burst_sec:.2f}s, "
                    f"then creep vx={self.fsm_creep_speed:.2f} for {self.fsm_creep_after_turn_sec:.2f}s (turn_creep)")
        return "STOP"

    def fsm_should_stop_for_person(self, front_distance):
        """Cheap on-every-tick trigger to leave CRUISE. Uses the LiDAR front
        distance as a proxy: if something comes closer than the trigger
        distance, yield to SOCIAL_STOP and let the VLM decide."""
        return front_distance < self.fsm_person_trigger_dist

    def fsm_step(self, front_distance):
        """One tick of the five-state social FSM.
        Returns (vx, vy, wz), action_label, decision_source, reason
        where ``decision_source`` is one of:
          - ``LIDAR``       : the current action was chosen by the LiDAR-only
                              layer (CRUISE forward / SOCIAL_STOP reflex stop).
          - ``HARD_SAFETY`` : LiDAR hard-stop override (front_dist < hard_stop_dist).
          - ``VLM``         : the current action is the VLM's committed decision
                              being applied as an EXECUTE primitive.
          - ``VLM_WAIT``    : FSM is in DECIDE waiting for VLM response, or in
                              HOLD_REVIEW after a REVIEW decision.
          - ``FSM_SETTLE``  : FSM is in SOCIAL_STOP dwelling before DECIDE.
        """
        # Authoritative hard-safety override (LiDAR).
        # Completely skippable via fsm_disable_hard_stop (DANGEROUS).
        if not self.fsm_disable_hard_stop and front_distance < self.fsm_hard_stop_dist:
            now = time.time()
            rising_edge = (
                self._fsm_prev_hard_stop_ok is True
                or self._fsm_prev_hard_stop_ok is None
            )
            self._fsm_prev_hard_stop_ok = False
            # If an escape primitive is already running, let the EXECUTE
            # branch tick it through. Don't re-trigger.
            if self.fsm_state == "EXECUTE" and self._fsm_in_hard_stop_escape:
                vel = self.fsm_tick_primitive()
                if vel is not None:
                    return vel, "ESCAPE", "HARD_SAFETY", \
                        f"escape primitive running front={front_distance:.2f}m"
                # Primitive ended but we're still inside hard-stop zone:
                # clear flag and fall through to freeze.
                self._fsm_in_hard_stop_escape = False
                self.fsm_exec_primitive = None
            # Rising-edge escape: only kick off a new primitive the moment we
            # transitioned into the hard-stop zone.
            if (rising_edge and self.fsm_escape_on_hard_stop
                    and not self._fsm_in_hard_stop_escape):
                side = (self.fsm_last_vlm.get("recommended_avoidance_side") or "").lower()
                if side == "left":
                    action = "LEFT"
                elif side == "right":
                    action = "RIGHT"
                else:
                    action = "LEFT" if self.fsm_escape_default_side == "left" else "RIGHT"
                self._fsm_in_hard_stop_escape = True
                self.fsm_begin_primitive(action)
                self.fsm_enter("EXECUTE")
                vel = self.fsm_tick_primitive() or (0.0, 0.0, 0.0)
                return vel, f"ESCAPE_{action}", "HARD_SAFETY", \
                    f"hard-stop escape {action} (VLM avoid={side!r}) front={front_distance:.2f}m"
            # Optional backward-recovery burst (unchanged behaviour).
            if (self.fsm_backup_on_hard_stop
                and (now - self.fsm_last_backup_time) > (2.0 * self.fsm_backup_sec)):
                self.fsm_last_backup_time = now
                return (-self.fsm_backup_speed, 0.0, 0.0), "BACKUP", "HARD_SAFETY", \
                    f"backup vx=-{self.fsm_backup_speed:.2f} (front={front_distance:.2f}m < hard_stop)"
            # Default: freeze.
            if self.fsm_state != "SOCIAL_STOP":
                self.fsm_enter("SOCIAL_STOP")
            return (0.0, 0.0, 0.0), "HARD_STOP", "HARD_SAFETY", \
                f"front_dist={front_distance:.2f}m < hard_stop={self.fsm_hard_stop_dist:.2f}m"
        # Above hard_stop: clear rising-edge state for next time.
        self._fsm_prev_hard_stop_ok = True
        # If an escape primitive was running and we're now clear, let EXECUTE
        # tick it to completion then fall to CRUISE.
        if self._fsm_in_hard_stop_escape and self.fsm_state == "EXECUTE":
            vel = self.fsm_tick_primitive()
            if vel is not None:
                return vel, "ESCAPE", "VLM", \
                    f"escape primitive continuing front={front_distance:.2f}m cleared hard_stop"
            # Primitive done and cleared; reset and go CRUISE.
            self._fsm_in_hard_stop_escape = False
            self.fsm_exec_primitive = None
            self.fsm_enter("CRUISE")
            return (0.0, 0.0, 0.0), "STOP", "VLM", "escape primitive finished; resuming CRUISE"

        state = self.fsm_state

        if state == "CRUISE":
            now = time.time()
            latched = now < self._fsm_latch_until
            # LiDAR person-trigger path (skippable by fsm_disable_lidar_trigger).
            if (not self.fsm_disable_lidar_trigger
                    and self.fsm_should_stop_for_person(front_distance)
                    and not latched):
                self.fsm_enter("SOCIAL_STOP")
                return (0.0, 0.0, 0.0), "STOP", "LIDAR", \
                    f"front_dist={front_distance:.2f}m < trigger={self.fsm_person_trigger_dist:.2f}m"
            # Periodic DECIDE when LiDAR trigger is disabled and an interval
            # is set: let the VLM get a say every N seconds even without a
            # LiDAR trigger.
            if (self.fsm_disable_lidar_trigger
                    and self.fsm_cruise_decide_interval_sec > 0.0
                    and (now - self.fsm_last_decision_time) >= self.fsm_cruise_decide_interval_sec):
                self.fsm_enter("SOCIAL_STOP")
                return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", \
                    f"periodic DECIDE interval={self.fsm_cruise_decide_interval_sec:.2f}s reached"
            if latched:
                return (self.forward_speed, 0.0, 0.0), "FORWARD", "VLM", \
                    (f"VLM latch active {max(0.0, self._fsm_latch_until - now):.2f}s "
                     f"({self._fsm_latch_reason}); front={front_distance:.2f}m but trigger suppressed")
            if self.fsm_disable_lidar_trigger:
                return (self.forward_speed, 0.0, 0.0), "FORWARD", "VLM", \
                    f"lidar_trigger disabled; cruising (front={front_distance:.2f}m)"
            return (self.forward_speed, 0.0, 0.0), "FORWARD", "LIDAR", \
                f"front_dist={front_distance:.2f}m >= trigger={self.fsm_person_trigger_dist:.2f}m -> CRUISE"

        if state == "SOCIAL_STOP":
            # Dwell to stabilize perception, then DECIDE.
            if (time.time() - self.fsm_state_entered) >= self.fsm_social_stop_settle_sec:
                self.fsm_enter("DECIDE")
            return (0.0, 0.0, 0.0), "STOP", "FSM_SETTLE", \
                f"settle {time.time()-self.fsm_state_entered:.2f}/{self.fsm_social_stop_settle_sec:.2f}s"

        if state == "DECIDE":
            # Respect decision cooldown so we do not spam the wrapper.
            if (time.time() - self.fsm_last_decision_time) < self.fsm_decide_cooldown_sec:
                return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", \
                    f"cooldown {(time.time()-self.fsm_last_decision_time):.2f}/{self.fsm_decide_cooldown_sec:.2f}s"
            frames = self.fsm_pick_sequence()
            if frames is None:
                # Not enough buffered frames yet; stay stopped.
                return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", \
                    "waiting for frames in ring buffer"
            action, rj, envelope, latency = self.fsm_call_sequence_wrapper(frames)
            self.fsm_last_decision_time = time.time()
            raw_action = (rj or {}).get("recommended_action", "") if rj else ""
            # Cache the latest VLM reasoning for subsequent debug ticks.
            self._cache_vlm_decision(action, rj, latency)
            if action is None:
                # Wrapper failure OR unrecognized action: stay conservative.
                rospy.logwarn("[social_fsm] decide failed; holding")
                self.log_vlm_decision(
                    state_entered_from="DECIDE",
                    front_dist=self.fsm_last_front_dist,
                    num_images=len(frames),
                    action_normalized=None,
                    action_raw=raw_action,
                    response_json=rj,
                    envelope=envelope,
                    latency_sec=latency,
                    fsm_next_state="HOLD_REVIEW",
                    robot_actuation="STOP",
                )
                self.fsm_enter("HOLD_REVIEW")
                return (0.0, 0.0, 0.0), "STOP", "VLM", \
                    f"wrapper_failed err={str((envelope or {}).get('error',''))[:80]}"
            rospy.loginfo(
                "[social_fsm] decision=%s motion=%s cross_dir=%s avoid=%s unc=%s",
                action,
                rj.get("motion"),
                rj.get("crossing_direction"),
                rj.get("recommended_avoidance_side"),
                (rj.get("uncertainty_reason") or "")[:80],
            )
            projected_action, projected_rj, projection_note = project_realtime_action(
                action,
                rj,
                allow_lateral=self.fsm_allow_vlm_lateral,
            )
            if projected_action != action:
                rospy.logwarn(
                    "[social_fsm] VLM suggested %s (side=%s) — NOT executing: %s",
                    action,
                    (rj or {}).get("recommended_avoidance_side"),
                    projection_note or "downgrading to STOP",
                )
                action = projected_action
                rj = projected_rj
                self._cache_vlm_decision(action, rj, latency)
            self.fsm_last_action = action
            if action == "REVIEW":
                self.log_vlm_decision(
                    state_entered_from="DECIDE", front_dist=self.fsm_last_front_dist,
                    num_images=len(frames), action_normalized=action, action_raw=raw_action,
                    response_json=rj, envelope=envelope, latency_sec=latency,
                    fsm_next_state="HOLD_REVIEW", robot_actuation="STOP",
                )
                self.fsm_enter("HOLD_REVIEW")
                return (0.0, 0.0, 0.0), "REVIEW", "VLM", \
                    f"REVIEW motion={rj.get('motion')} unc={(rj.get('uncertainty_reason') or '')[:60]}"
            if action == "STOP":
                self.log_vlm_decision(
                    state_entered_from="DECIDE", front_dist=self.fsm_last_front_dist,
                    num_images=len(frames), action_normalized=action, action_raw=raw_action,
                    response_json=rj, envelope=envelope, latency_sec=latency,
                    fsm_next_state="SOCIAL_STOP", robot_actuation="STOP",
                )
                self.fsm_enter("SOCIAL_STOP")
                return (0.0, 0.0, 0.0), "STOP", "VLM", \
                    f"STOP motion={rj.get('motion')} blocked={rj.get('path_blocked_latest_frame')}"
            # FORWARD / LEFT / RIGHT: execute a short guarded maneuver.
            # Set the VLM-clearance latch. While active, CRUISE ignores the
            # LiDAR person-trigger so the VLM's decision to keep moving is
            # not immediately overridden by a close LiDAR return. Hard-stop
            # always applies.
            if self.fsm_vlm_latch_sec > 0.0 and action in ("FORWARD", "LEFT", "RIGHT"):
                blocked = rj.get("path_blocked_latest_frame", None)
                # For FORWARD we require blocked=False (VLM says clear);
                # for LEFT/RIGHT we latch regardless because the whole point
                # of the chosen lateral is to clear the blockage.
                if action != "FORWARD" or blocked is False:
                    self._fsm_latch_until = time.time() + self.fsm_vlm_latch_sec
                    self._fsm_latch_reason = (
                        f"VLM={action} blocked={blocked}"
                    )
            # Track the lateral streak (used by fsm_force_decide_after_lateral
            # to cap infinite strafe-loops). Reset on FORWARD (escape hatch).
            if action in ("LEFT", "RIGHT"):
                self._fsm_lateral_streak += 1
            elif action == "FORWARD":
                self._fsm_lateral_streak = 0
            self.fsm_begin_primitive(action)
            self.fsm_enter("EXECUTE")
            act_desc = self.fsm_describe_actuation(action)
            self.log_vlm_decision(
                state_entered_from="DECIDE", front_dist=self.fsm_last_front_dist,
                num_images=len(frames), action_normalized=action, action_raw=raw_action,
                response_json=rj, envelope=envelope, latency_sec=latency,
                fsm_next_state="EXECUTE", robot_actuation=act_desc,
            )
            return (0.0, 0.0, 0.0), action, "VLM", \
                f"{action} motion={rj.get('motion')} cross_dir={rj.get('crossing_direction')} avoid={rj.get('recommended_avoidance_side')}"

        if state == "EXECUTE":
            vel = self.fsm_tick_primitive()
            if vel is None:
                finished_action = self.fsm_exec_primitive
                self.fsm_exec_primitive = None
                # OPTION X (new): after a lateral EXECUTE, force a DECIDE so
                # the VLM decides whether the forward corridor is now clear.
                # Capped by fsm_lateral_streak_max to prevent infinite
                # strafe-loops in unresolvable scenes.
                if (finished_action in ("LEFT", "RIGHT")
                        and self.fsm_force_decide_after_lateral):
                    if self._fsm_lateral_streak >= self.fsm_lateral_streak_max:
                        rospy.logwarn(
                            "[social_fsm] lateral streak cap %d reached; "
                            "stopping to let operator intervene",
                            self.fsm_lateral_streak_max,
                        )
                        self._fsm_lateral_streak = 0
                        self.fsm_enter("HOLD_REVIEW")
                        return (0.0, 0.0, 0.0), "REVIEW", "VLM_WAIT", \
                            f"lateral streak cap {self.fsm_lateral_streak_max} hit"
                    # Reset cooldown so DECIDE fires immediately next tick.
                    self.fsm_last_decision_time = 0.0
                    self.fsm_lateral_chain_count = 0
                    self.fsm_enter("DECIDE")
                    return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", \
                        (f"post-lateral DECIDE (streak {self._fsm_lateral_streak}/"
                         f"{self.fsm_lateral_streak_max}): asking VLM if forward clear")
                # Legacy lateral chaining (LiDAR-based): chain same-direction
                # lateral while LiDAR still reports trigger-range obstacle.
                if (finished_action in ("LEFT", "RIGHT")
                        and not self.fsm_force_decide_after_lateral
                        and front_distance < self.fsm_person_trigger_dist
                        and self.fsm_lateral_chain_count < self.fsm_lateral_chain_max):
                    self.fsm_lateral_chain_count += 1
                    self.fsm_last_lateral_action = finished_action
                    self.fsm_begin_primitive(finished_action)
                    return (0.0, 0.0, 0.0), finished_action, "VLM", \
                        f"chaining {finished_action} #{self.fsm_lateral_chain_count}/{self.fsm_lateral_chain_max} (front={front_distance:.2f}m still < trigger)"
                # Default: lateral/FORWARD done → CRUISE, reset counters.
                self.fsm_lateral_chain_count = 0
                self._fsm_lateral_streak = 0
                self.fsm_enter("CRUISE")
                return (0.0, 0.0, 0.0), "STOP", "VLM", "execute primitive finished"
            return vel, self.fsm_last_action, "VLM", \
                f"executing {self.fsm_last_action} (cached motion={self.fsm_last_vlm.get('motion')}, chain {self.fsm_lateral_chain_count}/{self.fsm_lateral_chain_max})"

        if state == "HOLD_REVIEW":
            # Option-B: track how long LiDAR has reported the corridor clear.
            now = time.time()
            if front_distance >= self.fsm_hold_review_clear_dist:
                if self._hold_review_clear_since is None:
                    self._hold_review_clear_since = now
                clear_for = now - self._hold_review_clear_since
            else:
                self._hold_review_clear_since = None
                clear_for = 0.0
            # Auto-recovery: if LiDAR has been continuously clear long enough,
            # reset review counter and return to CRUISE regardless of retries
            # state. Safe because it requires LiDAR confirmation, and the
            # hard-safety override still applies on every tick.
            if clear_for >= self.fsm_hold_review_clear_sec:
                rospy.loginfo(
                    "[social_fsm] HOLD_REVIEW auto-recover: clear for %.2fs >= %.2fs -> CRUISE",
                    clear_for,
                    self.fsm_hold_review_clear_sec,
                )
                self.fsm_review_count = 0
                self._hold_review_clear_since = None
                self.fsm_enter("CRUISE")
                return (0.0, 0.0, 0.0), "STOP", "LIDAR", \
                    f"LiDAR clear {clear_for:.2f}s >= {self.fsm_hold_review_clear_sec:.2f}s at front={front_distance:.2f}m -> CRUISE"
            held = now - self.fsm_state_entered
            if held < self.fsm_review_hold_sec:
                return (0.0, 0.0, 0.0), "REVIEW", "VLM_WAIT", \
                    f"HOLD_REVIEW {held:.2f}/{self.fsm_review_hold_sec:.2f}s clear_for={clear_for:.2f}s last_unc={(self.fsm_last_vlm.get('uncertainty_reason') or '')[:40]}"
            if self.fsm_review_count < self.fsm_review_retries:
                self.fsm_review_count += 1
                self.fsm_enter("DECIDE")
                return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", \
                    f"retry {self.fsm_review_count}/{self.fsm_review_retries}"
            # Retries exhausted but LiDAR still not clear for long enough:
            # stay stopped. Auto-recovery above will fire as soon as LiDAR
            # reports a sustained clear corridor.
            rospy.logwarn_throttle(
                5.0,
                "[social_fsm] REVIEW persisted after %d retries; clear_for=%.2fs of %.2fs needed",
                self.fsm_review_count,
                clear_for,
                self.fsm_hold_review_clear_sec,
            )
            return (0.0, 0.0, 0.0), "REVIEW", "VLM_WAIT", \
                f"retries exhausted; waiting for LiDAR clear ({clear_for:.2f}s / {self.fsm_hold_review_clear_sec:.2f}s need)"

        # Unknown state: safest thing to do is stop.
        rospy.logerr("[social_fsm] unknown state %s; forcing SOCIAL_STOP", state)
        self.fsm_enter("SOCIAL_STOP")
        return (0.0, 0.0, 0.0), "STOP", "FSM_SETTLE", f"unknown state {state}; reset"

    def _cache_vlm_decision(self, action, rj, latency):
        if not rj:
            return
        self.fsm_last_vlm = {
            "action": action,
            "motion": rj.get("motion"),
            "crossing_direction": rj.get("crossing_direction"),
            "safer_lateral_side": rj.get("safer_lateral_side"),
            "recommended_avoidance_side": rj.get("recommended_avoidance_side"),
            "risk_level": rj.get("risk_level"),
            "path_blocked_latest_frame": rj.get("path_blocked_latest_frame"),
            "uncertainty_reason": (rj.get("uncertainty_reason") or "")[:200],
            "latency_sec": latency,
            "timestamp": time.time(),
        }

    def run_social_fsm(self):
        """Dedicated run loop for the new state-machine controller. Keeps the
        original ``run`` entrypoint untouched."""
        rate = rospy.Rate(10)
        self.fsm_enter("CRUISE")
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            # Same LiDAR-freshness gate as the legacy run loop.
            if self.latest_scan is None or self.latest_scan_stamp is None:
                self.send_stop()
                rospy.logwarn_throttle(5.0, "Waiting for scan on %s", self.scan_topic)
                rate.sleep(); continue
            if (now - self.latest_scan_stamp).to_sec() > self.scan_timeout:
                self.send_stop()
                rospy.logwarn_throttle(5.0, "Scan timeout on %s", self.scan_topic)
                rate.sleep(); continue

            _tick_start = time.time()
            front_distance = self.compute_front_distance(self.latest_scan)
            self.fsm_last_front_dist = front_distance
            (vx, vy, wz), action, source, reason = self.fsm_step(front_distance)

            # Edge-triggered UDP output: only send on transitions + a low-rate
            # "refresh" when moving so the NX1 bridge's 500 ms no-command
            # watchdog never idles us mid-motion. When continuously stopped,
            # we stay silent and let the bridge's watchdog keep the dog idle.
            # This eliminates the "STOPPED STOPPED STOPPED..." bridge output
            # during long DECIDE / HOLD_REVIEW phases while keeping safety.
            want_stop = (action in ("STOP", "REVIEW", "HARD_STOP")
                         or (vx == 0.0 and vy == 0.0 and wz == 0.0))
            if want_stop:
                if getattr(self, "_last_udp_kind", None) != "stop":
                    self.send_stop()
                    self._last_udp_kind = "stop"
                # else: stay silent; bridge watchdog handles idle.
            else:
                now = time.time()
                # Re-send the command (a) on transition from stop/other, or
                # (b) when the vector changed meaningfully, or (c) every
                # 300 ms as a watchdog refresh while continuously moving.
                cur = (round(vx, 3), round(vy, 3), round(wz, 3))
                last_vec = getattr(self, "_last_udp_vec", None)
                last_time = getattr(self, "_last_udp_time", 0.0)
                if (self._last_udp_kind != "move"
                        or last_vec != cur
                        or (now - last_time) > 0.3):
                    self.send_velocity(vx, vy, wz)
                    self._last_udp_kind = "move"
                    self._last_udp_vec = cur
                    self._last_udp_time = now

            state = {
                "mode": self.mode,
                "vision_source": "social_fsm",
                "vision_frequency": 0.0,
                "vision_reference_time": time.time(),
                "vision_age": 0.0,
                "vision_stale": False,
                "front_dist": front_distance,
                "person_detected": False,
                "person_in_front": False,
                "active_threshold": self.fsm_person_trigger_dist,
                "threshold_source": f"fsm:{self.fsm_state}",
                "action": action,
            }
            self.debug_state_pub.publish(String(data=json.dumps(state, sort_keys=True)))
            if not self.fsm_disable_per_tick_csv:
                self.maybe_log_csv(state)
            self.maybe_log_state(state)
            if not self.fsm_disable_debug_image:
                self.publish_debug_image(state)

            # --- Realtime debug channel: one compact JSON per tick so the
            # operator can see exactly who (LiDAR vs VLM vs HARD_SAFETY)
            # chose the current action, with the currently-cached VLM
            # reasoning attached even on LiDAR ticks.
            debug = {
                "t": round(time.time(), 3),
                "state": self.fsm_state,
                "source": source,            # LIDAR / VLM / VLM_WAIT / HARD_SAFETY / FSM_SETTLE
                "action": action,
                "cmd": {"vx": round(vx, 3), "vy": round(vy, 3), "wz": round(wz, 3)},
                "front_dist": round(front_distance, 3),
                "trigger": self.fsm_person_trigger_dist,
                "hard_stop": self.fsm_hard_stop_dist,
                "reason": reason,
                "last_vlm": self.fsm_last_vlm,
            }
            try:
                self.fsm_debug_pub.publish(String(data=json.dumps(debug, sort_keys=True)))
            except Exception:
                pass

            # Color-coded stdout at a throttled cadence so it is actually
            # readable at 10 Hz. ANSI: red=HARD_SAFETY, yellow=LIDAR-stop,
            # green=LIDAR-FORWARD, cyan=VLM motion, magenta=VLM STOP/REVIEW.
            ansi = {
                "HARD_SAFETY": "\033[41;37m",     # red bg, white fg
                "LIDAR_FWD": "\033[32m",          # green fg
                "LIDAR_STOP": "\033[33m",         # yellow fg
                "VLM_MOVE": "\033[36m",           # cyan fg
                "VLM_STOP": "\033[35m",           # magenta fg
                "VLM_WAIT": "\033[90m",           # dim gray
                "FSM_SETTLE": "\033[90m",         # dim gray
                "RESET": "\033[0m",
            }
            if source == "HARD_SAFETY":
                color = ansi["HARD_SAFETY"]
            elif source == "LIDAR":
                color = ansi["LIDAR_FWD"] if action == "FORWARD" else ansi["LIDAR_STOP"]
            elif source == "VLM":
                color = ansi["VLM_MOVE"] if action in ("FORWARD", "LEFT", "RIGHT") else ansi["VLM_STOP"]
            else:
                color = ansi[source] if source in ansi else ""
            # Throttle: print on every state change OR every 0.5s otherwise.
            now_sec = time.time()
            sig = (self.fsm_state, source, action)
            if not hasattr(self, "_fsm_debug_last_sig") \
               or self._fsm_debug_last_sig != sig \
               or (now_sec - getattr(self, "_fsm_debug_last_time", 0.0)) > 0.5:
                self._fsm_debug_last_sig = sig
                self._fsm_debug_last_time = now_sec
                vlm_tail = ""
                if source == "VLM" or source == "VLM_WAIT":
                    mo = self.fsm_last_vlm.get("motion")
                    cd = self.fsm_last_vlm.get("crossing_direction")
                    av = self.fsm_last_vlm.get("recommended_avoidance_side")
                    bl = self.fsm_last_vlm.get("path_blocked_latest_frame")
                    rk = self.fsm_last_vlm.get("risk_level")
                    uc = (self.fsm_last_vlm.get("uncertainty_reason") or "")[:40]
                    vlm_tail = f" | VLM[motion={mo} cross_dir={cd} avoid={av} blocked={bl} risk={rk} unc={uc!r}]"
                line = (
                    f"{color}[{self.fsm_state:>11s}] source={source:<11s} "
                    f"action={action:<8s} cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f}) "
                    f"front={front_distance:.2f}m | {reason}{vlm_tail}{ansi['RESET']}"
                )
                print(line, flush=True)

            # Per-tick wall-clock profile: accumulate tick durations and
            # print max/avg/p95 every ``fsm_profile_interval`` seconds.
            if self.fsm_profile_loop:
                self._fsm_loop_samples.append(time.time() - _tick_start)
                if (time.time() - self._fsm_profile_last) >= self.fsm_profile_interval:
                    s = sorted(self._fsm_loop_samples)
                    n = len(s)
                    if n > 0:
                        avg = sum(s) / n
                        mx = s[-1]
                        p95 = s[int(0.95 * (n - 1))]
                        print(
                            f"\033[96m[profile] tick_count={n} avg={avg*1000:.1f}ms "
                            f"p95={p95*1000:.1f}ms max={mx*1000:.1f}ms "
                            f"(disable_debug_image={self.fsm_disable_debug_image} "
                            f"disable_csv={self.fsm_disable_per_tick_csv} "
                            f"disable_log={self.fsm_disable_text_log})\033[0m",
                            flush=True,
                        )
                    self._fsm_loop_samples = []
                    self._fsm_profile_last = time.time()

            rate.sleep()

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if not self.vlm_only_debug and (self.latest_scan is None or self.latest_scan_stamp is None):
                self.send_stop()
                rospy.logwarn_throttle(5.0, "Waiting for scan on %s", self.scan_topic)
                rate.sleep()
                continue

            if not self.vlm_only_debug and (now - self.latest_scan_stamp).to_sec() > self.scan_timeout:
                self.send_stop()
                rospy.logwarn_throttle(5.0, "Scan timeout on %s", self.scan_topic)
                rate.sleep()
                continue

            if self.vlm_only_debug:
                front_distance = self.max_range_fallback
            else:
                front_distance = self.compute_front_distance(self.latest_scan)
            active_threshold = self.baseline_stop_dist
            with self.state_lock:
                person_detected = self.person_detected
                person_in_front = self.person_in_front
                vision_source = self.vision_source
                last_valid_detection_stamp = self.last_valid_detection_stamp
                has_successful_vision_result = self.has_successful_vision_result
                vision_frequency = self.vision_frequency
                last_vision_update_time = self.last_vision_update_time
                vision_valid = self.vision_valid
            threshold_source = "baseline_stop_dist"
            vision_age = 0.0
            vision_stale = True
            vision_reference_time = (
                last_vision_update_time.to_sec() if last_vision_update_time is not None else 0.0
            )

            if self.mode == "human_aware":
                if last_valid_detection_stamp is None or not vision_valid:
                    person_detected = False
                    person_in_front = False
                    vision_source = "no_vision_cache"
                else:
                    vision_age = max(0.0, (now - last_valid_detection_stamp).to_sec())
                    vision_stale = vision_age > self.vision_stale_timeout
                if last_valid_detection_stamp is None or vision_stale:
                    person_detected = False
                    person_in_front = False
                    if last_valid_detection_stamp is not None:
                        vision_source = "stale_cache"
                        if time.time() - self.last_stale_log_time > self.debug_log_interval:
                            rospy.logwarn(
                                "[social_nav] vision cache stale age=%.2f timeout=%.2f source=%s; falling back to baseline",
                                vision_age,
                                self.vision_stale_timeout,
                                self.vision_source,
                            )
                            self.last_stale_log_time = time.time()
                    vision_stale = True
                else:
                    vision_stale = False

            if self.mode == "human_aware" and self.hold_until_first_vision and not has_successful_vision_result:
                action = "STOP"
                threshold_source = "waiting_for_vision"
                vision_source = "waiting_for_first_vision"
                if time.time() - self.last_wait_for_vision_log_time > self.debug_log_interval:
                    rospy.logwarn(
                        "[social_nav] waiting for first successful vision result before motion"
                    )
                    self.last_wait_for_vision_log_time = time.time()
            elif self.mode == "human_aware" and self.vlm_only_debug:
                active_threshold = self.social_stop_dist if person_detected else self.baseline_stop_dist
                threshold_source = (
                    "vlm_only_human_stop" if person_detected else "vlm_only_baseline"
                )
                action = "STOP" if (not vision_stale and person_detected) else "FORWARD"
            elif front_distance < self.baseline_stop_dist:
                active_threshold = self.baseline_stop_dist
                threshold_source = "baseline_stop_dist"
                action = "STOP"
            elif self.mode == "human_aware" and not vision_stale and (person_detected or person_in_front):
                active_threshold = self.social_stop_dist
                threshold_source = "social_stop_dist"
                if front_distance < self.social_stop_dist:
                    action = "STOP"
                else:
                    action = "FORWARD"
            else:
                active_threshold = self.baseline_stop_dist
                threshold_source = "baseline_stop_dist"
                action = "FORWARD"

            if action == "STOP":
                self.send_stop()
            else:
                self.send_velocity(self.forward_speed, 0.0, 0.0)

            self.loop_count += 1
            state = {
                "mode": self.mode,
                "vision_source": vision_source,
                "vision_frequency": vision_frequency,
                "vision_reference_time": vision_reference_time,
                "vision_age": vision_age,
                "vision_stale": vision_stale,
                "front_dist": front_distance,
                "person_detected": person_detected,
                "person_in_front": person_in_front,
                "active_threshold": active_threshold,
                "threshold_source": threshold_source,
                "action": action,
            }

            self.debug_state_pub.publish(String(data=json.dumps(state, sort_keys=True)))
            self.maybe_log_csv(state)

            if self.loop_count % self.log_every_n == 0:
                self.maybe_log_state(state)

            self.publish_debug_image(state)

            rate.sleep()


def main():
    controller = SocialNavController()
    if controller.mode == "social_fsm":
        controller.run_social_fsm()
    else:
        controller.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        sys.exit(0)
