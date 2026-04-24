#!/usr/bin/env python3
"""
VLM-only social navigation controller for Unitree Go1 (no LiDAR).

This controller is a lean companion to social_nav_controller.py in the same
directory. It removes every LiDAR code path so that all high-level
decisions (STOP / FORWARD / LEFT / RIGHT / REVIEW / BACKWARD) come from the
VLM wrapper. The only sensor it subscribes to is the front camera image
topic.

SAFETY NOTE
-----------
Because this controller has NO hard-stop safety sensor, the operator MUST
keep an ESTOP reachable at all times. The NX1 bridge still has a 500 ms
no-command watchdog that idles the dog if commands stop flowing, which is
the only hardware safety net this controller relies on.

States
------
  CRUISE       : default forward motion; drops into DECIDE on a periodic
                 timer so the VLM can intervene.
  DECIDE       : collect a short image sequence from the ring buffer and
                 call the VLM wrapper's /analyze_navigation endpoint.
  EXECUTE      : run a short guarded motion primitive for the current
                 action (FORWARD / BACKWARD / LEFT / RIGHT).
  HOLD_REVIEW  : VLM returned REVIEW; hold briefly, then retry up to
                 ``fsm_review_retries`` times. An optional BACKWARD
                 recovery primitive can be emitted when the FSM stays in
                 HOLD_REVIEW past the retry cap.

Lateral modes (~fsm_lateral_mode)
---------------------------------
  turn_only   : pure yaw, no translation (rotate in place to find a clear
                forward view).
  strafe      : pure lateral ``vy`` sidestep.
  blend       : diagonal ``vx + vy``.
  turn_creep  : yaw burst followed by forward creep (original default).

Backward primitive (~fsm_back_enabled + friends)
------------------------------------------------
  Opt-in BACKWARD motion for recovery scenarios. Triggered either
  explicitly by the VLM returning ``BACK`` / ``BACKWARD`` (tolerated via
  aliases) or implicitly after ``fsm_back_after_n_stuck`` consecutive
  STOP / REVIEW decisions while blocked.

Per-decision VLM log
--------------------
  motion_control/logs/vlm_only_<ts>.{csv,jsonl} — one row per VLM call
  with motion, crossing_direction, recommended_avoidance_side,
  uncertainty_reason, wrapper_ok, latency_sec, and the robot actuation
  that was applied.
"""
import base64
import csv
import io
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import requests
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

# Make the repo-root `social_nav_eval_prompts` importable so we can reuse the
# canonical social-navigation sequence prompt verbatim when building a
# goal-augmented prompt for direct-backend calls.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from social_nav_eval_prompts import PROMPT_SEQUENCE_IMAGES as _BASE_SEQ_PROMPT
except Exception:  # pragma: no cover - fall back if import path differs
    _BASE_SEQ_PROMPT = None
from social_nav_policy import project_realtime_action

try:
    from cv_bridge import CvBridge, CvBridgeError
except ImportError:  # pragma: no cover - depends on ROS env
    CvBridge = None
    CvBridgeError = Exception


ACTION_ALIASES = {
    "UNKNOWN": "REVIEW", "UNSURE": "REVIEW", "UNCERTAIN": "REVIEW",
    "WAIT": "REVIEW", "HOLD": "REVIEW", "DEFER": "REVIEW",
    "GO_LEFT": "LEFT", "STEP_LEFT": "LEFT", "SIDESTEP_LEFT": "LEFT",
    "TURN_LEFT": "LEFT", "AVOID_LEFT": "LEFT",
    "GO_RIGHT": "RIGHT", "STEP_RIGHT": "RIGHT", "SIDESTEP_RIGHT": "RIGHT",
    "TURN_RIGHT": "RIGHT", "AVOID_RIGHT": "RIGHT",
    "HALT": "STOP", "YIELD": "STOP",
    "CONTINUE": "FORWARD", "GO": "FORWARD", "PROCEED": "FORWARD",
    "BACK": "BACKWARD", "REVERSE": "BACKWARD", "GO_BACK": "BACKWARD",
}
VALID_ACTIONS = {"STOP", "FORWARD", "LEFT", "RIGHT", "REVIEW", "BACKWARD"}


def normalize_action(value):
    if value is None:
        return None
    act = str(value).strip().upper()
    act = ACTION_ALIASES.get(act, act)
    return act if act in VALID_ACTIONS else None


class VlmOnlyController:
    def __init__(self):
        rospy.init_node("vlm_only_controller", anonymous=False)

        # --- Transport ---
        self.image_topic = rospy.get_param("~image_topic", "/camera_face/left/image_raw")
        self.nx1_ip = rospy.get_param("~nx1_ip", "192.168.123.15")
        self.nx1_port = int(rospy.get_param("~nx1_port", 9900))
        self.wrapper_url = rospy.get_param(
            "~fsm_nav_wrapper_url",
            "http://10.157.141.10:8100/analyze_navigation",
        )
        self.prompt_name = rospy.get_param("~fsm_seq_prompt_name", "sequence_image_navigation")
        self.connect_timeout = float(rospy.get_param("~fsm_connect_timeout", 3.0))
        self.request_timeout = float(rospy.get_param("~fsm_request_timeout_sec", 8.0))

        # --- Motion speeds ---
        self.forward_speed = float(rospy.get_param("~forward_speed", 0.2))
        self.backward_speed = float(rospy.get_param("~fsm_backward_speed", 0.12))
        self.strafe_speed = float(rospy.get_param("~fsm_strafe_speed", 0.15))
        self.yaw_burst_speed = float(rospy.get_param("~fsm_yaw_burst_speed", 0.5))
        self.blend_forward_speed = float(rospy.get_param("~fsm_blend_forward_speed", 0.1))
        self.creep_speed = float(rospy.get_param("~fsm_creep_speed", 0.15))

        # --- Primitive durations ---
        self.forward_burst_sec = float(rospy.get_param("~fsm_forward_burst_sec", 0.6))
        self.backward_burst_sec = float(rospy.get_param("~fsm_backward_burst_sec", 0.5))
        self.strafe_sec = float(rospy.get_param("~fsm_strafe_sec", 0.8))
        self.yaw_burst_sec = float(rospy.get_param("~fsm_yaw_burst_sec", 0.6))
        self.creep_after_turn_sec = float(rospy.get_param("~fsm_creep_after_turn_sec", 0.6))

        # --- Lateral mode ---
        self.lateral_mode = rospy.get_param("~fsm_lateral_mode", "turn_only").strip().lower()
        if self.lateral_mode not in ("turn_creep", "strafe", "blend", "turn_only"):
            rospy.logwarn("unknown fsm_lateral_mode=%r; defaulting to turn_only", self.lateral_mode)
            self.lateral_mode = "turn_only"

        # --- FSM timing ---
        self.cruise_decide_interval_sec = float(rospy.get_param("~fsm_cruise_decide_interval_sec", 3.0))
        self.decide_cooldown_sec = float(rospy.get_param("~fsm_decide_cooldown_sec", 0.5))
        self.review_hold_sec = float(rospy.get_param("~fsm_review_hold_sec", 3.0))
        self.review_retries = int(rospy.get_param("~fsm_review_retries", 2))
        self.force_decide_after_lateral = bool(rospy.get_param("~fsm_force_decide_after_lateral", True))
        self.force_decide_after_forward = bool(rospy.get_param("~fsm_force_decide_after_forward", False))
        self.lateral_streak_max = int(rospy.get_param("~fsm_lateral_streak_max", 10))
        # After a VLM STOP decision, hold a stationary pose for this many
        # seconds before CRUISE is allowed to emit FORWARD again.
        self.stop_hold_sec = float(rospy.get_param("~fsm_stop_hold_sec", 2.0))
        self._fsm_last_stop_time = 0.0
        # Safety gate: VLM LEFT / RIGHT outputs are advisory only by
        # default. The VLM has no geometric / free-space verification,
        # so turning a high-level "bypass" hint into an executable vy
        # strafe is unsafe. When this flag is False (default), a VLM
        # LEFT / RIGHT is treated as STOP at the controller and the
        # suggested side is logged for operator review only.
        self.allow_vlm_lateral = bool(rospy.get_param("~fsm_allow_vlm_lateral", False))

        # --- Sequence sampling ---
        self.ring_size = int(rospy.get_param("~fsm_ring_size", 30))
        self.seq_length = int(rospy.get_param("~fsm_seq_length", 2))
        self.seq_past_min_sec = float(rospy.get_param("~fsm_seq_past_min_sec", 1.0))
        self.seq_past_max_sec = float(rospy.get_param("~fsm_seq_past_max_sec", 2.0))
        self.seq_span_sec = float(rospy.get_param("~fsm_seq_span_sec", 2.0))

        # --- Backward recovery ---
        self.back_enabled = bool(rospy.get_param("~fsm_back_enabled", False))
        self.back_after_n_stuck = int(rospy.get_param("~fsm_back_after_n_stuck", 3))

        # --- Goal-augmented prompting ---
        # When ``goal_description`` is non-empty the controller bypasses
        # the wrapper's fixed /analyze_navigation prompt and calls the
        # vLLM OpenAI-chat endpoint directly with a prompt that is the
        # canonical social-nav sequence prompt + a "Primary navigation
        # goal" block + a "goal-reached reporting" block. All social
        # safety rules still apply; the goal is only a task bias on top.
        self.goal_description = rospy.get_param("~goal_description", "").strip()
        self.backend_url = rospy.get_param(
            "~backend_url", "http://10.157.141.181:8000/v1/chat/completions"
        )
        self.backend_model = rospy.get_param(
            "~backend_model", "OpenGVLab/InternVL3_5-14B-HF"
        )
        if self.goal_description and _BASE_SEQ_PROMPT is None:
            rospy.logerr(
                "[vlm_only] goal_description set but social_nav_eval_prompts "
                "could not be imported; direct-backend path unavailable"
            )
        self._goal_reached = False

        # --- Logging controls ---
        self.disable_per_tick_csv = bool(rospy.get_param("~fsm_disable_per_tick_csv", True))
        self.profile_loop = bool(rospy.get_param("~fsm_profile_loop", False))
        self.profile_interval = float(rospy.get_param("~fsm_profile_interval", 5.0))
        self._profile_samples = []
        self._profile_last = time.time()

        # --- Debug topic ---
        self.fsm_debug_topic = rospy.get_param("~fsm_debug_topic", "/vlm_only/fsm_debug")
        self.fsm_debug_pub = rospy.Publisher(self.fsm_debug_topic, String, queue_size=1)

        # --- State ---
        self.stop_event = threading.Event()
        self.frames = deque(maxlen=self.ring_size)
        self.frames_lock = threading.Lock()
        self.fsm_state = "CRUISE"
        self.fsm_state_entered = time.time()
        self.fsm_last_action = None
        self.fsm_exec_primitive = None
        self.fsm_exec_started = 0.0
        self.fsm_exec_until = 0.0
        self.fsm_last_decision_time = 0.0
        self.fsm_review_count = 0
        self._fsm_lateral_streak = 0
        self._fsm_stuck_count = 0
        self._last_vlm = {
            "action": None, "motion": None, "crossing_direction": None,
            "safer_lateral_side": None, "recommended_avoidance_side": None,
            "risk_level": None, "path_blocked_latest_frame": None,
            "uncertainty_reason": None, "latency_sec": None,
        }
        # Edge-triggered UDP
        self._last_udp_kind = None
        self._last_udp_vec = None
        self._last_udp_time = 0.0

        # --- UDP socket ---
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.nx_addr = (self.nx1_ip, self.nx1_port)

        # --- cv_bridge + image sub ---
        if CvBridge is None:
            raise RuntimeError("cv_bridge unavailable; required for vlm_only_controller")
        self.bridge = CvBridge()
        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)

        # --- Logs ---
        self.logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.vlm_jsonl_path = os.path.join(self.logs_dir, f"vlm_only_{ts}.jsonl")
        self.vlm_csv_path = os.path.join(self.logs_dir, f"vlm_only_{ts}.csv")
        self.vlm_jsonl = open(self.vlm_jsonl_path, "w")
        self.vlm_csv_file = open(self.vlm_csv_path, "w", newline="")
        self.vlm_csv = csv.writer(self.vlm_csv_file)
        self.vlm_csv.writerow([
            "timestamp", "fsm_state_entered_from", "num_images",
            "wrapper_url", "prompt_name", "goal", "action_normalized", "action_raw",
            "motion", "crossing_direction", "safer_lateral_side",
            "recommended_avoidance_side", "risk_level",
            "path_blocked_latest_frame", "uncertainty_reason",
            "person_bbox_height_frac", "person_position",
            "goal_visible", "goal_reached",
            "goal_bbox_height_frac", "goal_position", "reasoning",
            "wrapper_ok", "wrapper_error", "fsm_next_state",
            "robot_actuation", "latency_sec",
        ])
        self.vlm_csv_file.flush()

        rospy.on_shutdown(self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        if self.goal_description:
            rospy.logwarn(
                "[vlm_only] GOAL MODE active: goal=%r backend=%s model=%s. "
                "Bypassing /analyze_navigation and sending full prompt inline.",
                self.goal_description, self.backend_url, self.backend_model,
            )
        rospy.logwarn(
            "[vlm_only] NO LIDAR AND NO HARD-STOP SAFETY SENSOR. "
            "Keep operator ESTOP reachable. wrapper=%s lateral_mode=%s "
            "back_enabled=%s goal=%r",
            self.wrapper_url, self.lateral_mode, self.back_enabled,
            self.goal_description or "",
        )
        print(f"[vlm_only] VLM decision log -> {self.vlm_csv_path}")

    # --------------------------------------------------------- transport
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
            for attr in ("vlm_jsonl", "vlm_csv_file"):
                try:
                    getattr(self, attr).close()
                except Exception:
                    pass
            try:
                self.sock.close()
            except Exception:
                pass

    # --------------------------------------------------------- camera
    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, "cv_bridge conversion failed: %s", exc)
            return
        with self.frames_lock:
            self.frames.append((time.time(), frame.copy()))

    # --------------------------------------------------------- sequence
    def pick_sequence(self):
        with self.frames_lock:
            snapshot = list(self.frames)
        if self.seq_length <= 0 or not snapshot:
            return None
        if self.seq_length == 2:
            now = time.time()
            newest_t, newest_f = snapshot[-1]
            past_f = None
            for (t, f) in snapshot:
                age = now - t
                if self.seq_past_min_sec <= age <= self.seq_past_max_sec:
                    past_f = f
                    break
            if past_f is None:
                oldest_t, oldest_f = snapshot[0]
                if (now - oldest_t) >= self.seq_past_min_sec:
                    past_f = oldest_f
                else:
                    return None
            return [past_f, newest_f]
        # N-frame
        if len(snapshot) < self.seq_length:
            return None
        now = time.time()
        window = [(t, f) for (t, f) in snapshot if now - t <= self.seq_span_sec]
        if len(window) < self.seq_length:
            window = snapshot[-self.seq_length:]
        n = len(window)
        if n == self.seq_length:
            picks = window
        else:
            last = n - 1
            idx = [round(i * last / (self.seq_length - 1)) for i in range(self.seq_length)]
            seen = set(); out = []
            for k in idx:
                if k not in seen:
                    seen.add(k); out.append(window[k])
            while len(out) < self.seq_length and out:
                out.append(window[-1])
            picks = out[:self.seq_length]
        return [f for (_t, f) in picks]

    @staticmethod
    def _jpeg_b64(frame):
        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            return None
        return base64.b64encode(enc.tobytes()).decode("ascii")

    def _build_goal_augmented_prompt(self):
        """Return ``PROMPT_SEQUENCE_IMAGES`` + a goal block + a
        goal-reached-reporting block. Social-nav rules stay primary;
        the goal is a task bias appended at the end. Includes explicit
        fisheye-distance-correction guidance so the model stops before
        colliding with the goal object."""
        base = _BASE_SEQ_PROMPT or ""
        goal = self.goal_description
        block = (
            "\n\nPRIMARY NAVIGATION GOAL (task context, secondary to the "
            "social-safety rules above):\n"
            f"- TASK: move to the {goal}, and the action will be STOP.\n"
            f"- The robot is trying to reach: {goal}\n"
            "- Prefer actions that make progress toward this goal, but do "
            "NOT override any social-safety rule above. If reaching the "
            "goal would require an unsafe action, choose STOP, REVIEW, or "
            "a lateral bypass exactly as the rules dictate.\n"
            "- When multiple safe actions are available, pick the one that "
            "best advances toward the goal.\n"
            "- Continue to classify person motion (approaching / receding / "
            "crossing / stationary / none) and apply the CROSSING DIRECTION "
            "RULE and RECEDING CASE RULE exactly as specified above — the "
            "goal must never cause you to ignore a person.\n"
            "\nFISHEYE CAMERA DISTORTION — CRITICAL:\n"
            "- The robot's front camera is a FISHEYE lens. Objects at the "
            "center of the image are compressed and appear SMALLER and "
            "FARTHER AWAY than they actually are. An object that looks "
            "'a few meters ahead' in the image may in reality be less "
            "than 1 meter from the robot.\n"
            "- Peripheral distortion also stretches objects near the "
            "edges; do not mistake edge-stretch for a far distance.\n"
            "- Because of this distortion, you MUST call goal_reached=true "
            "EARLIER than ordinary perspective would suggest. When in "
            "doubt about whether the goal is close enough, ASSUME IT IS "
            "CLOSER than it appears and STOP.\n"
            "- It is far better to stop slightly early (the robot can "
            "inch forward later) than to drive into the goal object. "
            "Running into the goal is a failure.\n"
            "\nLEFT / RIGHT ACTIONS — EXPANDED MEANING:\n"
            "- LEFT and RIGHT are NOT only for avoiding people. They are "
            "also used to:\n"
            "    (a) avoid STATIC obstacles (walls, furniture, boxes, "
            "cones, poles, chairs, equipment, doorways, anything the "
            "robot cannot drive over) in the forward corridor;\n"
            "    (b) STEER TOWARD THE GOAL when the goal is visible but "
            "not centered in the frame — turn LEFT if the goal is on the "
            "robot's left side, turn RIGHT if the goal is on the robot's "
            "right side, so that the next step the goal becomes more "
            "centered.\n"
            "- If the forward corridor is blocked by a static obstacle "
            "and one lateral side is clearly freer, answer LEFT or RIGHT "
            "toward the freer side. Only fall back to STOP if both "
            "lateral sides are also blocked.\n"
            "- If no person or static obstacle is blocking, and the goal "
            "is visible but OFF-CENTER, prefer LEFT or RIGHT toward the "
            "goal over plain FORWARD. If the goal is already centered, "
            "prefer FORWARD.\n"
            "- If the goal is not visible in the latest frame, and no "
            "obstacle is blocking, answer FORWARD to keep exploring.\n"
            "\nGOAL-REACHED REPORTING — add EXTRA fields to the JSON output:\n"
            "  \"goal_visible\": true|false  (the goal object is clearly "
            "visible in the latest frame)\n"
            "  \"goal_reached\": true|false  — set TRUE (with the fisheye "
            "correction above) when ANY of these hold:\n"
            "     (i) the goal object's bounding box covers roughly >= 15% "
            "of the image height OR >= 12% of the image width (remember: "
            "fisheye compresses the center, so these thresholds "
            "correspond to the goal being much closer than the raw image "
            "suggests),\n"
            "     (ii) the goal is in front of the robot within roughly "
            "1.0–1.5 meters of APPARENT distance (real distance will be "
            "smaller due to fisheye),\n"
            "     (iii) the goal occupies the lower half or lower third "
            "of the image (near-ground objects directly in front),\n"
            "     (iv) the robot is essentially touching or next to the "
            "goal.\n"
            "   ERR ON THE SIDE OF SETTING goal_reached=TRUE. Given "
            "fisheye under-estimation of proximity, a goal that LOOKS "
            "'somewhat close and roughly centered' is almost certainly "
            "close enough to stop. When goal_reached=true you MUST set "
            "recommended_action=STOP.\n"
            "  \"goal_bbox_height_frac\": float in [0,1] — estimated "
            "fraction of image height the goal's bounding box covers "
            "(0 if not visible). Useful for calibrating goal_reached.\n"
            "  \"goal_position\": one of [center/left/right/none] — where "
            "the goal sits horizontally in the latest frame.\n"
            "  \"reasoning\": string — a short (1-3 sentence) explanation "
            "of WHY you chose the recommended_action. Reference the "
            "observed person motion, static obstacles, the goal's "
            "position and apparent size in the frame, and/or the "
            "estimated distance to the goal (acknowledging fisheye "
            "compression). This field is mandatory on every response.\n"
        )
        return base + block

    def _build_social_nav_prompt(self):
        """Conservative social-nav prompt. The VLM is a HIGH-LEVEL
        DECISION-MAKER, not a low-level controller.

        Action space for ``recommended_action`` is restricted to
        {STOP, FORWARD, REVIEW}. LEFT / RIGHT become *advisory hints*
        emitted only in ``recommended_avoidance_side``; the controller
        decides whether to execute them based on its own safety gate.

        The VLM sees only the front-facing fisheye camera — it has no
        depth, no odometry, no scan, no geometric free-space check.
        Therefore it MUST NOT commit the robot to a lateral motion.
        """
        base = _BASE_SEQ_PROMPT or ""
        block = (
            "\n\nOVERRIDE — ACTION-SPACE RESTRICTION:\n"
            "- IGNORE any earlier instruction that told you to output "
            "LEFT or RIGHT in the `recommended_action` field.\n"
            "- You MUST now output `recommended_action` as exactly ONE "
            "of: STOP, FORWARD, REVIEW.\n"
            "- LEFT and RIGHT are NO LONGER valid values for "
            "`recommended_action`. They survive ONLY as advisory hints "
            "in the `recommended_avoidance_side` field (values: "
            "left, right, none). The controller may or may not act on "
            "that hint after its own geometric safety check.\n"
            "- You have only a front-facing fisheye camera. You "
            "CANNOT verify that a lateral side is geometrically free "
            "from obstacles. Do NOT commit the robot to a lateral "
            "motion — that is the controller's job, not yours.\n"
            "\nFISHEYE CAMERA DISTORTION — FOR PERSON DETECTION:\n"
            "- The robot's front camera is a FISHEYE lens. Objects "
            "near the image CENTER appear SMALLER and FARTHER AWAY "
            "than they actually are. A person who looks 'a few meters "
            "ahead' may actually be less than 1 meter from the robot.\n"
            "- When detecting a person, ASSUME they are CLOSER than "
            "they appear in the raw image.\n"
            "- Apparent-size → distance calibration:\n"
            "     bbox >= 40% of image height -> VERY CLOSE (<= 1 m)\n"
            "     20%–40% of image height     -> CLOSE      (<= 2 m)\n"
            "     8%–20% of image height      -> MEDIUM     (2–4 m)\n"
            "     < 8%  of image height       -> FAR        (> 4 m)\n"
            "- Peripheral stretching is fisheye distortion, not "
            "distance. Do not treat edge-stretched people as far.\n"
            "\nDECISION POLICY (STOP / FORWARD / REVIEW only):\n"
            "- FORWARD — answer FORWARD only when the forward corridor "
            "is UNAMBIGUOUSLY clear AND one of these holds:\n"
            "    (a) no person is visible in any frame, OR\n"
            "    (b) the person is clearly off to one side (left "
            "third or right third) of the image AND their motion is "
            "not into the corridor, OR\n"
            "    (c) the person is FAR (< 8% of image height), OR\n"
            "    (d) the person is receding and the latest frame "
            "shows a clear corridor.\n"
            "- STOP — answer STOP when the forward corridor is blocked "
            "AND lateral safety cannot be confirmed from the image. "
            "This covers the common case of a person stationary in "
            "the center of the frame — STOP is the correct default; "
            "let the operator / higher planner decide whether a "
            "lateral bypass is safe. Specifically, prefer STOP over "
            "any suggestion of lateral motion whenever:\n"
            "    (i) the person is CLOSE or VERY CLOSE (>= 20% bbox), "
            "centered, and stationary,\n"
            "    (ii) the person is approaching,\n"
            "    (iii) the person just blocked the corridor in the "
            "latest frame after being absent in earlier frames "
            "('entering_late').\n"
            "- REVIEW — answer REVIEW when the scene is AMBIGUOUS and "
            "you cannot responsibly commit to STOP or FORWARD. REVIEW "
            "is REQUIRED (not optional) in ALL of the following:\n"
            "    (a) a person appears only in the latest frame and "
            "their intent cannot be inferred,\n"
            "    (b) the person's motion direction is unreadable "
            "(approaching vs receding vs crossing unclear),\n"
            "    (c) a crossing person's direction (leftward vs "
            "rightward) is ambiguous,\n"
            "    (d) a side-standing person may or may not step into "
            "the corridor,\n"
            "    (e) heavy occlusion prevents judging blockage,\n"
            "    (f) the person is close but their trajectory is "
            "unclear.\n"
            "  Do NOT guess in any of these situations. Output REVIEW "
            "and let the operator decide.\n"
            "\nADVISORY LATERAL HINT (NOT an executable action):\n"
            "- Fill `recommended_avoidance_side` only when you would, "
            "hypothetically and given depth information you do not "
            "have, prefer one side over the other. Valid values: "
            "`left`, `right`, `none`.\n"
            "- This field is a HINT for the controller / operator. "
            "It does NOT make the robot strafe. Setting it does NOT "
            "authorize lateral motion.\n"
            "- Set it to `none` whenever you are uncertain — do NOT "
            "fill it with a guess just to avoid `none`.\n"
            "\nOUTPUT — add the following EXTRA JSON fields (in "
            "addition to the original schema above):\n"
            "  \"person_bbox_height_frac\": float in [0,1] — estimated "
            "fraction of image height the closest person's bounding "
            "box covers (0 if no person visible).\n"
            "  \"person_position\": one of [center/left/right/none].\n"
            "  \"lateral_safety_verified\": false — you do NOT have "
            "enough information to verify lateral safety. Always "
            "output this as false.\n"
            "  \"reasoning\": string — MANDATORY 1-3 sentences "
            "explaining WHY you chose the recommended_action, "
            "referencing person apparent-size, position, motion, "
            "and/or estimated distance (with fisheye compression "
            "acknowledged). Never leave empty.\n"
            "\nFinal reminder: `recommended_action` MUST be exactly "
            "one of {STOP, FORWARD, REVIEW}. If you catch yourself "
            "about to write LEFT or RIGHT there, STOP and re-evaluate "
            "— put the side in `recommended_avoidance_side` and set "
            "`recommended_action` to STOP or REVIEW.\n"
        )
        return base + block

    def _build_prompt(self):
        """Pick the prompt based on whether a goal was configured."""
        if self.goal_description:
            return self._build_goal_augmented_prompt()
        return self._build_social_nav_prompt()

    @staticmethod
    def _extract_first_json(text):
        s = (text or "").find("{")
        if s == -1:
            return None
        depth = 0; in_str = False; esc = False
        for i in range(s, len(text)):
            c = text[i]
            if in_str:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': in_str = False
                continue
            if c == '"': in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[s:i+1]
        return None

    def call_backend_direct(self, frames, prompt_text):
        """OpenAI-chat call to the vLLM backend with a custom prompt +
        images. Same return shape as call_wrapper."""
        imgs = [self._jpeg_b64(f) for f in frames]
        if any(b is None for b in imgs):
            return None, {}, {"ok": False, "error": "jpeg_encode_failed"}, 0.0
        content = [{"type": "text", "text": prompt_text}]
        for b in imgs:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b}"},
            })
        payload = {
            "model": self.backend_model,
            "temperature": 0,
            "messages": [{"role": "user", "content": content}],
        }
        t0 = time.time()
        try:
            r = requests.post(
                self.backend_url, json=payload,
                timeout=(self.connect_timeout, self.request_timeout),
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as exc:
            return None, {}, {"ok": False, "error": f"request_failed: {exc}"}, time.time() - t0
        except ValueError as exc:
            return None, {}, {"ok": False, "error": f"json_decode: {exc}"}, time.time() - t0
        latency = time.time() - t0
        try:
            raw = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None, {}, {"ok": False, "error": "missing_choices_content",
                              "raw_text": str(data)[:400]}, latency
        candidate = self._extract_first_json(raw)
        rj = {}
        if candidate:
            try:
                rj = json.loads(candidate)
            except json.JSONDecodeError:
                rj = {}
        envelope = {"ok": True, "raw_text": raw, "response_json": rj}
        return normalize_action(rj.get("recommended_action")), rj, envelope, latency

    def call_wrapper(self, frames):
        # Always route through backend-direct when the canonical prompt
        # is importable so we can enforce reasoning + fisheye + bias
        # rules. Falls back to the wrapper endpoint only if the prompt
        # module is unavailable.
        if _BASE_SEQ_PROMPT is not None:
            return self.call_backend_direct(frames, self._build_prompt())
        images_b64 = []
        for f in frames:
            b = self._jpeg_b64(f)
            if b is None:
                return None, {}, {"ok": False, "error": "jpeg_encode_failed"}, 0.0
            images_b64.append(b)
        payload = {"prompt_name": self.prompt_name, "images_base64": images_b64}
        t0 = time.time()
        try:
            resp = requests.post(
                self.wrapper_url, json=payload,
                timeout=(self.connect_timeout, self.request_timeout),
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            return None, {}, {"ok": False, "error": f"request_failed: {exc}"}, time.time()-t0
        except ValueError as exc:
            return None, {}, {"ok": False, "error": f"json_decode_failed: {exc}"}, time.time()-t0
        latency = time.time() - t0
        if not data.get("ok"):
            return None, data.get("response_json") or {}, data, latency
        rj = data.get("response_json") or {}
        action = normalize_action(rj.get("recommended_action"))
        return action, rj, data, latency

    def _cache_vlm(self, action, rj, latency):
        if not rj:
            return
        self._last_vlm = {
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

    def log_vlm(self, state_from, num_images, action_norm, action_raw,
                rj, env, latency, next_state, actuation):
        now = time.time()
        rj = rj or {}
        env = env or {}
        try:
            self.vlm_jsonl.write(json.dumps({
                "timestamp": now,
                "datetime": datetime.utcfromtimestamp(now).isoformat()+"Z",
                "fsm_state_entered_from": state_from,
                "num_images": num_images,
                "wrapper_url": self.wrapper_url,
                "backend_url": self.backend_url if self.goal_description else "",
                "prompt_name": self.prompt_name,
                "prompt_path": "backend_direct" if self.goal_description else "wrapper",
                "goal": self.goal_description,
                "action_normalized": action_norm,
                "action_raw": action_raw,
                "response_json": rj,
                "raw_text_snippet": str(env.get("raw_text", ""))[:400],
                "wrapper_ok": bool(env.get("ok", False)),
                "wrapper_error": env.get("error", ""),
                "latency_sec": latency,
                "fsm_next_state": next_state,
                "robot_actuation": actuation,
            }, sort_keys=True) + "\n")
            self.vlm_jsonl.flush()
        except Exception:
            pass
        try:
            self.vlm_csv.writerow([
                f"{now:.3f}", state_from, num_images,
                self.wrapper_url, self.prompt_name,
                self.goal_description,
                action_norm or "", action_raw or "",
                rj.get("motion", ""), rj.get("crossing_direction", ""),
                rj.get("safer_lateral_side", ""),
                rj.get("recommended_avoidance_side", ""),
                rj.get("risk_level", ""),
                rj.get("path_blocked_latest_frame", ""),
                (rj.get("uncertainty_reason", "") or "").replace("\n", " ")[:200],
                rj.get("person_bbox_height_frac", ""), rj.get("person_position", ""),
                rj.get("goal_visible", ""), rj.get("goal_reached", ""),
                rj.get("goal_bbox_height_frac", ""), rj.get("goal_position", ""),
                (rj.get("reasoning", "") or "").replace("\n", " ")[:400],
                bool(env.get("ok", False)),
                str(env.get("error", ""))[:200],
                next_state, actuation, f"{latency:.3f}",
            ])
            self.vlm_csv_file.flush()
        except Exception:
            pass

    # --------------------------------------------------------- primitives
    def primitive_duration(self, action):
        if action == "FORWARD":
            return self.forward_burst_sec
        if action == "BACKWARD":
            return self.backward_burst_sec
        if action in ("LEFT", "RIGHT"):
            if self.lateral_mode == "strafe":
                return self.strafe_sec
            if self.lateral_mode == "blend":
                return self.strafe_sec
            if self.lateral_mode == "turn_only":
                return self.yaw_burst_sec
            return self.yaw_burst_sec + self.creep_after_turn_sec
        return 0.0

    def begin_primitive(self, action):
        now = time.time()
        self.fsm_exec_primitive = action
        self.fsm_exec_started = now
        self.fsm_exec_until = now + self.primitive_duration(action)

    def tick_primitive(self):
        now = time.time()
        if self.fsm_exec_primitive is None or now >= self.fsm_exec_until:
            return None
        action = self.fsm_exec_primitive
        if action == "FORWARD":
            return (self.forward_speed, 0.0, 0.0)
        if action == "BACKWARD":
            return (-self.backward_speed, 0.0, 0.0)
        if action in ("LEFT", "RIGHT"):
            sign = 1.0 if action == "LEFT" else -1.0
            if self.lateral_mode == "strafe":
                return (0.0, sign * self.strafe_speed, 0.0)
            if self.lateral_mode == "blend":
                return (self.blend_forward_speed, sign * self.strafe_speed, 0.0)
            if self.lateral_mode == "turn_only":
                return (0.0, 0.0, sign * self.yaw_burst_speed)
            # turn_creep
            elapsed = now - self.fsm_exec_started
            if elapsed < self.yaw_burst_sec:
                return (0.0, 0.0, sign * self.yaw_burst_speed)
            return (self.creep_speed, 0.0, 0.0)
        return None

    def describe_actuation(self, action):
        if action == "FORWARD":
            return f"vx={self.forward_speed:.2f} for {self.forward_burst_sec:.2f}s"
        if action == "BACKWARD":
            return f"vx=-{self.backward_speed:.2f} for {self.backward_burst_sec:.2f}s"
        if action in ("LEFT", "RIGHT"):
            s = "+" if action == "LEFT" else "-"
            if self.lateral_mode == "strafe":
                return f"vy={s}{self.strafe_speed:.2f} for {self.strafe_sec:.2f}s (pure strafe)"
            if self.lateral_mode == "blend":
                return f"vx={self.blend_forward_speed:.2f} vy={s}{self.strafe_speed:.2f} for {self.strafe_sec:.2f}s (blend)"
            if self.lateral_mode == "turn_only":
                return f"wz={s}{self.yaw_burst_speed:.2f} for {self.yaw_burst_sec:.2f}s (turn in place)"
            return (f"wz={s}{self.yaw_burst_speed:.2f} for {self.yaw_burst_sec:.2f}s then creep "
                    f"vx={self.creep_speed:.2f} for {self.creep_after_turn_sec:.2f}s (turn_creep)")
        return "STOP"

    # --------------------------------------------------------- FSM
    def fsm_enter(self, new_state):
        if self.fsm_state != new_state:
            rospy.loginfo("[vlm_only] %s -> %s (t=%.2f)", self.fsm_state, new_state,
                          time.time() - self.fsm_state_entered)
        self.fsm_state = new_state
        self.fsm_state_entered = time.time()

    def step(self):
        """Single-tick FSM. Returns (vx, vy, wz), action_label, source, reason."""
        state = self.fsm_state
        now = time.time()

        if state == "HOLD_STOP":
            held = now - self.fsm_state_entered
            if held < self.stop_hold_sec:
                return (0.0, 0.0, 0.0), "STOP", "VLM", \
                    f"HOLD_STOP {held:.2f}/{self.stop_hold_sec:.2f}s (blocking; lateral not verified)"
            self.fsm_enter("CRUISE")
            return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", "HOLD_STOP expired -> CRUISE"

        if state == "CRUISE":
            since_decide = now - self.fsm_last_decision_time
            if since_decide >= self.cruise_decide_interval_sec:
                self.fsm_enter("DECIDE")
                return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", \
                    f"periodic DECIDE (since_last={since_decide:.2f}s)"
            return (self.forward_speed, 0.0, 0.0), "FORWARD", "VLM", \
                f"CRUISE forward (next DECIDE in {self.cruise_decide_interval_sec-since_decide:.2f}s)"

        if state == "DECIDE":
            if (now - self.fsm_last_decision_time) < self.decide_cooldown_sec:
                return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", \
                    f"cooldown {now-self.fsm_last_decision_time:.2f}/{self.decide_cooldown_sec:.2f}s"
            frames = self.pick_sequence()
            if frames is None:
                return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", "waiting for frames"
            action, rj, env, latency = self.call_wrapper(frames)
            self.fsm_last_decision_time = time.time()
            raw = (rj or {}).get("recommended_action", "") if rj else ""
            self._cache_vlm(action, rj, latency)

            # Goal-reached short-circuit: if the VLM reports the robot has
            # arrived at the goal, stop all motion and mark the controller
            # for shutdown. Social-safety rules still take precedence —
            # only fire this when goal_description is set AND the model
            # actually set goal_reached=true in its JSON output.
            goal_reached = False
            if self.goal_description and rj:
                gr = rj.get("goal_reached")
                if isinstance(gr, bool):
                    goal_reached = gr
                elif isinstance(gr, str):
                    goal_reached = gr.strip().lower() in ("true", "yes", "1")
            if goal_reached:
                self._goal_reached = True
                self.log_vlm("DECIDE", len(frames), "STOP", raw, rj, env, latency,
                             "GOAL_REACHED", "STOP (goal_reached)")
                self.fsm_enter("CRUISE")  # terminal; run loop will break
                return (0.0, 0.0, 0.0), "STOP", "VLM", \
                    f"GOAL REACHED: {self.goal_description!r}"

            if action is None:
                rospy.logwarn("[vlm_only] decide failed; holding")
                self.log_vlm("DECIDE", len(frames), None, raw, rj, env, latency, "HOLD_REVIEW", "STOP")
                self.fsm_enter("HOLD_REVIEW")
                return (0.0, 0.0, 0.0), "STOP", "VLM", \
                    f"wrapper_failed err={str((env or {}).get('error',''))[:80]}"
            rospy.loginfo(
                "[vlm_only] decision=%s motion=%s cross_dir=%s avoid=%s goal_vis=%s goal_reached=%s reason=%s unc=%s",
                action, rj.get("motion"), rj.get("crossing_direction"),
                rj.get("recommended_avoidance_side"),
                rj.get("goal_visible"), rj.get("goal_reached"),
                (rj.get("reasoning") or "")[:160],
                (rj.get("uncertainty_reason") or "")[:80],
            )
            # LATERAL SAFETY GATE — the VLM has no geometric / depth
            # verification, so treat any LEFT/RIGHT as an advisory side
            # hint and force the executed action back to STOP unless the
            # operator has explicitly opted in via ~fsm_allow_vlm_lateral.
            suggested_side = rj.get("recommended_avoidance_side") if rj else None
            projected_action, projected_rj, projection_note = project_realtime_action(
                action,
                rj,
                allow_lateral=self.allow_vlm_lateral,
            )
            if projected_action != action:
                rospy.logwarn(
                    "[vlm_only] VLM suggested %s (side=%s) — NOT executing: "
                    "%s",
                    action, suggested_side,
                    projection_note or "downgrading to STOP",
                )
                rj = projected_rj
                action = projected_action
                self._cache_vlm(action, rj, latency)

            self.fsm_last_action = action
            # Track "stuck" counter for optional BACKWARD recovery.
            if action in ("STOP", "REVIEW"):
                self._fsm_stuck_count += 1
            elif action == "FORWARD":
                self._fsm_stuck_count = 0
            # streak counter for LEFT/RIGHT cap
            if action in ("LEFT", "RIGHT"):
                self._fsm_lateral_streak += 1
            elif action == "FORWARD":
                self._fsm_lateral_streak = 0

            if action == "REVIEW":
                # Optional implicit BACKWARD recovery when stuck.
                if (self.back_enabled
                        and self._fsm_stuck_count >= self.back_after_n_stuck):
                    rospy.logwarn(
                        "[vlm_only] stuck (%d REVIEW/STOP); emitting BACKWARD recovery",
                        self._fsm_stuck_count,
                    )
                    self._fsm_stuck_count = 0
                    self.begin_primitive("BACKWARD")
                    self.fsm_enter("EXECUTE")
                    act = self.describe_actuation("BACKWARD")
                    self.log_vlm("DECIDE", len(frames), "BACKWARD", raw, rj, env, latency, "EXECUTE", act)
                    return (0.0, 0.0, 0.0), "BACKWARD", "VLM", "stuck-recovery BACKWARD"
                self.log_vlm("DECIDE", len(frames), action, raw, rj, env, latency, "HOLD_REVIEW", "STOP")
                self.fsm_enter("HOLD_REVIEW")
                return (0.0, 0.0, 0.0), "REVIEW", "VLM", \
                    f"REVIEW motion={rj.get('motion')} unc={(rj.get('uncertainty_reason') or '')[:40]}"

            if action == "STOP":
                self.log_vlm("DECIDE", len(frames), action, raw, rj, env, latency, "HOLD_STOP", "STOP")
                self._fsm_last_stop_time = time.time()
                self.fsm_enter("HOLD_STOP")
                # Push next DECIDE to happen sooner once the hold ends.
                self.fsm_last_decision_time = time.time() - max(
                    0.0, self.cruise_decide_interval_sec - 1.0)
                return (0.0, 0.0, 0.0), "STOP", "VLM", \
                    f"STOP motion={rj.get('motion')} blocked={rj.get('path_blocked_latest_frame')} side_hint={suggested_side}"

            # FORWARD / LEFT / RIGHT / BACKWARD → primitive
            self._fsm_last_stop_time = 0.0
            self.begin_primitive(action)
            self.fsm_enter("EXECUTE")
            act = self.describe_actuation(action)
            self.log_vlm("DECIDE", len(frames), action, raw, rj, env, latency, "EXECUTE", act)
            return (0.0, 0.0, 0.0), action, "VLM", \
                f"{action} motion={rj.get('motion')} cross_dir={rj.get('crossing_direction')} avoid={rj.get('recommended_avoidance_side')}"

        if state == "EXECUTE":
            vel = self.tick_primitive()
            if vel is None:
                finished = self.fsm_exec_primitive
                self.fsm_exec_primitive = None
                # Post-lateral immediate DECIDE so the VLM checks whether
                # the newly-visible view contains an empty forward path.
                if (finished in ("LEFT", "RIGHT")
                        and self.force_decide_after_lateral):
                    if self._fsm_lateral_streak >= self.lateral_streak_max:
                        rospy.logwarn(
                            "[vlm_only] lateral streak cap %d reached -> HOLD_REVIEW",
                            self.lateral_streak_max,
                        )
                        self._fsm_lateral_streak = 0
                        self.fsm_enter("HOLD_REVIEW")
                        return (0.0, 0.0, 0.0), "REVIEW", "VLM_WAIT", \
                            f"lateral streak cap {self.lateral_streak_max}"
                    self.fsm_last_decision_time = 0.0
                    self.fsm_enter("DECIDE")
                    return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", \
                        f"post-lateral DECIDE streak={self._fsm_lateral_streak}/{self.lateral_streak_max}"
                self._fsm_lateral_streak = 0
                # Optional fast-response path: after a FORWARD/BACKWARD burst
                # go straight to DECIDE instead of CRUISE so the VLM is
                # re-evaluated every burst (~every forward_burst_sec).
                if (finished in ("FORWARD", "BACKWARD")
                        and self.force_decide_after_forward):
                    self.fsm_last_decision_time = 0.0
                    self.fsm_enter("DECIDE")
                    return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", \
                        f"post-{finished} DECIDE (fast response)"
                self.fsm_enter("CRUISE")
                return (0.0, 0.0, 0.0), "STOP", "VLM", "execute primitive finished"
            return vel, self.fsm_last_action, "VLM", \
                f"executing {self.fsm_last_action} (motion={self._last_vlm.get('motion')})"

        if state == "HOLD_REVIEW":
            held = now - self.fsm_state_entered
            if held < self.review_hold_sec:
                return (0.0, 0.0, 0.0), "REVIEW", "VLM_WAIT", \
                    f"HOLD_REVIEW {held:.2f}/{self.review_hold_sec:.2f}s"
            if self.fsm_review_count < self.review_retries:
                self.fsm_review_count += 1
                self.fsm_enter("DECIDE")
                return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", \
                    f"retry {self.fsm_review_count}/{self.review_retries}"
            # Retries exhausted. Optional implicit BACKWARD.
            if self.back_enabled:
                rospy.logwarn("[vlm_only] HOLD_REVIEW retries exhausted; BACKWARD recovery")
                self.fsm_review_count = 0
                self.begin_primitive("BACKWARD")
                self.fsm_enter("EXECUTE")
                return (-self.backward_speed, 0.0, 0.0), "BACKWARD", "VLM", \
                    "stuck-after-review BACKWARD recovery"
            # Auto-recover: instead of freezing forever, reset the
            # retry counter and drop back to CRUISE so the next periodic
            # DECIDE can re-evaluate the scene.
            rospy.logwarn("[vlm_only] REVIEW retries exhausted; auto-recovering to CRUISE")
            self.fsm_review_count = 0
            self.fsm_last_decision_time = 0.0
            self.fsm_enter("CRUISE")
            return (0.0, 0.0, 0.0), "STOP", "VLM_WAIT", "HOLD_REVIEW auto-recover -> CRUISE"

        rospy.logerr("[vlm_only] unknown state %s; forcing CRUISE", state)
        self.fsm_enter("CRUISE")
        return (0.0, 0.0, 0.0), "STOP", "VLM", "unknown state reset"

    # --------------------------------------------------------- run loop
    def run(self):
        rate = rospy.Rate(10)
        self.fsm_enter("CRUISE")
        while not rospy.is_shutdown():
            tstart = time.time()
            (vx, vy, wz), action, source, reason = self.step()

            if self._goal_reached:
                self.send_stop()
                print(
                    f"\033[1;32m[vlm_only] GOAL REACHED: "
                    f"{self.goal_description!r}. stopping all actions "
                    f"and exiting.\033[0m",
                    flush=True,
                )
                break

            # Edge-triggered UDP transport.
            want_stop = (action in ("STOP", "REVIEW")
                         or (vx == 0.0 and vy == 0.0 and wz == 0.0))
            if want_stop:
                if self._last_udp_kind != "stop":
                    self.send_stop()
                    self._last_udp_kind = "stop"
            else:
                cur = (round(vx, 3), round(vy, 3), round(wz, 3))
                if (self._last_udp_kind != "move"
                        or self._last_udp_vec != cur
                        or (time.time() - self._last_udp_time) > 0.3):
                    self.send_velocity(vx, vy, wz)
                    self._last_udp_kind = "move"
                    self._last_udp_vec = cur
                    self._last_udp_time = time.time()

            # Debug message
            debug = {
                "t": round(time.time(), 3),
                "state": self.fsm_state,
                "source": source,
                "action": action,
                "cmd": {"vx": round(vx, 3), "vy": round(vy, 3), "wz": round(wz, 3)},
                "reason": reason,
                "last_vlm": self._last_vlm,
                "lateral_streak": self._fsm_lateral_streak,
                "stuck_count": self._fsm_stuck_count,
            }
            try:
                self.fsm_debug_pub.publish(String(data=json.dumps(debug, sort_keys=True)))
            except Exception:
                pass

            # Color-coded stdout (throttled)
            ansi = {"VLM_MOVE":"\033[36m","VLM_STOP":"\033[35m",
                    "VLM_WAIT":"\033[90m","RESET":"\033[0m"}
            if source == "VLM":
                color = ansi["VLM_MOVE"] if action in ("FORWARD","LEFT","RIGHT","BACKWARD") else ansi["VLM_STOP"]
            elif source == "VLM_WAIT":
                color = ansi["VLM_WAIT"]
            else:
                color = ""
            now = time.time()
            sig = (self.fsm_state, source, action)
            if not hasattr(self, "_last_sig") or self._last_sig != sig or (now - getattr(self, "_last_print_time", 0.0)) > 0.5:
                self._last_sig = sig
                self._last_print_time = now
                tail = ""
                if source in ("VLM", "VLM_WAIT"):
                    v = self._last_vlm
                    tail = (f" | VLM[motion={v.get('motion')} cross_dir={v.get('crossing_direction')} "
                            f"avoid={v.get('recommended_avoidance_side')} blocked={v.get('path_blocked_latest_frame')} "
                            f"risk={v.get('risk_level')} unc={(v.get('uncertainty_reason') or '')[:40]!r}]")
                print(f"{color}[{self.fsm_state:>11s}] source={source:<11s} "
                      f"action={action:<9s} cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f}) | {reason}{tail}{ansi['RESET']}",
                      flush=True)

            # Loop profiling
            if self.profile_loop:
                self._profile_samples.append(time.time() - tstart)
                if (time.time() - self._profile_last) >= self.profile_interval:
                    s = sorted(self._profile_samples)
                    if s:
                        n = len(s); avg=sum(s)/n; mx=s[-1]; p95=s[int(0.95*(n-1))]
                        print(f"\033[96m[profile] n={n} avg={avg*1000:.1f}ms p95={p95*1000:.1f}ms max={mx*1000:.1f}ms\033[0m",
                              flush=True)
                    self._profile_samples = []
                    self._profile_last = time.time()

            rate.sleep()


def main():
    c = VlmOnlyController()
    c.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        sys.exit(0)
