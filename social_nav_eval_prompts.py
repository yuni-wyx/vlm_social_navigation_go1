#!/usr/bin/env python3
"""
Shared prompt templates and action constants for offline social navigation eval.

Action space:
- STOP     : halt / yield; forward motion unsafe and no clearly safer lateral path
- FORWARD  : continue ahead; forward corridor clear
- LEFT     : bypass obstacle by stepping to the robot's left
- RIGHT    : bypass obstacle by stepping to the robot's right
- REVIEW   : genuinely uncertain; defer to a safer fallback / human operator

The prompts below are written so that LEFT, RIGHT, and REVIEW are first-class
choices rather than theoretical schema values. Hard decision rules discourage
defaulting to STOP when a lateral bypass is plausible, and explicit uncertainty
triggers encourage REVIEW when the model cannot confidently judge blockage,
intent, or lateral safety.

Sequence-mode prompts also include temporal-reasoning instructions so the model
explicitly compares older vs newer frames to classify motion
(approaching / receding / crossing / entering_late / stationary) and weights
the latest frame more heavily. This is intended to fix the known failure mode
where late-entering people get washed out by empty early frames.
"""

ACTIONS = ("STOP", "FORWARD", "LEFT", "RIGHT", "REVIEW")

PROMPT_SINGLE_IMAGE = """You are assisting a Unitree Go1 quadruped robot.
You see one image from the robot's front-facing camera.

Task: choose the single safest next action for the robot.

Action space (pick exactly one):
- FORWARD : continue straight; the forward corridor is clear
- STOP    : halt; forward is unsafe AND no clearly safer lateral path exists
- LEFT    : bypass by stepping to the robot's left
- RIGHT   : bypass by stepping to the robot's right
- REVIEW  : you are genuinely uncertain; defer to a human / safer fallback

Decision rules (apply in order):
1. If no person or obstacle blocks the robot's forward corridor, answer FORWARD.
2. If forward is blocked but one lateral side is clearly safer (open, no person there),
   answer LEFT or RIGHT. Prefer lateral bypass over STOP whenever it is plausible.
3. If forward is blocked and neither lateral side is clearly safer, answer STOP.
4. If you cannot confidently judge blockage, intent, or lateral safety, answer REVIEW.
   Do NOT guess.

Use REVIEW when any of the following apply:
- The person is only partially visible or just entering the frame.
- Occlusion prevents judging whether the forward path is blocked.
- Crossing intent is unclear (you cannot tell where the person is heading).
- Approach direction is unclear (you cannot tell if the person is moving toward or away).
- Whether the person actually blocks the robot's forward corridor is unclear.

Do NOT default to STOP. STOP is only for cases where forward is unsafe AND no clearly
safer lateral option exists.
Do NOT overuse REVIEW. Prefer FORWARD / LEFT / RIGHT / STOP when the image clearly
supports that choice.

Output ONLY JSON, no other text:
{
  "person_detected": true/false,
  "person_position": "left/center/right/none",
  "path_blocked": true/false,
  "safer_lateral_side": "left/right/none",
  "uncertainty_reason": "<short reason or empty string>",
  "recommended_action": "STOP/FORWARD/LEFT/RIGHT/REVIEW"
}"""

PROMPT_SEQUENCE_IMAGES = """You are assisting a Unitree Go1 quadruped robot.
You see a short time-ordered sequence of frames from the robot's front camera,
oldest first and newest last. Use the temporal change across frames to decide.

Action space:
- FORWARD : the forward corridor is clear or will remain clear
- STOP    : forward is unsafe AND no clearly safer lateral option exists
- LEFT    : bypass the obstacle by stepping to the robot's left
- RIGHT   : bypass the obstacle by stepping to the robot's right
- REVIEW  : genuinely uncertain; defer

Temporal reasoning — classify the person's motion across frames:
- approaching   : person appears larger / closer / more centered over time
                  (person is moving toward the robot; risk rising)
- receding      : person appears smaller / farther over time, distance clearly
                  increases, or the person moves out of the robot's forward
                  corridor. Key indicators:
                    * the person's bounding box/shrinks across frames,
                    * the person moves toward the image edges or out of frame,
                    * the forward corridor becomes clear in the latest frame.
                  Receding is LOWER risk than approaching or crossing.
- crossing_leftward : person is crossing sideways through the forward corridor
                      moving TOWARD THE ROBOT'S LEFT (from right to left).
                      The person's future position will be on the LEFT side.
- crossing_rightward: person is crossing sideways through the forward corridor
                      moving TOWARD THE ROBOT'S RIGHT (from left to right).
                      The person's future position will be on the RIGHT side.
- crossing      : person traverses sideways but direction cannot be confidently
                  determined (use this only when leftward vs rightward is
                  genuinely ambiguous).
- entering_late : early frames are empty, person appears only in the latest frame(s)
                  (weight the latest frame heavily; do not let empty early frames
                  wash out the interaction signal)
- stationary    : person stays in place; evaluate static blockage only
- none          : no person visible in any frame

CROSSING DIRECTION RULE (apply this when motion is crossing_leftward / crossing_rightward / crossing):
If a person is crossing laterally across the robot's forward corridor:
- Determine the person's direction of lateral motion across frames by comparing
  their horizontal position between the earliest and latest frames.
- Avoid moving in the SAME direction the person is moving, because that puts
  the robot on a collision course with the person's future position.
- Prefer the OPPOSITE side to reduce future-collision risk:
    * person moving LEFTWARD  (crossing_leftward)  -> prefer RIGHT
    * person moving RIGHTWARD (crossing_rightward) -> prefer LEFT
- Never pick LEFT/RIGHT arbitrarily in a crossing case. If direction is
  genuinely unclear, answer REVIEW instead.
- If both lateral sides would still intersect the person's path (e.g., the
  person already occupies the whole corridor), answer STOP.

RECEDING CASE RULE (apply this BEFORE the general rules below):
If the person is clearly moving away from the robot and the forward path is
clear in the latest frame, prefer FORWARD over STOP. Do not stop solely
because a person is present. Only choose STOP if the person is still blocking
the path, is likely to re-enter it, or the situation remains uncertain.
When the person's apparent size shrinks across frames, or the person walks
toward the image edges / out of frame, classify motion as "receding" rather
than "stationary", and choose FORWARD when the latest frame shows a clear
corridor.

Decision rules (apply in order; the LATEST frame matters more than older frames):
1. If the situation is genuinely ambiguous (intent, blockage, or motion
   direction cannot be confidently judged), answer REVIEW. Do not guess.
2. If no person or obstacle is in or entering the forward corridor across the
   sequence, answer FORWARD.
3. If the person is receding and the latest frame shows a clear path, answer FORWARD.
   This rule overrides any temptation to STOP based only on earlier frames or
   on the mere presence of the person.
4. If the person is crossing (leftward or rightward), apply the CROSSING
   DIRECTION RULE above:
     - crossing_leftward  -> prefer RIGHT
     - crossing_rightward -> prefer LEFT
   Only fall through to STOP if both lateral sides would still intersect the
   person's future path.
5. If the person is approaching OR is blocking the corridor in the latest frame,
   and one lateral side is clearly safer, answer LEFT or RIGHT. Prefer lateral
   bypass over STOP whenever it is plausible.
6. If the person blocks the corridor in the latest frame and neither lateral side
   is clearly safer, answer STOP.

Use REVIEW when any of the following apply:
- Person is visible in only a single frame and intent cannot be inferred.
- Occlusion prevents judging motion direction.
- Unclear whether a crossing person is inside or outside the forward corridor.
- Ambiguous approach vs recede direction.
- Late-entering person appears only in the very last frame with unclear intent.

Do NOT default to STOP when lateral bypass is plausible.
Do NOT ignore the latest frame: late-entering people must not be washed out by
empty early frames.

Output ONLY JSON, no other text:
{
  "person_detected": true/false,
  "motion": "approaching/receding/crossing_leftward/crossing_rightward/crossing/entering_late/stationary/none",
  "crossing_direction": "leftward/rightward/none",
  "path_blocked_latest_frame": true/false,
  "safer_lateral_side": "left/right/none",
  "recommended_avoidance_side": "left/right/none",
  "risk_level": "low/medium/high",
  "uncertainty_reason": "<short reason or empty string>",
  "recommended_action": "STOP/FORWARD/LEFT/RIGHT/REVIEW"
}"""

PROMPT_STRUCTURED_LOCALIZATION = """Given an image or image sequence from a robot's front camera.

Task:
1. Identify if a person is present.
2. Estimate the person's position relative to the robot.
3. Estimate movement direction.

Output ONLY JSON:
{
  "person_detected": true/false,
  "position": "left/center/right/none",
  "direction": "left/right/toward/away/none"
}"""

PROMPT_CONFIG = {
    "single_image_navigation": {
        "name": "single_image_navigation",
        "prompt": PROMPT_SINGLE_IMAGE,
        "expects_sequence": False,
    },
    "sequence_image_navigation": {
        "name": "sequence_image_navigation",
        "prompt": PROMPT_SEQUENCE_IMAGES,
        "expects_sequence": True,
    },
    "structured_localization": {
        "name": "structured_localization",
        "prompt": PROMPT_STRUCTURED_LOCALIZATION,
        "expects_sequence": True,
    },
}
