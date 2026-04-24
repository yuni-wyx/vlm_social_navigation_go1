#!/usr/bin/env python3
"""Pure-Python helpers shared by social navigation evaluation and control."""

from typing import Any, Dict, Optional, Tuple


def project_realtime_action(
    action: Optional[str],
    response_json: Optional[Dict[str, Any]] = None,
    *,
    allow_lateral: bool = False,
) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Project offline VLM actions into the Go1's real-time executable space.

    LEFT/RIGHT remain advisory by default because the VLM output alone does not
    prove that the corresponding lateral motion is geometrically safe on the
    robot. The advisory side hint is preserved in ``recommended_avoidance_side``
    so logs and higher-level recovery logic can still use it.
    """
    projected_action = action
    projected_response = dict(response_json or {})
    note = None

    if action in ("LEFT", "RIGHT") and not allow_lateral:
        projected_action = "STOP"
        if not projected_response.get("recommended_avoidance_side"):
            projected_response["recommended_avoidance_side"] = action.lower()
        note = (
            "LEFT/RIGHT are advisory only until a geometric safety projection "
            "or free-space verifier approves them."
        )

    return projected_action, projected_response, note
