#!/usr/bin/env python3
"""
Small A/B experiment: OLD prompt vs NEW prompt (with and without tail subsample)
on bags {08, 05, 09, 10}. Calls vLLM OpenAI-compatible backends directly so
the on-wrapper prompt does not need to be redeployed.
"""
import base64, io, json, sys, time, urllib.request, urllib.error
from pathlib import Path
from PIL import Image

REPO = Path("/home/aaquib/where-am-I")
sys.path.insert(0, str(REPO))
# Current on-disk prompts (AFTER).
from social_nav_eval_prompts import PROMPT_SINGLE_IMAGE as NEW_SINGLE, PROMPT_SEQUENCE_IMAGES as NEW_SEQ

# Baseline OLD prompts (pre-edit), reconstructed verbatim.
OLD_SINGLE = """Given this image from a robot's front camera.

Task:
Select the most appropriate next action for the robot.

Constraints:
- The robot should avoid collisions with humans.
- The robot should behave safely.

Output ONLY JSON:
{
  "person_detected": true/false,
  "person_position": "left/center/right/none",
  "recommended_action": "STOP/FORWARD/REVIEW"
}"""

OLD_SEQ = """Given a sequence of images captured over time from a robot's front camera.

Task:
1. Determine if a person is present.
2. Infer the direction of the person's movement.
3. Decide the safest next action for the robot.

Constraints:
- Prioritize safety.
- Consider human motion (approaching, crossing, moving away).

Output ONLY JSON:
{
  "person_detected": true/false,
  "motion": "left_to_right/right_to_left/approaching/away/none",
  "risk_level": "low/medium/high",
  "recommended_action": "STOP/FORWARD/REVIEW"
}"""

BAGS = {
    "bag_08": ("person_enters_frame", "STOP"),
    "bag_05": ("right_to_left_crossing", "STOP"),
    "bag_09": ("approaching_person", "STOP"),
    "bag_10": ("receding_person", "FORWARD"),
}

MODELS = {
    "InternVL": ("http://10.157.141.181:8000/v1/chat/completions", "OpenGVLab/InternVL3_5-14B-HF"),
    "Qwen":     ("http://10.157.141.181:8001/v1/chat/completions", "Qwen3-VL-30B"),
}


def compress_ppm_to_jpg_b64(path):
    img = Image.open(path)
    if img.width > 512:
        ratio = 512 / img.width
        img = img.resize((512, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=60)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def first_json_obj(text):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0; in_str = False; esc = False
    for i in range(start, len(text)):
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
                return text[start:i+1]
    return None


def call_backend(endpoint, model, prompt_text, images_b64, timeout_sec=90):
    content = [{"type": "text", "text": prompt_text}]
    for b in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    payload = {"model": model, "temperature": 0,
               "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=timeout_sec)
        data = json.loads(r.read().decode())
        text = data["choices"][0]["message"]["content"]
        lat = time.time() - t
        return {"ok": True, "raw": text, "lat": lat}
    except urllib.error.HTTPError as e:
        return {"ok": False, "err": f"HTTP {e.code}: {e.read()[:200]}", "lat": time.time()-t}
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {str(e)[:200]}", "lat": time.time()-t}


def parse_action(raw):
    candidate = first_json_obj(raw or "")
    if candidate is None:
        return None, None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None, None
    action = str(obj.get("recommended_action", "")).strip().upper()
    return action or None, obj


def load_frames(bag):
    return [json.loads(line) for line in open(REPO / "extracted_social_nav" / bag / "frames.jsonl")]


def pick_window(frames, length=5, policy="last"):
    """Pick one representative sequence window from the extracted frames."""
    n = len(frames)
    if n <= length:
        idx = list(range(n))
    elif policy == "last":
        idx = list(range(n - length, n))
    elif policy == "median":
        start = max(0, (n - length) // 2)
        idx = list(range(start, start + length))
    elif policy == "tail3":
        # For tail subsample: return last 3 frames within the last 5-frame window.
        last5 = list(range(n - length, n))
        idx = last5[-3:]
    else:
        raise ValueError(policy)
    return idx, [frames[i] for i in idx]


def run_one(bag, model_name, config, window_policy="median"):
    """Use the middle 5-frame window by default; that's where the interaction
    signal is strongest on these scenarios (end of clip often empty)."""
    # Load full (pre-subsample) frames so we don't miss the interaction.
    full_path = REPO / "extracted_social_nav" / bag / "frames.jsonl.full"
    src = full_path if full_path.exists() else REPO / "extracted_social_nav" / bag / "frames.jsonl"
    frames = [json.loads(l) for l in open(src)]
    if config == "BEFORE":
        idx, rows = pick_window(frames, 5, window_policy)
        prompt = OLD_SEQ
    elif config == "AFTER_prompt":
        idx, rows = pick_window(frames, 5, window_policy)
        prompt = NEW_SEQ
    elif config == "AFTER_tail":
        # AFTER_tail: take the full 5-window then keep only the last 3 frames.
        idx_full, _ = pick_window(frames, 5, window_policy)
        idx = idx_full[-3:]
        rows = [frames[i] for i in idx]
        prompt = NEW_SEQ
    else:
        raise ValueError(config)
    img_paths = [REPO / "extracted_social_nav" / bag / r["image_path"] for r in rows]
    b64 = [compress_ppm_to_jpg_b64(p) for p in img_paths]
    endpoint, model = MODELS[model_name]
    resp = call_backend(endpoint, model, prompt, b64)
    action, parsed = (None, None)
    if resp["ok"]:
        action, parsed = parse_action(resp["raw"])
    return {
        "bag": bag, "model": model_name, "config": config,
        "num_images": len(b64), "frame_indices": idx,
        "action": action, "ok": resp["ok"], "err": resp.get("err", ""),
        "lat": round(resp["lat"], 2),
        "response_json": parsed,
        "raw": (resp.get("raw") or "")[:400],
    }


def main():
    bags_arg = sys.argv[1:]
    targets = bags_arg if bags_arg else list(BAGS.keys())
    results = []
    for bag in targets:
        for model_name in ("Qwen", "InternVL"):
            for cfg in ("BEFORE", "AFTER_prompt", "AFTER_tail"):
                r = run_one(bag, model_name, cfg)
                results.append(r)
                mv = r["action"] or "?"
                print(f"{bag:<8} {model_name:<8} {cfg:<14} action={mv:<8} ok={r['ok']} lat={r['lat']:.2f}s  imgs={r['num_images']}  err={r['err'][:60]}")
    # Dump full results as JSON for later analysis
    out = Path("/tmp/prompt_ab_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results -> {out}")


if __name__ == "__main__":
    main()
