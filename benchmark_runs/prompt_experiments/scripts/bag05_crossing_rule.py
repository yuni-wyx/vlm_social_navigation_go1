#!/usr/bin/env python3
"""bag_05 (right-to-left crossing) A/B: current-prompt vs crossing-direction-rule prompt.

bag_05 = right_to_left_crossing → person moves LEFTWARD → robot should prefer RIGHT.
"""
import base64, io, json, sys, time, urllib.request
from pathlib import Path
from PIL import Image

REPO = Path("/home/aaquib/where-am-I")
sys.path.insert(0, str(REPO))
from social_nav_eval_prompts import PROMPT_SEQUENCE_IMAGES as AFTER_SEQ   # with crossing rule

# Snapshot of prompt BEFORE the crossing rule (captured earlier).
BEFORE_SEQ = Path("/tmp/prompt_before_crossing.txt").read_text()

BAG = "bag_05"
EXPECTED_MOTION_DIRECTION = "leftward"   # right_to_left
EXPECTED_AVOIDANCE_SIDE = "right"         # opposite of motion

MODELS = {
    "InternVL": ("http://10.157.141.181:8000/v1/chat/completions", "OpenGVLab/InternVL3_5-14B-HF"),
    "Qwen":     ("http://10.157.141.181:8001/v1/chat/completions", "Qwen3-VL-30B"),
}


def enc(p):
    img = Image.open(p)
    if img.width > 512:
        r = 512 / img.width
        img = img.resize((512, int(img.height * r)), Image.LANCZOS)
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG", quality=60)
    return base64.b64encode(buf.getvalue()).decode()


def first_json_obj(text):
    s = text.find("{");
    if s == -1: return None
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
            if depth == 0: return text[s:i+1]
    return None


def call(model_name, prompt, imgs):
    endpoint, model = MODELS[model_name]
    content = [{"type":"text","text":prompt}] + [
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b}"}} for b in imgs]
    body = json.dumps({"model": model, "temperature": 0,
                       "messages":[{"role":"user","content":content}]}).encode()
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type":"application/json"})
    t = time.time()
    try:
        raw = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
    except Exception as e:
        return {"ok": False, "err": str(e)[:200], "lat": time.time()-t}
    cand = first_json_obj(raw)
    obj = {}
    if cand:
        try: obj = json.loads(cand)
        except json.JSONDecodeError: obj = {}
    return {"ok": True, "raw": raw, "obj": obj,
            "action": (obj.get("recommended_action") or "").upper(),
            "lat": round(time.time()-t, 2)}


def probe_windows(frames, length=5):
    """Probe multiple windows along the clip so we see where crossing is visible."""
    n = len(frames)
    bounds = {
        "q1":  max(0, int(n*0.20)),
        "mid": max(0, (n - length) // 2),
        "q3":  min(n - length, int(n*0.75)),
    }
    return {w: list(range(s, s + length)) for w, s in bounds.items()}


def main():
    full = REPO / "extracted_social_nav" / BAG / "frames.jsonl.full"
    src = full if full.exists() else REPO / "extracted_social_nav" / BAG / "frames.jsonl"
    frames = [json.loads(l) for l in open(src)]
    print(f"{BAG} frames={len(frames)}  expected: motion=right->leftward, avoidance=RIGHT\n")

    windows = probe_windows(frames, 5)
    for w, idx in windows.items():
        imgs = [enc(REPO/"extracted_social_nav"/BAG/frames[i]["image_path"]) for i in idx]
        for model_name in ("InternVL", "Qwen"):
            for cfg, prompt in [("BEFORE", BEFORE_SEQ), ("AFTER", AFTER_SEQ)]:
                r = call(model_name, prompt, imgs)
                obj = r.get("obj") or {}
                motion = obj.get("motion", "-")
                cross_dir = obj.get("crossing_direction", "-")
                avoid = obj.get("recommended_avoidance_side", "-")
                safer = obj.get("safer_lateral_side", "-")
                blocked = obj.get("path_blocked_latest_frame", obj.get("path_blocked", "-"))
                print(f"{w:<4} idx={idx} {model_name:<8} {cfg:<6} action={r.get('action','ERR'):<8} "
                      f"motion={motion:<19} cross_dir={str(cross_dir):<10} "
                      f"avoid={str(avoid):<6} safer={str(safer):<6} blocked={blocked}")


if __name__ == "__main__":
    main()
