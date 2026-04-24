#!/usr/bin/env python3
"""bag_10 validation: BEFORE | AFTER_current_prompt | AFTER_receding_fix | AFTER_receding_fix_tail."""
import base64, io, json, sys, time, urllib.request, urllib.error
from pathlib import Path
from PIL import Image

REPO = Path("/home/aaquib/where-am-I")
sys.path.insert(0, str(REPO))
from social_nav_eval_prompts import PROMPT_SEQUENCE_IMAGES as RECEDING_FIX_SEQ

# Snapshot of the NEW-but-pre-receding-fix prompt (captured before editing).
CURRENT_NEW_SEQ = Path("/tmp/current_new_seq_prompt.txt").read_text()

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

BAG = "bag_10"
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
    start = text.find("{");
    if start == -1: return None
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
            if depth == 0: return text[start:i+1]
    return None


def call(model_name, prompt, images):
    endpoint, model = MODELS[model_name]
    content = [{"type":"text","text":prompt}] + [
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b}"}} for b in images]
    payload = {"model": model, "temperature": 0, "messages":[{"role":"user","content":content}]}
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"})
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
    return {"ok": True, "raw": raw, "action": (obj.get("recommended_action") or "").upper(),
            "obj": obj, "lat": round(time.time()-t, 2)}


def pick_window(frames, length=5):
    # Q1 window: receding signal is visible (person already walking away
    # but still clearly present). The middle of bag_10 shows a brief pause
    # the models read as "stationary+blocked", which defeats the receding
    # rule. Q1 is the representative receding window.
    n = len(frames)
    start = max(0, int(n * 0.20))
    start = min(start, n - length)
    return list(range(start, start + length))


def main():
    # Prefer full frames so receding is visible across the window
    full = REPO / "extracted_social_nav" / BAG / "frames.jsonl.full"
    src = full if full.exists() else REPO / "extracted_social_nav" / BAG / "frames.jsonl"
    frames = [json.loads(l) for l in open(src)]
    window5 = pick_window(frames, 5)
    window_tail3 = window5[-3:]
    imgs5 = [enc(REPO/"extracted_social_nav"/BAG/frames[i]["image_path"]) for i in window5]
    imgs3 = [enc(REPO/"extracted_social_nav"/BAG/frames[i]["image_path"]) for i in window_tail3]

    configs = [
        ("BEFORE",                 OLD_SEQ,          imgs5),
        ("AFTER_current_prompt",   CURRENT_NEW_SEQ,  imgs5),
        ("AFTER_receding_fix",     RECEDING_FIX_SEQ, imgs5),
        ("AFTER_receding_fix_tail",RECEDING_FIX_SEQ, imgs3),
    ]

    rows = []
    for model_name in ("InternVL", "Qwen"):
        for cfg_name, prompt, imgs in configs:
            r = call(model_name, prompt, imgs)
            action = r.get("action") or "ERR"
            obj = r.get("obj") or {}
            print(f"{BAG} {model_name:<8} {cfg_name:<24} action={action:<8} "
                  f"motion={obj.get('motion','-'):<12} blocked={obj.get('path_blocked_latest_frame', obj.get('path_blocked','-'))} "
                  f"lat={r.get('lat','?')}s")
            rows.append({"model": model_name, "config": cfg_name, "action": action,
                         "motion": obj.get("motion"), "blocked": obj.get("path_blocked_latest_frame", obj.get("path_blocked")),
                         "risk": obj.get("risk_level"), "unc": obj.get("uncertainty_reason",""),
                         "num_images": len(imgs)})
    Path("/tmp/bag10_receding_fix_results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
