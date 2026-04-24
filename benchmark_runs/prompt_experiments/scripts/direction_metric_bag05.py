#!/usr/bin/env python3
"""Produce real prediction rows on bag_05 and compute direction metrics.

bag_05 = right_to_left_crossing. We probe multiple stride-4 5-frame windows
along the clip so the models have enough temporal displacement to classify
motion as crossing. The outputs are written in the standard prediction-row
schema expected by ``social_nav_eval.compute_direction_metrics``.
"""
import base64, io, json, sys, time, urllib.request
from pathlib import Path
from PIL import Image

REPO = Path("/home/aaquib/where-am-I")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "motion_control" / "scripts"))
from social_nav_eval_prompts import PROMPT_SEQUENCE_IMAGES as PROMPT
from social_nav_eval import compute_direction_metrics, normalize_action

BAG = "bag_05"
SCAN_START = 80
SCAN_END = 260
STRIDE = 4
WIN_LEN = 5
WIN_SPACING = 10   # distinct anchors → more independent samples

MODELS = {
    "InternVL": ("http://10.157.141.181:8000/v1/chat/completions", "OpenGVLab/InternVL3_5-14B-HF"),
    "Qwen":     ("http://10.157.141.181:8001/v1/chat/completions", "Qwen3-VL-30B"),
}


def enc(p):
    img = Image.open(p)
    if img.width > 512:
        img = img.resize((512, int(img.height * 512 / img.width)), Image.LANCZOS)
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


def call(endpoint, model, imgs):
    content = [{"type":"text","text":PROMPT}] + [
        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b}"}} for b in imgs]
    body = json.dumps({"model": model, "temperature": 0,
                       "messages":[{"role":"user","content":content}]}).encode()
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type":"application/json"})
    t = time.time()
    raw = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
    cand = first_json_obj(raw)
    obj = {}
    if cand:
        try: obj = json.loads(cand)
        except json.JSONDecodeError: obj = {}
    return {"raw": raw, "obj": obj, "lat": round(time.time()-t, 2)}


def main():
    frames = [json.loads(l) for l in open(REPO/"extracted_social_nav"/BAG/"frames.jsonl.full")]
    n = len(frames)
    # Build windows: 5 frames spaced STRIDE apart; anchor stepping by WIN_SPACING.
    anchors = list(range(SCAN_START, min(SCAN_END, n - (WIN_LEN-1)*STRIDE), WIN_SPACING))
    print(f"{BAG} frames={n} windows={len(anchors)} anchors={anchors}")

    per_model_rows = {}
    for model_name, (ep, model) in MODELS.items():
        rows = []
        for anchor in anchors:
            idx = [anchor + i*STRIDE for i in range(WIN_LEN)]
            if idx[-1] >= n: continue
            imgs = [enc(REPO/"extracted_social_nav"/BAG/frames[i]["image_path"]) for i in idx]
            r = call(ep, model, imgs)
            obj = r["obj"]
            pred_action = (obj.get("recommended_action") or "").upper() or None
            rows.append({
                "sample_id": f"sequence_{anchor:06d}",
                "success": bool(obj),
                "predicted_action": pred_action,
                "response_json": obj,
                "raw_response": r["raw"],
                "latency_sec": r["lat"],
                "frame_indices": idx,
            })
            print(f"  {model_name:<8} anchor={anchor:<4} action={pred_action} "
                  f"motion={obj.get('motion','-')} cross_dir={obj.get('crossing_direction','-')} "
                  f"avoid={obj.get('recommended_avoidance_side','-')}")
        per_model_rows[model_name] = rows

    # Compute and print metrics.
    print("\nDirection Metrics Summary")
    for model_name, rows in per_model_rows.items():
        m = compute_direction_metrics(rows)
        drc = m["direction_rule_consistency"]; dar = m["direction_activation_rate"]; cdr = m["crossing_detection_rate"]
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        print(f"\nModel: {model_name}")
        print(f"- direction_rule_consistency: {fmt(drc)}  "
              f"(correct {m['n_direction_correct']} / valid {m['n_direction_valid']})")
        print(f"- direction_activation_rate: {fmt(dar)}  "
              f"(LEFT/RIGHT {m['n_direction_activated']} / crossing {m['n_crossing']})")
        print(f"- crossing_detection_rate:   {fmt(cdr)}  "
              f"(crossing {m['n_crossing']} / all {m['n_all_successful']})")

    # Persist rows for any follow-up analysis.
    out = Path("/tmp/bag05_direction_predictions.json")
    out.write_text(json.dumps(per_model_rows, indent=2, default=str))
    print(f"\nRaw predictions -> {out}")


if __name__ == "__main__":
    main()
