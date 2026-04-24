#!/usr/bin/env python3
"""
Small HTTP wrapper around configurable VLM backends for Go1 social nav.

API:
    GET  /health
    POST /analyze

Request format for /analyze:
    {
      "image_base64": "<base64 image bytes>"
    }

Response format:
    {
      "person_detected": true,
      "person_in_front": true,
      "raw_text": "...",
      "ok": true
    }
"""

import base64
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from social_nav_eval_prompts import PROMPT_CONFIG


HOST = os.getenv("VLM_WRAPPER_HOST", "0.0.0.0")
PORT = int(os.getenv("VLM_WRAPPER_PORT", "8000"))


def getenv_first(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def getenv_float(*names: str, default: float) -> float:
    raw = getenv_first(*names, default=str(default))
    return float(raw)


def getenv_bool(*names: str, default: bool) -> bool:
    raw = getenv_first(*names, default="true" if default else "false")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


VLM_BACKEND = getenv_first("VLM_BACKEND", default="local")
VLM_BASE_URL = getenv_first(
    "VLM_BASE_URL",
    "LM_STUDIO_BASE_URL",
    default="http://localhost:1234/v1",
)
VLM_MODEL = getenv_first("VLM_MODEL", "LM_STUDIO_MODEL", default="local-model")
VLM_API_KEY = getenv_first("VLM_API_KEY", default="")
VLM_TIMEOUT = getenv_float("VLM_TIMEOUT", "LM_STUDIO_TIMEOUT", default=60.0)
VLM_ROUTE_STYLE = getenv_first("VLM_ROUTE_STYLE", default="openai_chat")
VLM_CHAT_PATH = getenv_first("VLM_CHAT_PATH", default="/chat/completions")
VLM_DEBUG = getenv_bool("VLM_DEBUG", default=False)

PROMPT_TEXT = (
    "Is there a person in this image?\n"
    "Answer ONLY in JSON:\n"
    '{"person_detected": true/false, "person_in_front": true/false}'
)

logging.basicConfig(
    level=logging.DEBUG if VLM_DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s [vlm_wrapper] %(message)s",
)
logger = logging.getLogger("vlm_wrapper")

app = FastAPI()


class AnalyzeRequest(BaseModel):
    image_base64: str


class NavigationAnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    images_base64: Optional[List[str]] = None
    prompt_name: str = "single_image_navigation"


def error_response(message: str) -> Dict[str, Any]:
    return {
        "person_detected": False,
        "person_in_front": False,
        "raw_text": "",
        "ok": False,
        "error": message,
    }


def coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def parse_model_json(raw_text: str) -> Optional[Dict[str, bool]]:
    candidate = extract_first_json_object(raw_text.strip())
    if candidate is None:
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    person_detected = coerce_bool(parsed.get("person_detected"))
    person_in_front = coerce_bool(parsed.get("person_in_front"))
    if person_detected is None or person_in_front is None:
        return None

    return {
        "person_detected": person_detected,
        "person_in_front": person_in_front,
    }


def parse_generic_json(raw_text: str) -> Optional[Dict[str, Any]]:
    candidate = extract_first_json_object(raw_text.strip())
    if candidate is None:
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed


def sanitize_base_url(url: str) -> str:
    parts = urlsplit(url)
    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    if parts.username or parts.password:
        netloc = f"{hostname}{port}"
    else:
        netloc = parts.netloc
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def compress_image_to_base64(image: Image.Image) -> str:
    max_width = 512
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=60)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_and_compress_image_base64(image_base64: str) -> str:
    image_bytes = base64.b64decode(image_base64, validate=True)
    image = Image.open(io.BytesIO(image_bytes))
    return compress_image_to_base64(image)


def build_message_content(prompt_text: str, image_base64_list: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for image_base64 in image_base64_list:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            }
        )
    return content


def build_openai_vision_payload(prompt_text: str, image_base64_list: List[str]) -> Dict[str, Any]:
    return {
        "model": VLM_MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": build_message_content(prompt_text, image_base64_list),
            }
        ],
    }


def build_custom_backend_payload(prompt_text: str, image_base64_list: List[str]) -> Dict[str, Any]:
    return {
        "model": VLM_MODEL,
        "prompt": prompt_text,
        "images_base64": image_base64_list,
    }


def build_backend_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if VLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLM_API_KEY}"
    return headers


def backend_endpoint() -> str:
    if VLM_ROUTE_STYLE == "openai_chat":
        return urljoin(f"{VLM_BASE_URL.rstrip('/')}/", VLM_CHAT_PATH.lstrip("/"))
    return urljoin(f"{VLM_BASE_URL.rstrip('/')}/", VLM_CHAT_PATH.lstrip("/"))


def parse_openai_response_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            if text_parts:
                return "\n".join(text_parts)
    raise ValueError("openai_response_missing_message_content")


def parse_custom_response_text(data: Dict[str, Any]) -> str:
    for key in ("raw_text", "response", "generated_text", "text", "output"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(data.get("choices"), list):
        return parse_openai_response_text(data)
    raise ValueError("custom_response_missing_text")


def parse_vlm_response(data: Dict[str, Any]) -> str:
    if VLM_ROUTE_STYLE == "openai_chat":
        return parse_openai_response_text(data)
    return parse_custom_response_text(data)


def call_openai_compatible_backend(prompt_text: str, image_base64_list: List[str]) -> str:
    payload = build_openai_vision_payload(prompt_text, image_base64_list)
    endpoint = backend_endpoint()
    logger.info("Calling OpenAI-compatible VLM backend at %s", sanitize_base_url(endpoint))
    logger.info("Backend=%s model=%s images=%d", VLM_BACKEND, VLM_MODEL, len(image_base64_list))
    logger.info("Prompt length: %d chars", len(prompt_text))
    if VLM_DEBUG:
        logger.debug("Payload preview: %s", json.dumps(payload)[:1000])
    response = requests.post(
        endpoint,
        json=payload,
        headers=build_backend_headers(),
        timeout=VLM_TIMEOUT,
    )
    logger.info("VLM backend response status: %s", response.status_code)
    if not response.ok:
        logger.error("VLM backend error body: %s", response.text[:1000])
    response.raise_for_status()
    return parse_vlm_response(response.json())


def call_custom_backend(prompt_text: str, image_base64_list: List[str]) -> str:
    payload = build_custom_backend_payload(prompt_text, image_base64_list)
    endpoint = backend_endpoint()
    logger.info("Calling custom VLM backend at %s", sanitize_base_url(endpoint))
    logger.info("Backend=%s model=%s images=%d", VLM_BACKEND, VLM_MODEL, len(image_base64_list))
    if VLM_DEBUG:
        logger.debug("Custom payload preview: %s", json.dumps(payload)[:1000])
    response = requests.post(
        endpoint,
        json=payload,
        headers=build_backend_headers(),
        timeout=VLM_TIMEOUT,
    )
    logger.info("VLM backend response status: %s", response.status_code)
    if not response.ok:
        logger.error("VLM backend error body: %s", response.text[:1000])
    response.raise_for_status()
    return parse_vlm_response(response.json())


def call_vlm_backend(prompt_text: str, image_base64_list: List[str]) -> str:
    if VLM_ROUTE_STYLE == "openai_chat":
        return call_openai_compatible_backend(prompt_text, image_base64_list)
    return call_custom_backend(prompt_text, image_base64_list)


def call_lm_studio(image_base64: str) -> str:
    return call_vlm_backend(PROMPT_TEXT, [image_base64])


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "host": HOST,
        "port": PORT,
        "backend": VLM_BACKEND,
        "mode": "local" if VLM_BACKEND == "local" else "remote",
        "route_style": VLM_ROUTE_STYLE,
        "base_url": sanitize_base_url(VLM_BASE_URL),
        "chat_path": VLM_CHAT_PATH,
        "model": VLM_MODEL,
        "timeout_sec": VLM_TIMEOUT,
        "debug": VLM_DEBUG,
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> Dict[str, Any]:
    start_time = time.time()
    logger.info("Analyze request started")

    try:
        compressed_image_base64 = load_and_compress_image_base64(request.image_base64)
    except Exception:
        logger.exception("Invalid base64 image")
        return error_response("invalid image_base64 payload")

    try:
        raw_text = call_lm_studio(compressed_image_base64)
    except requests.RequestException as exc:
        logger.exception("VLM backend request failed")
        return error_response(f"vlm_backend_request_failed: {exc}")
    except ValueError as exc:
        logger.exception("VLM backend response parsing failed")
        return error_response(f"vlm_backend_response_parse_failed: {exc}")
    except Exception as exc:
        logger.exception("Unexpected VLM backend failure")
        return error_response(f"vlm_backend_unexpected_error: {exc}")

    parsed = parse_model_json(raw_text)
    if parsed is None:
        logger.warning("Failed to parse VLM output: %s", raw_text)
        logger.warning("Raw VLM response: %s", raw_text)
        return {
            "person_detected": False,
            "person_in_front": False,
            "raw_text": raw_text,
            "ok": False,
            "error": "json_parse_failed",
        }

    duration = time.time() - start_time
    logger.info("Parse success in %.3fs", duration)
    return {
        "person_detected": parsed["person_detected"],
        "person_in_front": parsed["person_in_front"],
        "raw_text": raw_text,
        "ok": True,
    }


@app.post("/analyze_navigation")
def analyze_navigation(request: NavigationAnalyzeRequest) -> Dict[str, Any]:
    start_time = time.time()
    prompt_cfg = PROMPT_CONFIG.get(request.prompt_name)
    if prompt_cfg is None:
        return {
            "ok": False,
            "error": f"unknown_prompt_name: {request.prompt_name}",
            "prompt_name": request.prompt_name,
            "response_json": None,
            "raw_text": "",
        }

    raw_images = []
    if request.images_base64:
        raw_images.extend(request.images_base64)
    if request.image_base64:
        raw_images.append(request.image_base64)
    if not raw_images:
        return {
            "ok": False,
            "error": "missing_image_payload",
            "prompt_name": request.prompt_name,
            "response_json": None,
            "raw_text": "",
        }

    try:
        compressed_images = [load_and_compress_image_base64(item) for item in raw_images]
    except Exception:
        logger.exception("Invalid base64 image sequence")
        return {
            "ok": False,
            "error": "invalid image payload",
            "prompt_name": request.prompt_name,
            "response_json": None,
            "raw_text": "",
        }

    try:
        raw_text = call_vlm_backend(prompt_cfg["prompt"], compressed_images)
    except requests.RequestException as exc:
        logger.exception("VLM backend request failed")
        return {
            "ok": False,
            "error": f"vlm_backend_request_failed: {exc}",
            "prompt_name": request.prompt_name,
            "response_json": None,
            "raw_text": "",
        }
    except ValueError as exc:
        logger.exception("VLM backend response parsing failed")
        return {
            "ok": False,
            "error": f"vlm_backend_response_parse_failed: {exc}",
            "prompt_name": request.prompt_name,
            "response_json": None,
            "raw_text": "",
        }
    except Exception as exc:
        logger.exception("Unexpected VLM backend failure")
        return {
            "ok": False,
            "error": f"vlm_backend_unexpected_error: {exc}",
            "prompt_name": request.prompt_name,
            "response_json": None,
            "raw_text": "",
        }

    parsed = parse_generic_json(raw_text)
    duration = time.time() - start_time
    if parsed is None:
        logger.warning("Failed to parse navigation response: %s", raw_text)
        return {
            "ok": False,
            "error": "json_parse_failed",
            "prompt_name": request.prompt_name,
            "response_json": None,
            "raw_text": raw_text,
            "num_images": len(compressed_images),
            "latency_sec": duration,
        }

    return {
        "ok": True,
        "error": "",
        "prompt_name": request.prompt_name,
        "response_json": parsed,
        "raw_text": raw_text,
        "num_images": len(compressed_images),
        "latency_sec": duration,
    }


if __name__ == "__main__":
    import uvicorn

    print(f"Server host/port: {HOST}:{PORT}")
    print(f"VLM backend: {VLM_BACKEND}")
    print(f"VLM mode: {'local' if VLM_BACKEND == 'local' else 'remote'}")
    print(f"VLM base URL: {sanitize_base_url(VLM_BASE_URL)}")
    print(f"VLM route style: {VLM_ROUTE_STYLE}")
    print(f"Model name: {VLM_MODEL}")
    uvicorn.run(app, host=HOST, port=PORT)
