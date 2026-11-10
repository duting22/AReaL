"""One SFT-compatible request, followed by fixed Blender execution and packaging."""

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from examples.blender_overfit.prepare import digest, extract_code, write_json


def session_headers(block: str) -> dict[str, str]:
    """Keep optional gateway headers scoped to this request and preserve their bytes."""
    headers = {}
    for line in block.splitlines():
        if not line:
            continue
        name, separator, value = line.partition(":")
        if not separator or not name.strip() or "\r" in value:
            raise ValueError("Malformed custom header block")
        if name.strip().lower() in {"authorization", "host", "content-length"}:
            raise ValueError(
                "Custom headers cannot override gateway authentication or routing"
            )
        if name.strip().lower() in {key.lower() for key in headers}:
            raise ValueError("Duplicate custom header")
        headers[name.strip()] = value.lstrip(" ")
    session = {key.lower() for key in headers} & {"x-session-id", "x-session-token"}
    if len(session) == 1:
        raise ValueError("Both session gateway headers are required together")
    return headers


def clean_process_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not re.search(
            r"KEY|TOKEN|SECRET|PASSWORD|AUTHORIZATION|CREDENTIAL|CUSTOM_HEADERS",
            key,
            re.I,
        )
    }


def generate(
    messages: list[dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    timeout: float,
    stream: bool = False,
) -> dict[str, Any]:
    if not model or not base_url or not api_key:
        raise ValueError("Model name, base URL and API key must be configured")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    headers = {
        "Content-Type": "application/json",
        **session_headers(os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")),
        "Authorization": "Bearer " + api_key,
    }
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    request = urllib.request.Request(
        url + "/chat/completions", data=json.dumps(payload).encode(), headers=headers
    )
    # No model retry, repair turn, or tool feedback is allowed in the SFT arm.
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not stream:
            return json.load(response)
        result: dict[str, Any] = {"stream_transport": True}
        message: dict[str, Any] = {"role": "assistant", "content": ""}
        finish = None
        done = False
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError("The model gateway returned a streaming error")
            for key in ("id", "model", "created", "usage"):
                if chunk.get(key) is not None:
                    result[key] = chunk[key]
            for choice in chunk.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise RuntimeError("Unexpected additional completion choice")
                delta = choice.get("delta", {})
                for key in ("content", "reasoning_content", "reasoning"):
                    if delta.get(key):
                        message[key] = message.get(key, "") + delta[key]
                for key in ("tool_calls", "reasoning_details"):
                    if delta.get(key):
                        message.setdefault(key, []).extend(delta[key])
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
        if not done or finish is None:
            raise RuntimeError("The completion stream ended without a terminal event")
        result["choices"] = [{"index": 0, "message": message, "finish_reason": finish}]
        return result


def build_scene(
    code: Path, output: Path, blender: Path, timeout: float, render: bool
) -> None:
    command = [
        str(blender.resolve()),
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--threads",
        "2",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).with_name("scene.py").resolve()),
        "--",
        "--code",
        str(code.resolve()),
        "--output",
        str(output.resolve()),
    ]
    if render:
        command.append("--render")
    with (output / "blender.log").open("wb") as log:
        subprocess.run(
            command,
            cwd=output,
            env=clean_process_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    if not (output / "scene.blend").is_file():
        raise ValueError("Blender exited successfully without a scene.blend")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-timeout", type=float, default=1800)
    parser.add_argument("--blender-timeout", type=float, default=600)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = args.messages.read_bytes()
    (output / "input-messages.json").write_bytes(raw)
    started = time.monotonic()
    result: dict[str, Any] = {
        "mode": "sft",
        "status": "ERROR",
        "model": args.model,
        "input_sha256": digest(raw),
        "enable_thinking": False,
        "model_calls": 0,
    }
    try:
        result["model_calls"] = 1
        response = generate(
            json.loads(raw),
            model=args.model,
            base_url=args.base_url,
            api_key=os.environ.get("API_KEY", ""),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            timeout=args.model_timeout,
            stream=True,
        )
        write_json(output / "model-response.json", response)
        result["usage"] = response.get("usage")
        choice = response["choices"][0]
        message = choice["message"]
        if choice["finish_reason"] != "stop":
            raise ValueError("The one-shot completion did not finish normally")
        if message.get("tool_calls") or message.get("reasoning_content"):
            raise ValueError("The SFT arm must produce code without tools or thinking")
        completion = message["content"]
        (output / "completion.txt").write_text(completion)
        code = extract_code(completion)
        path = output / "generated.py"
        path.write_text(code)
        build_scene(path, output, args.blender, args.blender_timeout, args.render)
        result.update(
            status="OK", scene_sha256=digest((output / "scene.blend").read_bytes())
        )
    except urllib.error.HTTPError as exc:
        result.update(
            status="INFRA_ERROR", error_type="HTTPError", http_status=exc.code
        )
        raise
    except (urllib.error.URLError, TimeoutError) as exc:
        result.update(status="INFRA_ERROR", error_type=type(exc).__name__)
        raise
    except Exception as exc:
        result.update(error_type=type(exc).__name__)
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_json(output / "result.json", result)


if __name__ == "__main__":
    main()
