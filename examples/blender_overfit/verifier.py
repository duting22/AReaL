"""Image-only adapter of the history32 Stream's Q/D/A visual judging rubrics."""

import argparse
import base64
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from examples.blender_overfit.prepare import digest, write_json

WEIGHTS = {"quality": 0.4, "detail": 0.4, "aesthetics": 0.2}


def score_integer(value: Any) -> float | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= 100:
        raise ValueError("Expected a 0–100 integer score or null")
    return value / 100.0


def parse_votes(qa: dict, detail: dict, features: list[dict]) -> dict[str, Any]:
    sample = qa["samples"]["sample"]
    quality = score_integer(sample["quality"]["score"])
    aesthetics = score_integer(sample["aesthetics"]["score"])
    if sample.get("zoom_requests"):
        raise ValueError("The fixed evidence budget permits no further zoom requests")
    if detail.get("sample_id") != "sample":
        raise ValueError("Detail vote sample identity mismatch")
    items = detail["items"]
    expected = {item["id"] for item in features}
    if len(items) != len(expected) or {item["id"] for item in items} != expected:
        raise ValueError("Detail vote must cover each frozen feature exactly once")
    scores = []
    for item in items:
        if item["status"] not in {"rated", "unknown", "reference_conflict"}:
            raise ValueError("Unknown detail evidence status")
        value = score_integer(item["score"])
        if item["status"] == "rated":
            if value is None or not item.get("observed_regions"):
                raise ValueError("Rated detail requires a score and observed regions")
            scores.append(value)
        elif value is not None:
            raise ValueError(
                "Unknown or conflicting detail must not receive a numeric score"
            )
    detail_score = sum(scores) / len(scores) if scores else None
    values = {"quality": quality, "detail": detail_score, "aesthetics": aesthetics}
    complete = all(value is not None for value in values.values())
    result = {
        "schema": "blender-overfit-visual/v1",
        "status": "OK" if complete else "INSUFFICIENT_EVIDENCE",
        "score": sum(WEIGHTS[key] * value for key, value in values.items())
        if complete
        else None,
        "components": {
            **{
                key: {"value": value, "weight": WEIGHTS[key]}
                for key, value in values.items()
            },
            "shape": {"value": None, "weight": 0, "status": "NOT_APPLICABLE"},
            "structure": {"value": None, "weight": 0, "status": "NOT_APPLICABLE"},
        },
        "detail_coverage": {"rated": len(scores), "total": len(features)},
        "formula": "0.4*quality + 0.4*detail + 0.2*aesthetics",
        "raw": {"qa": qa, "detail": detail},
    }
    result["sub_scores"] = {
        "可见几何质量 Q（视觉总分40%）": quality,
        "具体可见细部 D（视觉总分40%）": detail_score,
        "细部美术精度 A（视觉总分20%）": aesthetics,
    }
    result["raw"]["reward_observation"] = {
        "schema_version": 1,
        "verifier_version": "blender-overfit-visual-v1",
        "full_score": result["score"],
        "formula": result["formula"],
        "additive_component_ids": list(WEIGHTS),
        "components": {
            key: {
                **component,
                "counted": key in WEIGHTS,
                "status": "OK"
                if component["value"] is not None
                else component.get("status", "INSUFFICIENT_EVIDENCE"),
                "contribution": component["value"] * component["weight"]
                if component["value"] is not None
                else None,
            }
            for key, component in result["components"].items()
        },
    }
    return result


def request_body(
    task_root: Path, task: dict, evidence_root: Path, kind: str, model: str
) -> dict:
    if kind not in {"qa", "detail"}:
        raise ValueError("Expected qa or detail")
    prompt = Path(__file__).with_name("prompts") / (kind + ".txt")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "description": task["description"],
                    "features": task["visual_features"],
                },
                ensure_ascii=False,
            ),
        }
    ]
    if len(task["reference_images"]) != 4:
        raise ValueError("Exactly the four query images are required")
    for index, image in enumerate(task["reference_images"], 1):
        path = task_root / image["path"]
        data = path.read_bytes()
        if digest(data) != image["sha256"]:
            raise ValueError("The original query image changed")
        mime = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
        content.extend(
            [
                {"type": "text", "text": f"P{index}: original query reference"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64," + base64.b64encode(data).decode(),
                        "detail": "high",
                    },
                },
            ]
        )
    evidence = json.loads((evidence_root / "render-manifest.json").read_text())
    expected = [f"sample-V{index}" for index in range(1, 10)] + [
        "sample-Z1",
        "sample-Z2",
    ]
    if [image["id"] for image in evidence["images"]] != expected:
        raise ValueError("Nine candidate views and two fixed zooms are required")
    for image in evidence["images"]:
        path = evidence_root / image["path"]
        data = path.read_bytes()
        if digest(data) != image["sha256"]:
            raise ValueError("Candidate evidence changed after rendering")
        content.extend(
            [
                {
                    "type": "text",
                    "text": image["id"]
                    + ": actual neutral-material candidate geometry",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(data).decode(),
                        "detail": "high",
                    },
                },
            ]
        )
    return {
        "model": model,
        "stream": False,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt.read_text()},
            {"role": "user", "content": content},
        ],
    }


def read_judge_stream(response: Any) -> dict:
    """Assemble one streamed completion without changing its sampling parameters."""
    result: dict[str, Any] = {}
    message = {"role": "assistant", "content": ""}
    finish = None
    done = False
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise RuntimeError("Judge stream returned an error")
        for key in ("id", "model", "created", "usage"):
            if chunk.get(key) is not None:
                result[key] = chunk[key]
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Expected one judge completion")
            for key in ("content", "reasoning_content"):
                value = choice.get("delta", {}).get(key)
                if value:
                    message[key] = message.get(key, "") + value
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    if not done or finish is None:
        raise ValueError("Incomplete judge stream")
    result["choices"] = [{"index": 0, "message": message, "finish_reason": finish}]
    return result


def call_judge(
    body: dict,
    base_url: str,
    api_key: str,
    timeout: float,
    response_path: Path | None = None,
) -> tuple[dict, dict]:
    if not base_url or not api_key:
        raise ValueError(
            "Judge URL and credential must be supplied in the reward environment"
        )
    url = base_url.rstrip("/") + "/chat/completions"
    streaming = os.environ.get("BUILDING_VLM_STREAM") == "1"
    transmitted = (
        {**body, "stream": True, "stream_options": {"include_usage": True}}
        if streaming
        else body
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(transmitted).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = read_judge_stream(response) if streaming else json.load(response)
    if response_path is not None:
        write_json(response_path, payload)
    if payload["choices"][0]["finish_reason"] != "stop":
        raise ValueError("Judge completion was truncated")
    parsed = json.loads(payload["choices"][0]["message"]["content"])
    return parsed, {"model": payload.get("model"), "usage": payload.get("usage")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    task = next(task for task in manifest["tasks"] if task["task_id"] == args.task_id)
    provenance = json.loads(
        Path(__file__).with_name("prompts").joinpath("provenance.json").read_text()
    )
    model = os.environ.get("BUILDING_VLM_MODEL", provenance["judge_model"])
    args.output.mkdir(parents=True, exist_ok=False)
    votes = {}
    receipts = {}
    for kind in ["qa", "detail"]:
        body = request_body(args.dataset, task, args.evidence, kind, model)
        write_json(args.output / (kind + "-request.json"), body)
        if args.dry_run:
            continue
        started = time.monotonic()
        try:
            votes[kind], receipts[kind] = call_judge(
                body,
                os.environ.get("BUILDING_VLM_BASE_URL", ""),
                os.environ.get("BUILDING_VLM_API_KEY", ""),
                args.timeout,
                args.output / (kind + "-response.json"),
            )
            write_json(args.output / (kind + "-vote.json"), votes[kind])
        except Exception as exc:
            write_json(
                args.output / "result.json",
                {
                    "status": "EVAL_ERROR",
                    "score": None,
                    "phase": kind,
                    "error_type": type(exc).__name__,
                    "seconds": time.monotonic() - started,
                },
            )
            raise
    if args.dry_run:
        write_json(
            args.output / "result.json",
            {"status": "DRY_RUN", "model_calls": 0, "score": None},
        )
        return
    result = parse_votes(votes["qa"], votes["detail"], task["visual_features"])
    result.update(
        task_id=args.task_id,
        checkpoint=manifest["checkpoint"],
        provenance=provenance,
        receipts=receipts,
        candidate_sha256=json.loads(
            (args.evidence / "render-manifest.json").read_text()
        )["candidate_sha256"],
    )
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
