#!/usr/bin/env python3
"""Bind four query images and keep native deliveries available for visual grading."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

SOURCE_RUNNER_SHA256 = (
    "09fe4852156fae51e63445031468305adc774600857d33de07a692e4036af745"
)
# Preserve the pinned Core's 1 MiB reasoning guard and its original error receipt.
# The paired report applies the experiment's explicit budget policy separately.


def adapt_runner(raw: bytes) -> bytes:
    if hashlib.sha256(raw).hexdigest() != SOURCE_RUNNER_SHA256:
        raise ValueError("The published history32 input adapter changed")
    source = raw.decode()
    replacements = {
        "materials = workspace / 'materials'": "materials = workspace / 'problem_assets'",
        "names = {'anchor.jpg', 'three-views.png'}": (
            "names = {'reference-01.jpg', 'reference-02.jpg', 'reference-03.jpg', 'reference-04.jpg'}"
        ),
        "len(images) != 2": "len(images) != 4",
        "Exactly the two approved public images required": "Exactly the four approved query images required",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("The frozen history32 input contract no longer matches")
        source = source.replace(old, new)
    compile(source, "history32-query-input", "exec")
    return source.encode()


def normalize_terminal(
    returncode: int, result_path: Path, destination: Path, workspace: Path
) -> int:
    if returncode != 0 and result_path.is_file():
        result = json.loads(result_path.read_text())
        outcome = result.get("raw", {}).get("outcome", {})
        if outcome.get("class") in {"agent", "model"}:
            result.update(status="OK", generation_status="ERROR", model_outcome=outcome)
            result_path.write_text(json.dumps(result, indent=2))
            delivered = [destination / "scene.blend", workspace / "scene.blend"]
            if not any(path.is_file() and path.stat().st_size for path in delivered):
                (destination / "model-failure.json").write_text(
                    json.dumps({"status": "MODEL_FAILURE", "outcome": outcome})
                )
            return 0
    return returncode


def main() -> None:
    original = Path(os.environ["ARENA_ENV_DIR"]) / "native/history32.pyz"
    scratch = Path(tempfile.mkdtemp(prefix="overfit-history32-"))
    target = scratch / "history32.pyz"
    runner_path = "sdk-overlays/runner"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(target, "w") as output:
        original_runner = source.read(runner_path)
        adapted = adapt_runner(original_runner)
        binding = json.loads(source.read("release-binding.json"))
        if binding["sdk_overlay_sha256"]["runner"] != SOURCE_RUNNER_SHA256:
            raise ValueError("The source release binding does not match its runner")
        binding["sdk_overlay_sha256"]["runner"] = hashlib.sha256(adapted).hexdigest()
        for item in source.infolist():
            if item.filename == "release-binding.json":
                output.writestr(item, json.dumps(binding))
                continue
            output.writestr(
                item,
                adapted if item.filename == runner_path else source.read(item.filename),
            )
    receipt = {
        "source": "areal-harness-rl-history32@1.0.0",
        "original_runner_sha256": SOURCE_RUNNER_SHA256,
        "adapted_runner_sha256": hashlib.sha256(adapted).hexdigest(),
        "adaptation": "Bind exactly the four original query images from problem_assets",
        "native_binary_and_settings_unchanged": True,
    }
    destination = Path(os.environ["ARENA_AGENT_OUTPUT_DIR"])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "history32-input-adaptation.json").write_text(
        json.dumps(receipt, indent=2)
    )
    process = subprocess.Popen([sys.executable, str(target)])
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(
            signum, lambda number, frame: process.send_signal(number)
        )
    try:
        returncode = process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    result_path = Path(os.environ["ARENA_OUTPUT_DIR"]) / "harness_result.json"
    raise SystemExit(
        normalize_terminal(
            returncode, result_path, destination, Path(os.environ["ARENA_WORKSPACE"])
        )
    )


if __name__ == "__main__":
    main()
