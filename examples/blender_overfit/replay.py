#!/usr/bin/env python3
"""Deliver a recorded scene to Arena's separate grader without another model call."""

import hashlib
import json
import os
import shutil
from pathlib import Path


def delivered_scene(directory: Path) -> Path | None:
    """Retain a native delivery even when presentation collection failed."""
    for name in ("scene.blend", "scene-native.blend"):
        path = directory / name
        if path.is_file() and path.stat().st_size:
            return path
    return None


def main() -> None:
    evidence = Path(os.environ["ARENA_ENV_DIR"]) / "evidence"
    metadata = json.loads((evidence / "source.json").read_text())
    scene = evidence / "scene.blend"
    if hashlib.sha256(scene.read_bytes()).hexdigest() != metadata["scene_sha256"]:
        raise ValueError("Recorded delivery does not match its source scene")
    output = Path(os.environ["ARENA_AGENT_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scene, output / "scene.blend")
    (output / "replay-source.json").write_text(json.dumps(metadata, indent=2))
    result = {
        "status": "OK",
        "model_calls": 0,
        "scene_sha256": metadata["scene_sha256"],
        "source_task_id": metadata["source_task_id"],
    }
    (Path(os.environ["ARENA_OUTPUT_DIR"]) / "harness_result.json").write_text(
        json.dumps(result)
    )


if __name__ == "__main__":
    main()
