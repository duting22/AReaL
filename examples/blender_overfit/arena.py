"""EnvArena entrypoints for the one-shot arm, collection, and image-only reward."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

from examples.blender_overfit import single_turn
from examples.blender_overfit.prepare import digest, extract_code, write_json


def model_connection() -> tuple[str, str, str]:
    model = (
        os.environ.get("OPENAI_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or os.environ.get("MODEL_NAME", "")
    )
    base = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or os.environ.get("BASE_URL", "")
    )
    key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("API_KEY", "")
    )
    return model, base, key


def sft_main() -> None:
    output = Path(os.environ["ARENA_AGENT_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    messages = json.loads(Path(os.environ["ARENA_QUERY_PATH"]).read_text())
    model, base, key = model_connection()
    result = {
        "status": "ERROR",
        "mode": "sft",
        "model_calls": 0,
        "enable_thinking": False,
    }
    exit_code = 0
    try:
        result["model_calls"] = 1
        response = single_turn.generate(
            messages,
            model=model,
            base_url=base,
            api_key=key,
            temperature=1.0,
            max_tokens=65536,
            seed=42,
            timeout=1800,
            stream=True,
        )
        write_json(output / "model-response.json", response)
        choice = response["choices"][0]
        message = choice["message"]
        if any(
            message.get(field)
            for field in ("reasoning_content", "reasoning", "reasoning_details")
        ):
            raise RuntimeError("The serving configuration did not disable thinking")
        if choice["finish_reason"] != "stop" or message.get("tool_calls"):
            raise ValueError("Invalid non-thinking, non-agentic completion")
        (output / "completion.txt").write_text(message["content"])
        code = extract_code(message["content"])
        path = output / "generated.py"
        path.write_text(code)
        blender = Path(os.environ.get("BLENDER_BIN", "/opt/blender/blender"))
        single_turn.build_scene(path, output, blender, timeout=600, render=True)
        result.update(
            status="OK",
            usage=response.get("usage"),
            scene_sha256=digest((output / "scene.blend").read_bytes()),
        )
    except (
        ValueError,
        SyntaxError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        # A malformed program is a valid model failure; let the reward phase record zero.
        result.update(status="MODEL_FAILURE", error_type=type(exc).__name__)
        write_json(output / "model-failure.json", result)
    except Exception as exc:
        result.update(status="INFRA_ERROR", error_type=type(exc).__name__)
        traceback.print_exc()
        exit_code = 1
    finally:
        platform_result = {**result, "model_status": result["status"]}
        if result["status"] == "MODEL_FAILURE":
            platform_result["status"] = "OK"
        write_json(
            Path(os.environ["ARENA_OUTPUT_DIR"]) / "harness_result.json",
            platform_result,
        )
        write_json(output / "single-turn-result.json", result)
    raise SystemExit(exit_code)


def collect_main() -> None:
    output = Path(os.environ["ARENA_AGENT_OUTPUT_DIR"])
    workspace = Path(os.environ["ARENA_WORKSPACE"])
    if (output / "model-failure.json").is_file() or (
        output / "presentation.json"
    ).is_file():
        return
    source = output / "scene.blend"
    if not source.is_file():
        source = workspace / "scene.blend"
    if not source.is_file():
        write_json(
            output / "model-failure.json",
            {"status": "MODEL_FAILURE", "reason": "No delivered scene.blend"},
        )
        return
    # Keep the agent's exact delivery alongside the viewable packaged scene.
    shutil.copy2(source, output / "scene-native.blend")
    source = output / "scene-native.blend"
    presented = output / "presentation"
    command = [
        os.environ.get("BLENDER_BIN", "/opt/blender/blender"),
        "-b",
        "--factory-startup",
        "--disable-autoexec",
        "--threads",
        "2",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).with_name("scene.py")),
        "--",
        "--blend",
        str(source),
        "--output",
        str(presented),
        "--render",
    ]
    with (output / "presentation.log").open("wb") as log:
        subprocess.run(
            command,
            check=True,
            timeout=600,
            env=single_turn.clean_process_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    for name in ["scene.blend", "preview.png", "presentation.json"]:
        shutil.copy2(presented / name, output / name)


def materialize_references(oracle: Path, root: Path) -> None:
    if oracle.is_dir() and (oracle / "manifest.json").is_file():
        shutil.copytree(oracle, root, dirs_exist_ok=True)
        return
    if oracle.is_dir():
        archives = [
            path
            for path in oracle.iterdir()
            if path.is_file() and zipfile.is_zipfile(path)
        ]
        if len(archives) != 1:
            raise ValueError(
                "Expected exactly one query-reference archive in the oracle mount"
            )
        oracle = archives[0]
    with zipfile.ZipFile(oracle) as archive:
        for name in archive.namelist():
            if not (root / name).resolve().is_relative_to(root.resolve()):
                raise ValueError("Invalid oracle archive member")
        archive.extractall(root)


def renderer_threads() -> int:
    threads = int(os.environ.get("OVERFIT_RENDER_THREADS", "2"))
    if threads < 1:
        raise ValueError("OVERFIT_RENDER_THREADS must be positive")
    return threads


def reward_main() -> None:
    from examples.blender_overfit import verifier

    output = Path(os.environ["ARENA_REWARD_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    destination = Path(os.environ["ARENA_REWARD_OUTPUT_FILE"])
    candidate = Path(os.environ["ARENA_AGENT_OUTPUT_DIR"]) / "scene.blend"
    if not candidate.is_file():
        failure = candidate.with_name("model-failure.json")
        if failure.is_file():
            write_json(
                destination,
                {
                    "status": "OK",
                    "score": 0.0,
                    "failure_kind": "model_failure",
                    "raw": json.loads(failure.read_text()),
                },
            )
            return
        raise FileNotFoundError("Missing candidate without an attributed model failure")
    with tempfile.TemporaryDirectory(prefix="overfit-visual-reward-") as scratch:
        root = Path(scratch)
        materialize_references(Path(os.environ["ARENA_ORACLE_PATH"]), root)
        manifest = json.loads((root / "manifest.json").read_text())
        if len(manifest["tasks"]) != 1:
            raise ValueError("Each reward archive must bind exactly one task")
        evidence = output / "candidate-evidence"
        command = [
            os.environ.get("BLENDER_BIN", "/opt/blender/blender"),
            "-b",
            "--factory-startup",
            "--disable-autoexec",
            "--threads",
            str(renderer_threads()),
            "--python-exit-code",
            "1",
            "--python",
            str(Path(__file__).with_name("render.py")),
            "--",
            "--blend",
            str(candidate),
            "--output",
            str(evidence),
        ]
        render_only = os.environ.get("OVERFIT_RENDER_ONLY") == "1"
        if os.environ.get("OVERFIT_SAVE_DISPLAY_SCENE") == "1":
            command.append("--save-display")
        if render_only:
            configuration = json.loads(Path(os.environ["ARENA_QUERY_PATH"]).read_text())
            if configuration["candidate_sha256"] != digest(candidate.read_bytes()):
                raise ValueError("Rendering-only query binds a different candidate")
            command.extend(
                [
                    "--views",
                    ",".join(str(index) for index in configuration["view_indices"]),
                ]
            )
        with (output / "render.log").open("wb") as log:
            subprocess.run(
                command,
                check=True,
                timeout=int(os.environ.get("OVERFIT_RENDER_TIMEOUT_SECONDS", "3600")),
                env=single_turn.clean_process_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if render_only:
            write_json(
                destination,
                {
                    "status": "OK",
                    "score": 0.0,
                    "diagnostic_only": True,
                    "model_calls": 0,
                    "render_manifest": json.loads(
                        (evidence / "render-manifest.json").read_text()
                    ),
                },
            )
            return
        original_argv = sys.argv
        try:
            sys.argv = [
                "verify",
                "--dataset",
                str(root),
                "--task-id",
                manifest["tasks"][0]["task_id"],
                "--evidence",
                str(evidence),
                "--output",
                str(output / "visual-judgments"),
            ]
            verifier.main()
        finally:
            sys.argv = original_argv
        result = json.loads((output / "visual-judgments/result.json").read_text())
        write_json(destination, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["sft", "collect", "reward"], nargs="?", default="sft"
    )
    command = parser.parse_args().command
    try:
        {"sft": sft_main, "collect": collect_main, "reward": reward_main}[command]()
    except SystemExit:
        raise
    except Exception as exc:
        # Tracebacks contain local file locations; never serialize provider headers or credentials.
        if command == "reward" and os.environ.get("ARENA_REWARD_OUTPUT_FILE"):
            write_json(
                Path(os.environ["ARENA_REWARD_OUTPUT_FILE"]),
                {
                    "status": "EVAL_ERROR",
                    "score": None,
                    "error_type": type(exc).__name__,
                },
            )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
