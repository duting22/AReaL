"""Audit generation failures from retained tool evidence without rerunning candidates."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from examples.blender_overfit.prepare import write_json
from examples.blender_overfit.replay import delivered_scene
from examples.blender_overfit.report import budget_reason


def decode_output(value: Any, depth: int = 0) -> list[dict]:
    """Unwrap native tool results, including JSON escaped inside inputText blocks."""
    if depth > 8:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            if value.startswith(('{\\"', '[\\"')):
                try:
                    return decode_output(json.loads('"' + value + '"'), depth + 1)
                except ValueError:
                    pass
            return []
        return decode_output(parsed, depth + 1)
    if isinstance(value, list):
        return [row for item in value for row in decode_output(item, depth + 1)]
    if isinstance(value, dict):
        if "stdout" in value or "stderr" in value or "error" in value:
            return [value]
        return decode_output(value.get("text", value.get("content")), depth + 1)
    return []


def analyze_trajectory(events: list[dict], receipt: dict, delivery_dir: str) -> dict:
    blocks = [
        block
        for event in events
        for block in event.get("message", {}).get("content", [])
        if isinstance(block, dict)
    ]
    calls = [block for block in blocks if block.get("type") == "tool_use"]
    results = {
        block["tool_use_id"]: block
        for block in blocks
        if block.get("type") == "tool_result"
    }
    reasoning = []
    for event in events:
        item = event.get("item", {})
        if event.get("type") == "core_item" and item.get("type") == "reasoning":
            for part in item.get("content", []):
                reasoning.append(
                    part if isinstance(part, str) else part.get("text", "")
                )
    script_paths = []
    scripts_with_save = []
    commands = []
    saves = []
    reads = []
    errors = []
    listings = []
    tool_errors = []
    killed = []
    for call in calls:
        args = call.get("input", {})
        source = args.get("text", "")
        path = args.get("path", "")
        if call.get("name") in {"fs_create", "fs_write"} and "import bpy" in source:
            script_paths.append(path)
            if re.search(r"bpy\.ops\.wm\.save_(?:as_)?mainfile", source):
                scripts_with_save.append(path)
        outputs = decode_output(results.get(call.get("id"), {}).get("content"))
        for output in outputs:
            if output.get("error"):
                tool_errors.append(output["error"])
        if call.get("name") != "run_command":
            continue
        command = args.get("command", "")
        commands.append(command)
        for output in outputs:
            if output.get("exitCode") in (137, 139):
                killed.append(
                    {"exit_code": output["exitCode"], "command": command[:800]}
                )
            for line in (
                output.get("stdout", "") + "\n" + output.get("stderr", "")
            ).splitlines():
                match = re.search(
                    r'(?i)\bsaved(?: as)?\s*:?\s+["\']?(/[^"\'\n]+\.blend)', line
                )
                if match:
                    saves.append({"path": match[1], "evidence": line[:800]})
                if "Read blend:" in line:
                    reads.append(line[:800])
                if re.match(r"[A-Za-z]+(?:Error|Exception):", line):
                    errors.append(line[:800])
                if re.match(r"^-[-rwx]+\s+.*\.blend$", line):
                    listings.append(line[:800])
    build_runs = [
        command
        for command in commands
        if "blender" in command
        and re.search(r"(?:--python(?:\s|=)|(?:^|\s)-P(?:\s|/))", command)
        and any(
            path and (path in command or PurePosixPath(path).name in command)
            for path in script_paths
        )
    ]
    fragments = Counter(
        part.strip()
        for text in reasoning
        for part in re.split(r"[;\n]", text)
        if len(part.strip()) >= 16
    )
    expected = PurePosixPath(delivery_dir) / "scene.blend"
    outside = [save for save in saves if PurePosixPath(save["path"]) != expected]
    if any(
        "authorization directory identity changed" in str(error)
        for error in tool_errors
    ):
        stage = "tool_scope_invalid_after_execution"
    elif outside:
        stage = "saved_outside_delivery_directory"
    elif saves:
        stage = "saved_in_delivery_directory_check_collection"
    elif not script_paths:
        stage = "no_completed_blender_script_write"
    elif not build_runs:
        stage = "blender_script_written_not_executed"
    else:
        stage = "blender_executed_without_confirmed_scene_save"
    usage = receipt.get("usage", {})
    return {
        "tool_calls": len(calls),
        "tool_names": dict(Counter(call.get("name") for call in calls)),
        "input_tokens_completed_or_reported": usage.get("inputTokens"),
        "output_tokens_completed_or_reported": usage.get("outputTokens"),
        "duration_seconds": receipt.get("duration_ms", 0) / 1000,
        "completed_blender_script_paths": sorted(set(script_paths)),
        "initial_script_writes_containing_save": sorted(set(scripts_with_save)),
        "blender_script_command_count": len(build_runs),
        "confirmed_save_logs": saves,
        "confirmed_blend_read_logs": reads,
        "blend_directory_listings": listings,
        "save_logs_outside_delivery": outside,
        "observed_errors": errors,
        "tool_errors": tool_errors,
        "killed_or_crashed_processes": killed,
        "retained_reasoning_items": len(reasoning),
        "largest_retained_reasoning_bytes": max(
            (len(t.encode()) for t in reasoning), default=0
        ),
        "repeated_reasoning_fragments": [
            {"text": text[:250], "count": count}
            for text, count in fragments.most_common(5)
            if count > 1
        ],
        "last_retained_reasoning_tail": reasoning[-1][-1200:] if reasoning else "",
        "observed_stage": stage,
        "evidence_limit": "Usage is aggregate, and a failed response may be incompletely retained; do not classify every length stop as thinking exhaustion.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--delivery-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads((args.run / "state.json").read_text())
    rows = []
    for record in state["attempts"]:
        if not record.get("include_in_comparison") or record["mode"] != "agent":
            continue
        source = args.run / "tasks" / record["job_id"]
        trace_path = source / "trajectory.json"
        receipt_path = source / "artifacts/native-receipt.json"
        if not trace_path.is_file() or not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text())
        trace = json.loads(trace_path.read_text())
        reason = budget_reason(record, source / "artifacts")
        delivered = delivered_scene(source / "artifacts") is not None
        rows.append(
            {
                "task_id": record["source_task"],
                "job_id": record["job_id"],
                "budget_reason": reason,
                "delivered": delivered,
                "budget_without_delivery": bool(reason and not delivered),
                **analyze_trajectory(trace["events"], receipt, args.delivery_dir),
            }
        )
    write_json(args.output, {"model": state["model"], "cases": rows})
    print(
        json.dumps(
            {
                "cases": len(rows),
                "budget_without_delivery": sum(
                    r["budget_without_delivery"] for r in rows
                ),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
