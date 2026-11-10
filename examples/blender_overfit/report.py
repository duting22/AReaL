"""Assemble paired scores and viewable deliveries without resampling any model."""

import argparse
import csv
import json
import os
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from examples.blender_overfit.prepare import write_json
from examples.blender_overfit.replay import delivered_scene


def budget_reason(record: dict, artifacts: Path) -> str | None:
    receipt_path = artifacts / "native-receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    outcome = receipt.get("raw", {}).get("outcome", {})
    code = record.get("outcome_code") or outcome.get("code")
    if code in {
        "LLM_OUTPUT_TOKEN_LIMIT_EXCEEDED",
        "LLM_CONTEXT_LENGTH_EXCEEDED",
        "MODEL_CONTEXT_WINDOW_EXCEEDED",
        "MODEL_REQUEST_BODY_TOO_LARGE",
        "AGENT_RUN_TIMEOUT",
    }:
        return code
    if outcome.get("details", {}).get("code") == "tool_call_budget_exceeded":
        return "TOOL_CALL_ARGUMENT_BYTES_LIMIT_EXCEEDED"
    for error in receipt.get("error", []) or []:
        if (
            isinstance(error, dict)
            and error.get("message") == "reasoning exceeds 1 MiB"
        ):
            return "REASONING_BYTE_LIMIT_1MIB"
    response_path = artifacts / "model-response.json"
    if response_path.is_file():
        response = json.loads(response_path.read_text())
        if any(
            choice.get("finish_reason") == "length"
            for choice in response.get("choices", [])
        ):
            return "LLM_OUTPUT_TOKEN_LIMIT_EXCEEDED"
    single_path = artifacts / "single-turn-result.json"
    if (
        single_path.is_file()
        and json.loads(single_path.read_text()).get("error_type") == "TimeoutExpired"
    ):
        return "SCENE_EXECUTION_TIME_LIMIT"
    return None


def apply_budget_policy(
    score: float | None, reason: str | None, delivered: bool, terminal: bool
) -> tuple[float | None, bool]:
    excluded = bool(reason and not delivered and terminal)
    return (0.0 if excluded else score), excluded


def paired_summary(rows: list[dict], exclude_budget: bool = False) -> dict:
    pairs = [
        row
        for row in rows
        if row["sft_score"] is not None
        and row["agent_score"] is not None
        and (
            not exclude_budget
            or not (
                row.get("sft_budget_without_scene")
                or row.get("agent_budget_without_scene")
            )
        )
    ]
    return {
        "count": len(pairs),
        "sft_mean": statistics.mean(row["sft_score"] for row in pairs)
        if pairs
        else None,
        "agent_mean": statistics.mean(row["agent_score"] for row in pairs)
        if pairs
        else None,
        "mean_agent_minus_sft": statistics.mean(
            row["agent_score"] - row["sft_score"] for row in pairs
        )
        if pairs
        else None,
        "task_ids": [row.get("task_id") for row in pairs],
    }


def find_visual(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    if value.get("schema") == "blender-overfit-visual/v1":
        return value
    for item in value.values():
        found = find_visual(item)
        if found is not None:
            return found
    return None


def build_report(dataset: Path, run: Path, site_base: str) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text())
    state = json.loads((run / "state.json").read_text())
    selected = {}
    for record in state["attempts"]:
        if record.get("include_in_comparison"):
            selected[record["mode"], record["source_task"]] = record
    rows = []
    all_results = []
    for index, task in enumerate(manifest["tasks"], 1):
        row = {
            "index": index,
            "task_id": task["task_id"],
            "source_row": task["source_row_index"],
        }
        for mode in ("sft", "agent"):
            record = selected.get((mode, task["task_id"]), {})
            status = record.get("status", "NOT_SUBMITTED")
            score = record.get("score")
            source = run / "tasks" / record.get("job_id", "missing")
            evaluation = (
                run
                / "tasks"
                / record.get("evaluation_job_id", record.get("job_id", "missing"))
            )
            target = dataset / "results" / mode / task["task_id"]
            target.mkdir(parents=True, exist_ok=True)
            previous_path = target / "result.json"
            previous = (
                json.loads(previous_path.read_text()) if previous_path.is_file() else {}
            )
            if previous and previous.get("job_id") != record.get("job_id"):
                for name in (
                    "scene.blend",
                    "scene-native.blend",
                    "preview.png",
                    "verification-preview.png",
                    "display/scene.blend",
                    "generated.py",
                    "completion.txt",
                    "blender.log",
                    "presentation.json",
                    "single-turn-result.json",
                    "native-receipt.json",
                    "full.jsonl",
                    "model-failure.json",
                    "history32-input-adaptation.json",
                    "visual-score.json",
                    "trajectory.json",
                ):
                    (target / name).unlink(missing_ok=True)
            if previous and previous.get("evaluation_task_id") != record.get(
                "evaluation_task_id", record.get("task_id")
            ):
                # A regrade must not display evidence from a superseded renderer.
                # The source presentation is copied again below when available.
                for name in (
                    "preview.png",
                    "verification-preview.png",
                    "display/scene.blend",
                ):
                    (target / name).unlink(missing_ok=True)
                if (target / "verification").is_dir():
                    shutil.rmtree(target / "verification")
            visual = None
            for directory in (evaluation / "artifacts",):
                if not directory.is_dir():
                    continue
                for name in ("reward_result.json", "result.json"):
                    path = directory / name
                    if path.is_file():
                        visual = find_visual(json.loads(path.read_text()))
                        if visual is not None:
                            break
                if visual is not None:
                    break
            if (
                visual is None
                and status == "DONE"
                and (source / "result.json").is_file()
            ):
                visual = find_visual(json.loads((source / "result.json").read_text()))
            if visual is not None and status == "DONE":
                score = visual["score"]
                write_json(target / "visual-score.json", visual)
            else:
                (target / "visual-score.json").unlink(missing_ok=True)
            raw_score = score
            limit_reason = budget_reason(record, source / "artifacts")
            receipt_path = source / "artifacts/native-receipt.json"
            native_outcome = (
                json.loads(receipt_path.read_text()).get("raw", {}).get("outcome", {})
                if receipt_path.is_file()
                else {}
            )
            delivery = delivered_scene(source / "artifacts")
            delivered = delivery is not None or bool(
                record.get("delivery_available") or record.get("replay_scene_sha256")
            )
            source_status = record.get("source_status", status)
            terminal = source_status in {
                "DONE",
                "MODEL_FAILURE",
                "HARNESS_FAILED",
                "EVAL_FAILED",
                "FAILED",
                "TIMEOUT",
                "COLLECT_FAILED",
            }
            score, exclude_budget = apply_budget_policy(
                score, limit_reason, delivered, terminal
            )
            if exclude_budget:
                status = "BUDGET_EXCEEDED"
            for name in (
                "scene.blend",
                "scene-native.blend",
                "preview.png",
                "generated.py",
                "completion.txt",
                "blender.log",
                "presentation.json",
                "single-turn-result.json",
                "native-receipt.json",
                "full.jsonl",
                "model-failure.json",
                "history32-input-adaptation.json",
            ):
                path = source / "artifacts" / name
                if path.is_file():
                    shutil.copy2(path, target / name)
            if delivery is not None and delivery.name == "scene-native.blend":
                shutil.copy2(delivery, target / "scene.blend")
            grading_preview = evaluation / "artifacts/view-01.png"
            if grading_preview.is_file():
                shutil.copy2(grading_preview, target / "verification-preview.png")
            display_scene = evaluation / "artifacts/display-scene.blend"
            if display_scene.is_file():
                (target / "display").mkdir(exist_ok=True)
                shutil.copy2(display_scene, target / "display/scene.blend")
            evidence_names = [f"view-{index:02d}.png" for index in range(1, 10)]
            evidence_names += [f"zoom-{index:02d}.png" for index in range(1, 3)]
            evidence_names += [
                "render-manifest.json",
                "qa-vote.json",
                "detail-vote.json",
                "qa-response.json",
                "detail-response.json",
            ]
            for name in evidence_names:
                path = evaluation / "artifacts" / name
                if path.is_file():
                    (target / "verification").mkdir(exist_ok=True)
                    shutil.copy2(path, target / "verification" / name)
            preview_kind = "colored_presentation"
            if not (source / "artifacts/preview.png").is_file():
                evidence_preview = evaluation / "artifacts/view-01.png"
                if evidence_preview.is_file():
                    shutil.copy2(evidence_preview, target / "preview.png")
                    preview_kind = "verifier_neutral"
            result = {
                "mode": mode,
                "task_id": task["task_id"],
                "status": status,
                "score": score,
                "raw_score": raw_score,
                "budget_reason": limit_reason,
                "budget_without_scene": exclude_budget,
                "score_excluding_budget": None if exclude_budget else score,
                "failure_kind": record.get("failure_kind"),
                "job_id": record.get("job_id"),
                "job_url": site_base.rstrip("/") + "/jobs/" + record["job_id"]
                if record.get("job_id")
                else None,
                "evaluation_task_id": record.get(
                    "evaluation_task_id", record.get("task_id")
                ),
                "generation_release": record.get("version"),
                "evaluation_reward_version": record.get(
                    "evaluation_reward_version", record.get("version")
                ),
                "generation_status": record.get(
                    "source_status", record.get("platform_status", status)
                ),
                "generation_outcome_code": record.get("outcome_code")
                or native_outcome.get("code"),
                "evaluation_job_url": site_base.rstrip("/")
                + "/jobs/"
                + record["evaluation_job_id"]
                if record.get("evaluation_job_id")
                else None,
                "components": visual.get("components") if visual else None,
                "scene_available": (target / "scene.blend").is_file(),
                "scene_source_artifact": delivery.name if delivery else None,
                "preview_kind": preview_kind,
                "preview_available": (target / "preview.png").is_file(),
                "display_scene_available": (target / "display/scene.blend").is_file(),
            }
            if (target / "model-failure.json").is_file():
                result["failure_kind"] = "model_failure"
                if score == 0 and not delivered and not exclude_budget:
                    status = result["status"] = "MODEL_FAILURE"
            if (target / "single-turn-result.json").is_file():
                result["execution_error_type"] = json.loads(
                    (target / "single-turn-result.json").read_text()
                ).get("error_type")
            trajectory = source / "trajectory.json"
            if trajectory.is_file():
                shutil.copy2(trajectory, target / "trajectory.json")
                events = json.loads(trajectory.read_text()).get("events", [])
                result["tool_calls"] = sum(
                    block.get("type") == "tool_use"
                    for event in events
                    for block in event.get("message", {}).get("content", [])
                    if isinstance(block, dict)
                )
            all_results.append(result)
            row.update(
                {
                    mode + "_status": status,
                    mode + "_score": score,
                    mode + "_score_excluding_budget": None if exclude_budget else score,
                    mode + "_budget_without_scene": exclude_budget,
                    mode + "_budget_reason": limit_reason,
                    mode + "_scene": result["scene_available"],
                    mode + "_job": record.get("job_id"),
                }
            )
            for component in ("quality", "detail", "aesthetics"):
                row[mode + "_" + component] = (
                    (result["components"] or {}).get(component, {}).get("value")
                )
        row["agent_minus_sft"] = (
            row["agent_score"] - row["sft_score"]
            if row["agent_score"] is not None and row["sft_score"] is not None
            else None
        )
        if row["sft_budget_without_scene"] or row["agent_budget_without_scene"]:
            row["sft_score_excluding_budget"] = None
            row["agent_score_excluding_budget"] = None
        rows.append(row)
    summary = {
        "model": state["model"],
        "checkpoint": manifest["checkpoint"],
        "task_count": len(rows),
        "formula": "0.4Q + 0.4D + 0.2A",
        "budget_policy": "Score delivered scenes; otherwise count explicit budget exhaustion as zero and also report paired exclusion. Ordinary failures score zero and do not independently trigger paired exclusion.",
        "modes": {},
    }
    deployed_path = run / "validation/deployed-model.json"
    if deployed_path.is_file():
        summary["deployed_checkpoint"] = json.loads(deployed_path.read_text()).get(
            "model_path"
        )
    for mode in ("sft", "agent"):
        items = [item for item in all_results if item["mode"] == mode]
        scores = [item["score"] for item in items if item["score"] is not None]
        summary["modes"][mode] = {
            "statuses": dict(Counter(item["status"] for item in items)),
            "scored": len(scores),
            "mean": statistics.mean(scores) if scores else None,
            "scenes": sum(item["scene_available"] for item in items),
        }
    summary["paired"] = paired_summary(rows)
    summary["paired_excluding_budget"] = paired_summary(rows, exclude_budget=True)
    summary["budget_excluded_task_ids"] = [
        row["task_id"]
        for row in rows
        if row["sft_budget_without_scene"] or row["agent_budget_without_scene"]
    ]
    for result in all_results:
        excluded = result["task_id"] in summary["budget_excluded_task_ids"]
        result["paired_excluded_for_budget"] = excluded
        result["score_excluding_budget"] = None if excluded else result["score"]
        write_json(
            dataset / "results" / result["mode"] / result["task_id"] / "result.json",
            result,
        )
    write_json(run / "comparison.json", {"summary": summary, "rows": rows})
    with (run / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        dataset / "comparison-summary.json",
        {
            "summary": summary,
            "json_path": os.path.relpath(run / "comparison.json", dataset),
            "csv_path": os.path.relpath(run / "comparison.csv", dataset),
        },
    )
    from examples.blender_overfit.gallery import build_gallery

    build_gallery(dataset)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--site-base", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_report(args.dataset, args.run_dir, args.site_base),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
