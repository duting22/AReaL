# SPDX-License-Identifier: Apache-2.0

"""Batch-convert Ling3 rollout JSONL files with deterministic multiprocessing."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from examples.prefix_replay.ling3_rollout_converter.convert import (
    convert_rollout_record,
    validate_ling3_chat_template,
)

_RECORD_INDEX_STRIDE = 1_000_000_000


@dataclass(frozen=True)
class _FileTask:
    sequence: int
    input_path: Path
    chunk_path: Path
    model_name: str
    max_tokens: int | None
    instance_id: str | None
    validate_roundtrip: bool
    skip_errors: bool


@dataclass(frozen=True)
class _FileResult:
    sequence: int
    input_path: str
    chunk_path: str
    records: int
    errors: list[dict[str, Any]]
    anomalous_records: list[dict[str, Any]]


def _natural_path_key(path: Path) -> tuple[tuple[tuple[int, int | str], ...], ...]:
    """Sort numeric path components naturally (for example, 2 before 10)."""
    return tuple(
        tuple(
            (0, int(piece)) if piece.isdigit() else (1, piece.casefold())
            for piece in re.split(r"(\d+)", part)
            if piece
        )
        for part in path.resolve().parts
    )


def discover_jsonl_inputs(
    inputs: Sequence[Path], *, output_path: Path | None = None
) -> list[Path]:
    """Expand files and directories into a deduplicated, naturally sorted file list."""
    if not inputs:
        raise ValueError("At least one input path is required")
    excluded = output_path.resolve() if output_path is not None else None
    discovered: dict[Path, None] = {}
    for input_path in inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        candidates = (
            input_path.rglob("*.jsonl") if input_path.is_dir() else [input_path]
        )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if candidate.suffix.casefold() != ".jsonl":
                raise ValueError(f"Input file must have a .jsonl suffix: {candidate}")
            resolved = candidate.resolve()
            if resolved != excluded:
                discovered[resolved] = None
    files = sorted(discovered, key=_natural_path_key)
    if not files:
        raise ValueError("No input JSONL files were found")
    return files


def _available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _resolve_worker_count(workers: int, file_count: int) -> int:
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise ValueError("workers must be a non-negative integer")
    requested = min(16, _available_cpu_count()) if workers == 0 else workers
    return max(1, min(requested, file_count))


def _process_file(task: _FileTask) -> _FileResult:
    records = 0
    errors: list[dict[str, Any]] = []
    anomalous_records: list[dict[str, Any]] = []
    with (
        task.input_path.open(encoding="utf-8") as input_file,
        task.chunk_path.open("w", encoding="utf-8") as chunk_file,
    ):
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                converted = convert_rollout_record(
                    row,
                    record_index=(task.sequence * _RECORD_INDEX_STRIDE) + line_number,
                    model_name=task.model_name,
                    max_tokens=task.max_tokens,
                    instance_id=task.instance_id,
                    validate_roundtrip=task.validate_roundtrip,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                error = {
                    "input": str(task.input_path),
                    "line": line_number,
                    "error": str(exc),
                }
                if not task.skip_errors:
                    raise ValueError(f"{task.input_path}:{line_number}: {exc}") from exc
                errors.append(error)
                continue
            chunk_file.write(json.dumps(converted, ensure_ascii=False) + "\n")
            if converted["conversion_anomalies"]:
                anomalous_records.append(
                    {
                        "input": str(task.input_path),
                        "line": line_number,
                        "instance_id": converted["instance_id"],
                        "tags": converted["conversion_tags"],
                        "anomalies": converted["conversion_anomalies"],
                    }
                )
            records += 1

    if records == 0 and not errors:
        error = {
            "input": str(task.input_path),
            "line": 0,
            "error": "input dataset is empty",
        }
        if not task.skip_errors:
            raise ValueError(f"Input dataset is empty: {task.input_path}")
        errors.append(error)
    return _FileResult(
        sequence=task.sequence,
        input_path=str(task.input_path),
        chunk_path=str(task.chunk_path),
        records=records,
        errors=errors,
        anomalous_records=anomalous_records,
    )


def _append_result(
    result: _FileResult,
    output_file: BinaryIO,
    errors: list[dict[str, Any]],
    anomalous_records: list[dict[str, Any]],
) -> int:
    with Path(result.chunk_path).open("rb") as chunk_file:
        shutil.copyfileobj(chunk_file, output_file)
    errors.extend(result.errors)
    anomalous_records.extend(result.anomalous_records)
    return result.records


def _run_parallel(
    tasks: Sequence[_FileTask],
    output_file: BinaryIO,
    errors: list[dict[str, Any]],
    anomalous_records: list[dict[str, Any]],
    *,
    workers: int,
) -> int:
    records = 0
    next_submit = 0
    next_write = 0
    window = max(workers * 2, 1)
    buffered: dict[int, _FileResult] = {}
    pending: dict[Future[_FileResult], int] = {}

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        while next_submit < len(tasks) and len(pending) < window:
            task = tasks[next_submit]
            pending[executor.submit(_process_file, task)] = task.sequence
            next_submit += 1

        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                sequence = pending.pop(future)
                buffered[sequence] = future.result()

            while next_write in buffered:
                records += _append_result(
                    buffered.pop(next_write),
                    output_file,
                    errors,
                    anomalous_records,
                )
                next_write += 1

            while next_submit < len(tasks) and len(pending) + len(buffered) < window:
                task = tasks[next_submit]
                pending[executor.submit(_process_file, task)] = task.sequence
                next_submit += 1

    return records


def convert_jsonl_batch(
    inputs: Sequence[Path],
    output_path: Path,
    *,
    model_path: Path,
    model_name: str | None = None,
    max_tokens: int | None = None,
    instance_id: str | None = None,
    workers: int = 0,
    validate_roundtrip: bool = True,
    skip_errors: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert multiple rollout files into one ordered OpenAI-format JSONL."""
    template_path = validate_ling3_chat_template(model_path)
    input_files = discover_jsonl_inputs(inputs, output_path=output_path)
    if instance_id is not None and len(input_files) != 1:
        raise ValueError("instance_id can only be overridden for one input file")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass overwrite=True to replace it"
        )
    worker_count = _resolve_worker_count(workers, len(input_files))
    resolved_model_name = model_name or template_path.parent.name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    errors: list[dict[str, Any]] = []
    anomalous_records: list[dict[str, Any]] = []
    records = 0
    try:
        with (
            tempfile.TemporaryDirectory(
                dir=output_path.parent,
                prefix=f".{output_path.name}.chunks.",
            ) as chunk_directory,
            os.fdopen(output_descriptor, "wb") as output_file,
        ):
            chunk_root = Path(chunk_directory)
            tasks = [
                _FileTask(
                    sequence=sequence,
                    input_path=input_path,
                    chunk_path=chunk_root / f"{sequence:08d}.jsonl",
                    model_name=resolved_model_name,
                    max_tokens=max_tokens,
                    instance_id=instance_id,
                    validate_roundtrip=validate_roundtrip,
                    skip_errors=skip_errors,
                )
                for sequence, input_path in enumerate(input_files)
            ]
            if worker_count == 1:
                for task in tasks:
                    records += _append_result(
                        _process_file(task),
                        output_file,
                        errors,
                        anomalous_records,
                    )
            else:
                records = _run_parallel(
                    tasks,
                    output_file,
                    errors,
                    anomalous_records,
                    workers=worker_count,
                )
        if records == 0:
            raise ValueError("No records were converted")
        os.replace(temporary_name, output_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    anomaly_record_counts = Counter(
        tag for record in anomalous_records for tag in record["tags"]
    )
    anomaly_counts = Counter(
        anomaly["code"]
        for record in anomalous_records
        for anomaly in record["anomalies"]
    )
    return {
        "inputs": [str(path) for path in input_files],
        "output": str(output_path),
        "files": len(input_files),
        "records": records,
        "workers": worker_count,
        "model": resolved_model_name,
        "chat_template": str(template_path),
        "roundtrip_validated": validate_roundtrip,
        "records_with_anomalies": len(anomalous_records),
        "anomaly_counts": dict(sorted(anomaly_counts.items())),
        "anomaly_record_counts": dict(sorted(anomaly_record_counts.items())),
        "anomalous_records": anomalous_records,
        "skipped_records": len(errors),
        "errors": errors,
    }


def write_batch_report(
    report: dict[str, Any], report_path: Path, *, overwrite: bool = False
) -> None:
    """Atomically write a JSON batch report."""
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            f"Report already exists: {report_path}; pass overwrite=True to replace it"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=report_path.parent,
        prefix=f".{report_path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, ensure_ascii=False, indent=2)
            report_file.write("\n")
        os.replace(temporary_name, report_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Input JSONL or directory; repeat for multiple roots.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Merged JSONL.")
    parser.add_argument("--report", type=Path, help="Optional JSON conversion report.")
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Ling3 model directory, or its chat_template.jinja file.",
    )
    parser.add_argument(
        "--model-name",
        help="Value stored in conversations[].model (default: model directory name).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Optional conversations[].max_tokens generation setting.",
    )
    parser.add_argument(
        "--instance-id",
        help="Override instance_id; valid only when one input file is selected.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes (default: auto, capped at 16; use 1 for serial).",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip invalid records and include them in the optional report.",
    )
    parser.add_argument(
        "--no-roundtrip-validation",
        action="store_true",
        help="Disable semantic chat-template round-trip validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output and report files atomically.",
    )
    args = parser.parse_args()
    if args.report is not None and args.report.resolve() == args.output.resolve():
        parser.error("--report must differ from --output")
    if args.report is not None and args.report.exists() and not args.overwrite:
        raise FileExistsError(
            f"Report already exists: {args.report}; pass --overwrite to replace it"
        )

    report = convert_jsonl_batch(
        args.input,
        args.output,
        model_path=args.model_path,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        instance_id=args.instance_id,
        workers=args.workers,
        validate_roundtrip=not args.no_roundtrip_validation,
        skip_errors=args.skip_errors,
        overwrite=args.overwrite,
    )
    if args.report is not None:
        write_batch_report(report, args.report, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
