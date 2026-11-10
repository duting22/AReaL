# SPDX-License-Identifier: Apache-2.0
"""Recover Qwen XML-tool-call rollouts as audited OpenAI conversations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any

HEADER = re.compile(r"<\|im_start\|>(system|user|assistant)\n")
CALL = re.compile(
    r"<tool_call>\s*<function=([^>]+)>\n?(.*?)</function>\s*</tool_call>", re.S
)
PARAM = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.S)
# SGLang Qwen3CoderDetector's non-streaming fallback accepts a function or
# parameter ending at EOF. Preserve the raw block and keep it review-only.
PARTIAL_FUNCTION = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.S)
PARTIAL_PARAMETER = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.S
)
RESPONSE = re.compile(r"<tool_response>\n?(.*?)\n?</tool_response>", re.S)
TOOLS_PREFIX = "# Tools\n\nYou have access to the following functions:\n\n<tools>"


def compile_template(text: str) -> Any:
    """Use Transformers-compatible Jinja rendering without loading model libraries."""
    from jinja2 import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise TemplateError(message)

    def tojson(value: Any, **kwargs: Any) -> str:
        return json.dumps(value, **{"ensure_ascii": False, **kwargs})

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(text)


def parse_transcript(text: str) -> dict[str, Any]:
    """Parse role frames, recover tools and reasoning, and flag ambiguous repairs."""
    messages: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    pending: deque[str] = deque()
    effort = "medium"

    def flag(code: str, severity: str = "error", **details: Any) -> None:
        anomalies.append({"code": code, "severity": severity, **details})

    def flush() -> None:
        if pending:
            flag("tool_calls_without_responses", call_ids=list(pending))
            pending.clear()

    headers = list(HEADER.finditer(text))
    if not headers or headers[0].start() != 0:
        raise ValueError("Transcript does not begin with a supported Qwen role header")
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        body = text[header.end() : end]
        if body.endswith("<|im_end|>\n"):
            body = body[:-11]
        elif body.endswith("<|im_end|>"):
            body = body[:-10]
        else:
            flag("unterminated_message", message_index=index)
        if "<|im_start|>" in body or "<|im_end|>" in body:
            flag("embedded_role_token", message_index=index)
        role = header[1]
        if role == "system":
            if index:
                flag("noninitial_system_message")
            if body.startswith("Reasoning effort is set to "):
                effort = body.split("Reasoning effort is set to ", 1)[1].split(".", 1)[
                    0
                ]
                body = body.split("\n\n", 1)[1] if "\n\n" in body else ""
            if body.startswith(TOOLS_PREFIX):
                tail = body[len(TOOLS_PREFIX) :]
                raw, tail = tail.split("\n</tools>", 1)
                tools = [json.loads(line) for line in raw.strip().splitlines()]
                if "</IMPORTANT>" not in tail:
                    raise ValueError("Unsupported Qwen tools instructions")
                body = tail.split("</IMPORTANT>", 1)[1].removeprefix("\n\n")
            if body:
                messages.append({"role": "system", "content": body})
        elif role == "user":
            responses = (
                list(RESPONSE.finditer(body))
                if body.startswith("<tool_response>")
                else []
            )
            if responses and not RESPONSE.sub("", body).strip():
                for response in responses:
                    if pending:
                        call_id = pending.popleft()
                    else:
                        call_id = f"orphan_{index}_{response.start()}"
                        flag("tool_responses_without_calls", message_index=index)
                    messages.append(
                        {
                            "role": "tool",
                            "content": response[1],
                            "tool_call_id": call_id,
                        }
                    )
            else:
                if body.startswith("<tool_response>"):
                    flag("malformed_tool_response", message_index=index)
                messages.append({"role": "user", "content": body})
        else:
            flush()
            msg: dict[str, Any] = {"role": "assistant", "content": ""}
            if body.startswith("<think>\n"):
                thinking = body[len("<think>\n") :]
                if "</think>" in thinking:
                    reasoning, body = thinking.split("</think>", 1)
                    msg["reasoning_content"] = reasoning.removesuffix("\n")
                    body = body.removeprefix("\n\n")
                elif "<tool_call>" in thinking:
                    # Qwen3Detector in the experiment's SGLang runtime uses
                    # <tool_call> as an implicit reasoning terminator.
                    reasoning, remainder = thinking.split("<tool_call>", 1)
                    flag(
                        "reasoning_terminated_by_tool_call",
                        severity="warning",
                        message_index=index,
                        source_assistant_text=body,
                    )
                    msg["reasoning_content"] = reasoning.rstrip("\n")
                    body = "<tool_call>" + remainder
                else:
                    flag(
                        "unterminated_reasoning",
                        message_index=index,
                        source_assistant_text=body,
                    )
                    # Keep unfinished thinking in its actual channel. Rendering
                    # inserts a closing tag, so this remains a review-only row.
                    msg["reasoning_content"] = thinking
                    messages.append(msg)
                    continue
            calls = list(CALL.finditer(body))
            remaining = CALL.sub("", body)
            partial = False
            if not calls and "<tool_call>" in body and "<function=" in body:
                recovered = []
                for first, second in PARTIAL_FUNCTION.findall(body):
                    function = first or second
                    if ">" in function:
                        name, raw = function.split(">", 1)
                        recovered.append((body, name, raw))
                if recovered:
                    calls = recovered
                    remaining = body[: body.index("<tool_call>")]
                    partial = True
                    flag(
                        "incomplete_tool_call_recovered",
                        message_index=index,
                        source_assistant_text=body,
                    )
            if "<tool_call>" in remaining or "</tool_call>" in remaining:
                flag("malformed_tool_call", message_index=index)
            msg["content"] = remaining.strip()
            schemas = {
                t["function"]["name"]: t["function"].get("parameters", {})
                for t in tools
            }
            for call_index, call in enumerate(calls):
                name, raw = call[1], call[2]
                schema = schemas.get(name, {})
                if name not in schemas:
                    flag("unknown_tool", tool=name, message_index=index)
                args: dict[str, Any] = {}
                unparsed = "" if partial else PARAM.sub("", raw).strip()
                if unparsed:
                    redundant_close = bool(
                        re.fullmatch(r"(?:</function>\s*)+", unparsed)
                    )
                    flag(
                        "redundant_function_close"
                        if redundant_close
                        else "malformed_tool_arguments",
                        severity="warning" if redundant_close else "error",
                        tool=name,
                        message_index=index,
                        source_tool_call=call[0],
                        unparsed_text=unparsed,
                    )
                if partial:
                    parameters = [
                        fragment.split(">", 1)
                        for fragment in PARTIAL_PARAMETER.findall(raw)
                        if ">" in fragment
                    ]
                    parameters = [
                        (key, value.removeprefix("\n").removesuffix("\n"))
                        for key, value in parameters
                    ]
                else:
                    parameters = [(p[1], p[2]) for p in PARAM.finditer(raw)]
                for key, value in parameters:
                    if key in args:
                        flag(
                            "duplicate_tool_argument",
                            tool=name,
                            argument=key,
                            message_index=index,
                        )
                    spec = schema.get("properties", {}).get(key, {})
                    kind = spec.get("type")
                    # The rollout proxy preserves strings when there is no
                    # argument schema; guessing JSON here changes execution
                    # semantics for unknown tools and undeclared arguments.
                    if kind is None:
                        flag(
                            "argument_schema_missing",
                            severity="warning",
                            tool=name,
                            argument=key,
                            message_index=index,
                        )
                    elif kind != "string":
                        try:
                            value = json.loads(value)
                        except ValueError:
                            flag(
                                "invalid_tool_argument_json",
                                tool=name,
                                argument=key,
                                message_index=index,
                            )
                    expected = {
                        "string": str,
                        "integer": int,
                        "number": (int, float),
                        "boolean": bool,
                        "object": dict,
                        "array": list,
                        "null": type(None),
                    }
                    kinds = kind if isinstance(kind, list) else [kind]
                    if kind and not any(
                        k in expected
                        and isinstance(value, expected[k])
                        and not (k in ("integer", "number") and isinstance(value, bool))
                        for k in kinds
                    ):
                        flag(
                            "tool_argument_type_mismatch",
                            tool=name,
                            argument=key,
                            message_index=index,
                        )
                    args[key] = value
                missing = set(schema.get("required", [])) - args.keys()
                if missing:
                    flag(
                        "missing_required_arguments",
                        tool=name,
                        arguments=sorted(missing),
                        message_index=index,
                    )
                call_id = f"call_{index}_{call_index}"
                pending.append(call_id)
                msg.setdefault("tool_calls", []).append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
            messages.append(msg)
    if pending and messages and messages[-1]["role"] == "assistant":
        # SFT can end at an assistant tool call. There is no later assistant
        # turn depending on an absent response, unlike an intermediate gap.
        flag(
            "terminal_tool_calls_without_responses",
            severity="info",
            call_ids=list(pending),
        )
        pending.clear()
    flush()
    if any(
        token in text
        for token in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>")
    ):
        flag("unrecoverable_vision_payload")
    return {
        "messages": messages,
        "tools": tools,
        "anomalies": anomalies,
        "reasoning_effort": effort,
    }


def render(parsed: dict[str, Any], template: Any) -> str:
    """Render using the actual experiment model template, without tokenization."""
    messages = copy.deepcopy(parsed["messages"])
    for message in messages:
        for call in message.get("tool_calls", []):
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return template.render(
        messages=messages,
        tools=parsed["tools"],
        add_generation_prompt=False,
        preserve_thinking=True,
        enable_thinking=True,
        reasoning_effort=parsed["reasoning_effort"],
    )


def semantic_messages(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Ignore synthetic IDs and template-trimmed boundary whitespace."""
    result = copy.deepcopy(parsed["messages"])
    for message in result:
        message.pop("tool_call_id", None)
        for key in ("content", "reasoning_content"):
            if key in message:
                message[key] = message[key].strip()
        for call in message.get("tool_calls", []):
            call.pop("id", None)
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return result


def convert_record(row: dict[str, Any], template: Any, model: str) -> dict[str, Any]:
    """Validate the dump and preserve metadata with an explicit conversion audit."""
    text = row["prompt"] + row["completion"]
    if "segments" in row and "".join(s["text"] for s in row["segments"]) != text:
        raise ValueError("segments differ from prompt + completion")
    parsed = parse_transcript(text)
    anomalies = parsed["anomalies"]
    rendered = render(parsed, template)
    reparsed = parse_transcript(rendered)
    if (
        semantic_messages(parsed) != semantic_messages(reparsed)
        or parsed["tools"] != reparsed["tools"]
    ):
        anomalies.append({"code": "semantic_roundtrip_mismatch", "severity": "error"})
    exact = rendered.rstrip("\n") == text.rstrip("\n")
    output = {
        k: v for k, v in row.items() if k not in {"prompt", "completion", "segments"}
    }
    output.update(
        {
            "instance_id": str(
                row.get("instance_id", row.get("arena_task_id", row["task_id"]))
            ),
            "conversations": [
                {
                    "messages": parsed["messages"],
                    "tools": parsed["tools"],
                    "model": model,
                }
            ],
            "conversion_status": (
                "anomalous"
                if any(a["severity"] == "error" for a in anomalies)
                else "clean"
            ),
            "conversion_tags": sorted({a["code"] for a in anomalies}),
            "conversion_anomalies": anomalies,
            "conversion_roundtrip_exact": exact,
            "conversion_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    template_text = (args.model_path / "chat_template.jinja").read_text()
    for fragment in (
        "<|im_start|>",
        "<function=",
        "<parameter=",
        TOOLS_PREFIX.replace("\n", "\\n"),
    ):
        if fragment not in template_text:
            raise ValueError(f"Unsupported template: missing {fragment!r}")
    template = compile_template(template_text)
    files = sorted(args.input.glob("*.jsonl")) if args.input.is_dir() else [args.input]
    if not files:
        raise ValueError("No input JSONL files")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "input": str(args.input),
        "model_path": str(args.model_path),
        "template_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
        "files": len(files),
        "records": 0,
        "clean": 0,
        "anomalous": 0,
        "failed": 0,
        "roundtrip_exact": 0,
        "anomaly_record_counts": Counter(),
        "failures": [],
    }
    with (
        (args.output_dir / "all.jsonl").open("w") as all_out,
        (args.output_dir / "clean.jsonl").open("w") as clean_out,
        (args.output_dir / "anomalous.jsonl").open("w") as bad_out,
    ):
        for path in files:
            for line_number, line in enumerate(path.open(), 1):
                report["records"] += 1
                try:
                    row = json.loads(line)
                    converted = convert_record(row, template, args.model_path.name)
                    converted["conversion_source"] = {
                        "path": str(path),
                        "line": line_number,
                    }
                    serialized = json.dumps(converted, ensure_ascii=False) + "\n"
                    all_out.write(serialized)
                    is_clean = converted["conversion_status"] == "clean"
                    (clean_out if is_clean else bad_out).write(serialized)
                    report["clean" if is_clean else "anomalous"] += 1
                    report["roundtrip_exact"] += converted["conversion_roundtrip_exact"]
                    report["anomaly_record_counts"].update(converted["conversion_tags"])
                except Exception as exc:
                    report["failed"] += 1
                    report["failures"].append(
                        {"path": str(path), "line": line_number, "error": str(exc)}
                    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if report["failed"]:
        raise SystemExit(f"{report['failed']} records failed; inspect report.json")


if __name__ == "__main__":
    main()
