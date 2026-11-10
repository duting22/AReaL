# SPDX-License-Identifier: Apache-2.0

"""Reverse Bailing/Ling3 chat-template rollouts into OpenAI JSONL records."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tempfile
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROLE_END = "<|role_end|>"
_ROLE_RE = re.compile(r"<role>(SYSTEM|HUMAN|ASSISTANT|OBSERVATION)</role>")
_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_TOOL_RESPONSE_START = "\n<tool_response>\n"
_TOOL_RESPONSE_END = "\n</tool_response>"
_THINK_START = "\n<think>"
_THINK_END = "</think>"

_TOOLS_PREAMBLE = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>"
)
_TOOLS_POSTAMBLE = (
    "\n</tools>\n\n"
    "If none of the functions can be used, point it out. If the given question "
    "lacks the parameters required by the function, also point it out.\n"
    "If you need to use a function, for each function call, output the function "
    "name and arguments within the following XML format:\n"
    "<tool_call>{function-name}\n"
    "<arg_key>{arg-key-1}</arg_key>\n"
    "<arg_value>{arg-value-1}</arg_value>\n"
    "<arg_key>{arg-key-2}</arg_key>\n"
    "<arg_value>{arg-value-2}</arg_value>\n"
    "...\n"
    "</tool_call>\n"
)
_THINKING_RE = re.compile(r"detailed thinking (on|off)$")
_TOOL_DETAIL_RE = re.compile(
    r"^<tool_call>\s*(.*?)(?:(?:\\n|\n)\s*|(?=<arg_key>)|(?=</tool_call>))"
    r"(<arg_key>.*?)?</tool_call>$",
    re.DOTALL,
)
_TOOL_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)
_ROLLOUT_PAYLOAD_FIELDS = {
    "prompt",
    "completion",
    "segments",
    "instance_id",
    "conversations",
    "conversion_anomalies",
    "conversion_status",
    "conversion_tags",
}


@dataclass(frozen=True)
class ParsedTranscript:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    thinking_option: str
    anomalies: list[dict[str, Any]]


def _anomaly(
    code: str,
    severity: str,
    message: str,
    **context: Any,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        **context,
    }


def _template_path(model_path: Path) -> Path:
    return model_path if model_path.is_file() else model_path / "chat_template.jinja"


def validate_ling3_chat_template(model_path: Path) -> Path:
    """Check that the selected model contains the supported Ling3 template."""
    template_path = _template_path(model_path)
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read chat template {template_path}: {exc}") from exc

    required_fragments = (
        "Bailing V3 chat template",
        "'<role>SYSTEM</role>'",
        "'<role>ASSISTANT</role>'",
        "'<tool_call>' + tc.name",
        "'<arg_key>' + k + '</arg_key>'",
        "'\\n<tool_response>\\n'",
        "'detailed thinking ' + thinking_option",
    )
    missing = [fragment for fragment in required_fragments if fragment not in template]
    if missing:
        raise ValueError(
            f"Unsupported Ling3 chat template {template_path}; missing fragments: "
            + ", ".join(repr(fragment) for fragment in missing)
        )
    return template_path


def _parse_json_sequence(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    values: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid tool definition JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("Each Ling3 tool definition must be a JSON object")
        values.append(value)
    return values


def _parse_system_body(body: str) -> tuple[str | None, list[dict[str, Any]], str]:
    preamble_index = body.rfind(_TOOLS_PREAMBLE)
    if preamble_index >= 0:
        system_content = body[:preamble_index]
        if system_content.endswith("\n"):
            system_content = system_content[:-1]
        tools_start = preamble_index + len(_TOOLS_PREAMBLE)
        tools_end = body.find(_TOOLS_POSTAMBLE, tools_start)
        if tools_end < 0:
            raise ValueError("Ling3 system block has an unterminated tools section")
        tools = _parse_json_sequence(body[tools_start:tools_end])
        suffix = body[tools_end + len(_TOOLS_POSTAMBLE) :]
    else:
        tools = []
        thinking_match = _THINKING_RE.search(body)
        if thinking_match is None:
            raise ValueError("Ling3 system block has no detailed-thinking option")
        content_end = thinking_match.start()
        system_content = body[:content_end]
        if system_content.endswith("\n"):
            system_content = system_content[:-1]
        suffix = body[content_end:]

    thinking_match = _THINKING_RE.fullmatch(suffix)
    if thinking_match is not None:
        thinking_option = thinking_match.group(1)
    elif not suffix and system_content is not None:
        # The template does not append an option when the original system
        # message already contains one.
        embedded_options = re.findall(r"detailed thinking (on|off)", system_content)
        if not embedded_options:
            raise ValueError(
                "Ling3 system block does not contain a detailed-thinking option"
            )
        thinking_option = embedded_options[-1]
    else:
        raise ValueError(
            "Ling3 system block does not end in 'detailed thinking on/off'"
        )
    return system_content or None, tools, thinking_option


def _schema_type(
    tools_by_name: Mapping[str, Mapping[str, Any]], tool_name: str, key: str
) -> str | None:
    tool = tools_by_name.get(tool_name)
    if tool is None:
        return None
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return None
    parameters = function.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        return None
    schema = properties.get(key)
    if not isinstance(schema, Mapping):
        return None
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list):
        non_null_types = [value for value in schema_type if value != "null"]
        if len(non_null_types) == 1 and isinstance(non_null_types[0], str):
            return non_null_types[0]
    return None


def _parse_argument_value(value: str, schema_type: str | None) -> tuple[Any, str]:
    """Mirror SGLang's JSON, unescape, literal-eval, then string fallbacks."""
    try:
        parsed_value = json.loads(value)
        strategy = "json"
    except (json.JSONDecodeError, ValueError):
        try:
            wrapped = json.loads('{"tmp": "' + value + '"}')
            parsed_value = json.loads(wrapped["tmp"])
            strategy = "unescaped_json"
        except (json.JSONDecodeError, ValueError, KeyError):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    parsed_value = ast.literal_eval(value)
                strategy = "python_literal"
            except (ValueError, SyntaxError):
                parsed_value = str(value)
                strategy = "string_fallback"

    if schema_type == "number" and isinstance(parsed_value, str):
        try:
            parsed_value = (
                float(parsed_value)
                if "." in parsed_value or "e" in parsed_value.lower()
                else int(parsed_value)
            )
        except (ValueError, AttributeError):
            pass
    if schema_type == "string" and not isinstance(parsed_value, str):
        if isinstance(parsed_value, (dict, list)):
            parsed_value = json.dumps(parsed_value, ensure_ascii=False)
        else:
            parsed_value = str(parsed_value)
    return parsed_value, strategy


def _matches_schema_type(value: Any, schema_type: str | None) -> bool:
    if schema_type is None:
        return True
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return True


def _parse_tool_call(
    block: str,
    tools_by_name: Mapping[str, Mapping[str, Any]],
    *,
    call_index: int,
) -> tuple[tuple[str, dict[str, Any]] | None, list[dict[str, Any]]]:
    anomalies: list[dict[str, Any]] = []
    detail = _TOOL_DETAIL_RE.fullmatch(block)
    if detail is None:
        anomalies.append(
            _anomaly(
                "malformed_tool_call",
                "error",
                "A closed tool-call block did not match the Ling3 parser format.",
                tool_call_index=call_index,
            )
        )
        return None, anomalies
    name = detail.group(1).strip()
    if not name:
        anomalies.append(
            _anomaly(
                "empty_tool_name",
                "error",
                "A tool-call block has an empty function name.",
                tool_call_index=call_index,
            )
        )
        return None, anomalies
    if name not in tools_by_name:
        anomalies.append(
            _anomaly(
                "unknown_tool",
                "error",
                "SGLang drops calls to functions absent from the supplied tools.",
                tool_call_index=call_index,
                tool_name=name,
            )
        )
        return None, anomalies

    arguments_text = detail.group(2) or ""
    arguments: dict[str, Any] = {}
    argument_matches = list(_TOOL_ARG_RE.finditer(arguments_text))
    residual_parts: list[str] = []
    position = 0
    for match in argument_matches:
        residual_parts.append(arguments_text[position : match.start()])
        position = match.end()
        key = match.group(1).strip()
        if key in arguments:
            anomalies.append(
                _anomaly(
                    "duplicate_tool_argument",
                    "error",
                    "A repeated argument key was overwritten by its last value.",
                    tool_call_index=call_index,
                    tool_name=name,
                    argument_key=key,
                )
            )
        schema_type = _schema_type(tools_by_name, name, key)
        parsed_value, strategy = _parse_argument_value(match.group(2), schema_type)
        arguments[key] = parsed_value
        if not _matches_schema_type(parsed_value, schema_type):
            anomalies.append(
                _anomaly(
                    "tool_argument_type_mismatch",
                    "error",
                    "The parsed argument value does not match its JSON schema type.",
                    tool_call_index=call_index,
                    tool_name=name,
                    argument_key=key,
                    expected_type=schema_type,
                    parsed_type=type(parsed_value).__name__,
                    parse_strategy=strategy,
                )
            )
    residual_parts.append(arguments_text[position:])
    residual = "".join(residual_parts).replace("\\n", "").strip()
    if residual:
        anomalies.append(
            _anomaly(
                "malformed_tool_arguments",
                "error",
                "Incomplete argument XML was ignored by the Ling3 regular expression.",
                tool_call_index=call_index,
                tool_name=name,
                discarded_text=residual[:200],
            )
        )
    return (name, arguments), anomalies


def _find_tool_call_end(text: str, start: int) -> int | None:
    end = text.find(_TOOL_CALL_END, start + len(_TOOL_CALL_START))
    if end < 0:
        return None
    return end + len(_TOOL_CALL_END)


def _parse_assistant_body(
    body: str,
    tools_by_name: Mapping[str, Mapping[str, Any]],
    *,
    assistant_index: int,
    call_id_prefix: str,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    anomalies: list[dict[str, Any]] = []
    reasoning_content = ""
    content_and_calls = body
    if body.startswith(_THINK_START):
        after_start = body[len(_THINK_START) :]
        think_end = after_start.find(_THINK_END)
        tool_start = after_start.find(_TOOL_CALL_START)
        if think_end >= 0 and (tool_start < 0 or think_end < tool_start):
            reasoning_content = after_start[:think_end]
            content_and_calls = after_start[think_end + len(_THINK_END) :]
        elif tool_start >= 0:
            # Ling3's reasoning parser treats a tool call as an implicit
            # reasoning terminator when the model omits </think>.
            reasoning_content = after_start[:tool_start]
            content_and_calls = after_start[tool_start:]
            anomalies.append(
                _anomaly(
                    "reasoning_terminated_by_tool_call",
                    "warning",
                    "The reasoning block omitted </think>; Ling3 terminates it at "
                    "the first tool call.",
                    assistant_index=assistant_index,
                )
            )
        else:
            # Ling3's non-streaming parser moves a reasoning-only truncated
            # generation into normal content so clients receive non-empty text.
            content_and_calls = after_start
            anomalies.append(
                _anomaly(
                    "unterminated_reasoning_to_content",
                    "error",
                    "The reasoning block omitted </think> and contained no tool call; "
                    "Ling3 exposes the text as assistant content.",
                    assistant_index=assistant_index,
                )
            )

    if reasoning_content and not content_and_calls:
        # Ling3's non-streaming force_nonempty_content fallback also applies
        # when a reasoning-only block is properly closed.
        content_and_calls = reasoning_content
        reasoning_content = ""
        anomalies.append(
            _anomaly(
                "reasoning_only_to_content",
                "warning",
                "A closed reasoning-only response was exposed as assistant content "
                "by Ling3's non-streaming fallback.",
                assistant_index=assistant_index,
            )
        )

    first_call = content_and_calls.find(_TOOL_CALL_START)
    if first_call < 0:
        content = content_and_calls
        parsed_calls: list[tuple[str, dict[str, Any]]] = []
    else:
        parsed_calls = []
        closed_blocks: list[tuple[int, int]] = []
        position = first_call
        raw_call_index = 0
        partial_start: int | None = None
        while position < len(content_and_calls):
            start = content_and_calls.find(_TOOL_CALL_START, position)
            if start < 0:
                break
            end = _find_tool_call_end(content_and_calls, start)
            if end is None:
                partial_start = start
                break
            closed_blocks.append((start, end))
            parsed_call, call_anomalies = _parse_tool_call(
                content_and_calls[start:end],
                tools_by_name,
                call_index=raw_call_index,
            )
            anomalies.extend(
                {**anomaly, "assistant_index": assistant_index}
                for anomaly in call_anomalies
            )
            if parsed_call is not None:
                parsed_calls.append(parsed_call)
            raw_call_index += 1
            position = end

        if partial_start is not None:
            anomalies.append(
                _anomaly(
                    "unterminated_tool_call",
                    "error",
                    "The final <tool_call> has no closing tag.",
                    assistant_index=assistant_index,
                    tool_call_index=raw_call_index,
                    recovery=(
                        "discarded_after_complete_calls"
                        if parsed_calls
                        else "preserved_as_content"
                    ),
                    raw_text=content_and_calls[partial_start : partial_start + 200],
                )
            )

        if parsed_calls:
            # This matches FunctionCallParser.parse_non_stream: once at least
            # one valid call exists, only text before the first tool tag is
            # returned as normal content.
            content = content_and_calls[:first_call].strip()
            residual_parts: list[str] = []
            residual_position = closed_blocks[0][1]
            for start, end in closed_blocks[1:]:
                residual_parts.append(content_and_calls[residual_position:start])
                residual_position = end
            if partial_start is not None:
                residual_parts.append(
                    content_and_calls[residual_position:partial_start]
                )
            else:
                residual_parts.append(content_and_calls[residual_position:])
            discarded_text = "".join(residual_parts).strip()
            if discarded_text:
                anomalies.append(
                    _anomaly(
                        "discarded_text_between_tool_calls",
                        "error",
                        "Text after the first tool-call marker was not represented "
                        "by a parsed tool call.",
                        assistant_index=assistant_index,
                        discarded_text=discarded_text[:200],
                    )
                )
        else:
            # SGLang returns the original text when its detector yields no
            # valid calls, even if it contains a tool-call start marker.
            content = content_and_calls

        recovery = (
            "discarded_after_valid_calls" if parsed_calls else "preserved_as_content"
        )
        for anomaly in anomalies:
            if anomaly.get("assistant_index") == assistant_index and anomaly[
                "code"
            ] in {"empty_tool_name", "malformed_tool_call", "unknown_tool"}:
                anomaly["recovery"] = recovery

    tool_calls: list[dict[str, Any]] = []
    call_ids: list[str] = []
    for call_index, (name, arguments) in enumerate(parsed_calls):
        call_id = f"{call_id_prefix}_{call_index}"
        call_ids.append(call_id)
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content if content or not tool_calls else None,
    }
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message, call_ids, anomalies


def _find_tool_response_end(body: str, content_start: int) -> int:
    search_from = content_start
    while True:
        end = body.find(_TOOL_RESPONSE_END, search_from)
        if end < 0:
            raise ValueError("Observation block has an unterminated tool response")
        after_end = end + len(_TOOL_RESPONSE_END)
        if after_end == len(body) or body.startswith(_TOOL_RESPONSE_START, after_end):
            return end
        search_from = end + 1


def _parse_observation_body(body: str) -> list[str]:
    responses: list[str] = []
    position = 0
    while position < len(body):
        if not body.startswith(_TOOL_RESPONSE_START, position):
            raise ValueError("Malformed Ling3 observation block")
        content_start = position + len(_TOOL_RESPONSE_START)
        content_end = _find_tool_response_end(body, content_start)
        responses.append(body[content_start:content_end])
        position = content_end + len(_TOOL_RESPONSE_END)
    if not responses:
        raise ValueError("Ling3 observation block contains no tool responses")
    return responses


def _split_role_blocks(transcript: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    position = 0
    while position < len(transcript):
        role_match = _ROLE_RE.match(transcript, position)
        if role_match is None:
            raise ValueError(
                f"Expected a Ling3 role marker at character {position}, got "
                f"{transcript[position : position + 80]!r}"
            )
        body_start = role_match.end()
        body_end = transcript.find(_ROLE_END, body_start)
        if body_end < 0:
            if role_match.group(1) != "ASSISTANT":
                raise ValueError(f"Ling3 {role_match.group(1)} block is unterminated")
            blocks.append((role_match.group(1), transcript[body_start:]))
            break
        blocks.append((role_match.group(1), transcript[body_start:body_end]))
        position = body_end + len(_ROLE_END)
    return blocks


def parse_ling3_transcript(
    transcript: str, *, call_id_prefix: str = "call_0"
) -> ParsedTranscript:
    """Reverse one fully rendered Ling3 transcript into OpenAI messages."""
    blocks = _split_role_blocks(transcript)
    if not blocks or blocks[0][0] != "SYSTEM":
        raise ValueError("Ling3 transcript must start with a SYSTEM block")

    system_content, tools, thinking_option = _parse_system_body(blocks[0][1])
    messages: list[dict[str, Any]] = []
    if system_content is not None:
        messages.append({"role": "system", "content": system_content})
    tools_by_name = {
        function["name"]: tool
        for tool in tools
        if isinstance((function := tool.get("function")), Mapping)
        and isinstance(function.get("name"), str)
    }

    pending_call_ids: list[str] = []
    pending_assistant_index: int | None = None
    assistant_index = 0
    anomalies: list[dict[str, Any]] = []
    if not transcript.endswith(_ROLE_END):
        anomalies.append(
            _anomaly(
                "unterminated_assistant_block",
                "error",
                "The final assistant role has no <|role_end|> marker.",
                assistant_index=sum(role == "ASSISTANT" for role, _ in blocks) - 1,
            )
        )

    for role, body in blocks[1:]:
        if pending_call_ids and role != "OBSERVATION":
            anomalies.append(
                _anomaly(
                    "tool_calls_without_responses",
                    "error",
                    "Tool calls were followed by another role without responses.",
                    assistant_index=pending_assistant_index,
                    missing_response_count=len(pending_call_ids),
                    next_role=role.lower(),
                )
            )
        if role == "HUMAN":
            pending_call_ids = []
            pending_assistant_index = None
            messages.append({"role": "user", "content": body})
        elif role == "SYSTEM":
            pending_call_ids = []
            pending_assistant_index = None
            messages.append({"role": "system", "content": body})
        elif role == "ASSISTANT":
            pending_call_ids = []
            pending_assistant_index = None
            message, pending_call_ids, assistant_anomalies = _parse_assistant_body(
                body,
                tools_by_name,
                assistant_index=assistant_index,
                call_id_prefix=f"{call_id_prefix}_{assistant_index}",
            )
            anomalies.extend(assistant_anomalies)
            messages.append(message)
            if pending_call_ids:
                pending_assistant_index = assistant_index
            assistant_index += 1
        else:
            responses = _parse_observation_body(body)
            if len(responses) < len(pending_call_ids):
                anomalies.append(
                    _anomaly(
                        "tool_calls_without_responses",
                        "error",
                        "An observation block contains fewer responses than the "
                        "preceding assistant tool calls.",
                        assistant_index=pending_assistant_index,
                        tool_call_count=len(pending_call_ids),
                        tool_response_count=len(responses),
                        missing_response_count=len(pending_call_ids) - len(responses),
                    )
                )
            elif len(responses) > len(pending_call_ids):
                anomalies.append(
                    _anomaly(
                        "tool_responses_without_calls",
                        "error",
                        "An observation block contains more responses than the "
                        "preceding assistant tool calls.",
                        assistant_index=pending_assistant_index,
                        tool_call_count=len(pending_call_ids),
                        tool_response_count=len(responses),
                        orphan_response_count=len(responses) - len(pending_call_ids),
                    )
                )
            response_call_ids = list(pending_call_ids[: len(responses)])
            response_call_ids.extend(
                f"{call_id_prefix}_orphan_{assistant_index}_{index}"
                for index in range(len(response_call_ids), len(responses))
            )
            for call_id, content in zip(response_call_ids, responses, strict=True):
                messages.append(
                    {"role": "tool", "content": content, "tool_call_id": call_id}
                )
            pending_call_ids = []
            pending_assistant_index = None

    if pending_call_ids:
        anomalies.append(
            _anomaly(
                "tool_calls_without_responses",
                "error",
                "The transcript ends before responses to the final tool calls.",
                assistant_index=pending_assistant_index,
                missing_response_count=len(pending_call_ids),
                next_role="end_of_transcript",
            )
        )

    return ParsedTranscript(
        messages=messages,
        tools=tools,
        thinking_option=thinking_option,
        anomalies=anomalies,
    )


def _semantic_signature(parsed: ParsedTranscript) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in parsed.messages:
        normalized = dict(message)
        reasoning = normalized.get("reasoning_content")
        if isinstance(reasoning, str):
            normalized["reasoning_content"] = reasoning.strip("\n")
        messages.append(normalized)
    return {
        "messages": messages,
        "tools": parsed.tools,
        "thinking_option": parsed.thinking_option,
    }


def _render_tool_call(tool_call: Mapping[str, Any]) -> str:
    function = tool_call.get("function", tool_call)
    if not isinstance(function, Mapping):
        raise ValueError("OpenAI tool call has no function object")
    name = function.get("name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise ValueError("OpenAI tool call has an invalid name or arguments")
    # The non-whitespace-trimmed ``{% set _args = ... %}`` line in the model's
    # Jinja template emits one newline after the function name.
    rendered = _TOOL_CALL_START + name + "\n"
    for key, value in arguments.items():
        rendered += f"<arg_key>{key}</arg_key>\n<arg_value>"
        rendered += (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
        rendered += "</arg_value>"
    return rendered + "\n" + _TOOL_CALL_END


def render_ling3_transcript(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    thinking_option: str = "on",
) -> str:
    """Render parsed messages with the supported Ling3 template for validation."""
    if thinking_option not in {"on", "off"}:
        raise ValueError("thinking_option must be 'on' or 'off'")
    first_system = bool(messages and messages[0].get("role") == "system")
    system_content = messages[0].get("content", "") if first_system else ""
    if not isinstance(system_content, str):
        raise ValueError("System message content must be a string")

    output = "<role>SYSTEM</role>"
    if tools:
        if first_system:
            output += system_content + "\n"
        output += _TOOLS_PREAMBLE
        for tool in tools:
            output += "\n" + json.dumps(tool, ensure_ascii=False)
        output += _TOOLS_POSTAMBLE
        if "detailed thinking on" not in system_content and (
            "detailed thinking off" not in system_content
        ):
            output += "detailed thinking " + thinking_option
        output += _ROLE_END
    else:
        if first_system:
            output += system_content
            if "detailed thinking on" not in system_content and (
                "detailed thinking off" not in system_content
            ):
                output += "\ndetailed thinking " + thinking_option
        else:
            output += "detailed thinking " + thinking_option
        output += _ROLE_END

    for index, message in enumerate(messages):
        role = message.get("role")
        content_value = message.get("content")
        content = content_value if isinstance(content_value, str) else ""
        if role == "system":
            if index:
                output += f"<role>SYSTEM</role>{content}{_ROLE_END}"
        elif role == "user":
            output += f"<role>HUMAN</role>{content}{_ROLE_END}"
        elif role == "assistant":
            reasoning = message.get("reasoning_content", "")
            if not isinstance(reasoning, str):
                reasoning = ""
            output += "<role>ASSISTANT</role>\n<think>"
            output += reasoning.strip("\n") + _THINK_END + content.lstrip("\n")
            tool_calls = message.get("tool_calls") or []
            for call_index, tool_call in enumerate(tool_calls):
                if (call_index == 0 and content) or call_index > 0:
                    output += "\n"
                output += _render_tool_call(tool_call)
            output += _ROLE_END
        elif role == "tool":
            if index == 0 or messages[index - 1].get("role") != "tool":
                output += "<role>OBSERVATION</role>"
            output += _TOOL_RESPONSE_START + content + _TOOL_RESPONSE_END
            if index + 1 == len(messages) or messages[index + 1].get("role") != "tool":
                output += _ROLE_END
    return output


def _validate_segments(row: Mapping[str, Any], transcript: str) -> None:
    segments = row.get("segments")
    if segments is None:
        return
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ValueError("rollout segments must be a sequence")
    segment_texts: list[str] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping) or not isinstance(segment.get("text"), str):
            raise ValueError(f"rollout segments[{index}] has no string text")
        segment_texts.append(segment["text"])
    if "".join(segment_texts) != transcript:
        raise ValueError("rollout segments do not reconstruct prompt + completion")


def convert_rollout_record(
    row: Mapping[str, Any],
    *,
    record_index: int = 0,
    model_name: str,
    max_tokens: int | None = None,
    instance_id: str | None = None,
    validate_roundtrip: bool = True,
) -> dict[str, Any]:
    """Convert one rollout dump record to the nested OpenAI trajectory layout."""
    if max_tokens is not None and (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer or None")
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, str) or not isinstance(completion, str):
        raise ValueError("rollout record requires string prompt and completion fields")
    transcript = prompt + completion
    _validate_segments(row, transcript)
    call_id_prefix = f"call_{record_index}"
    parsed = parse_ling3_transcript(transcript, call_id_prefix=call_id_prefix)
    anomalies = list(parsed.anomalies)
    if validate_roundtrip:
        rendered = render_ling3_transcript(
            parsed.messages,
            parsed.tools,
            thinking_option=parsed.thinking_option,
        )
        if rendered != transcript:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(rendered, transcript)
                    )
                    if actual != expected
                ),
                min(len(rendered), len(transcript)),
            )
            reparsed = parse_ling3_transcript(rendered, call_id_prefix=call_id_prefix)
            if _semantic_signature(reparsed) != _semantic_signature(parsed):
                raise ValueError(
                    "Ling3 semantic round-trip validation failed after a byte "
                    f"mismatch at character {mismatch}: "
                    f"rendered={rendered[mismatch : mismatch + 80]!r}, "
                    f"source={transcript[mismatch : mismatch + 80]!r}"
                )
            anomalies.append(
                _anomaly(
                    "noncanonical_roundtrip",
                    "warning",
                    "The source transcript required normalization before it could "
                    "round-trip semantically.",
                    first_mismatch_character=mismatch,
                )
            )

    resolved_instance_id = instance_id
    if resolved_instance_id is None:
        source_instance_id = row.get("instance_id", row.get("task_id"))
        if source_instance_id is None:
            raise ValueError("rollout record has neither instance_id nor task_id")
        resolved_instance_id = str(source_instance_id)

    conversation = {
        "messages": parsed.messages,
        "tools": parsed.tools,
        "model": model_name,
    }
    if max_tokens is not None:
        conversation["max_tokens"] = max_tokens
    converted: dict[str, Any] = {
        "instance_id": resolved_instance_id,
        "conversations": [conversation],
        "conversion_status": "anomalous" if anomalies else "clean",
        "conversion_tags": sorted({anomaly["code"] for anomaly in anomalies}),
        "conversion_anomalies": anomalies,
    }
    converted.update(
        {key: value for key, value in row.items() if key not in _ROLLOUT_PAYLOAD_FIELDS}
    )
    return converted


def convert_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    model_path: Path,
    model_name: str | None = None,
    max_tokens: int | None = None,
    instance_id: str | None = None,
    limit: int | None = None,
    validate_roundtrip: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stream-convert a rollout JSONL and atomically publish the result."""
    template_path = validate_ling3_chat_template(model_path)
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer or None")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from input path")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass overwrite=True to replace it"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        text=True,
    )
    converted_count = 0
    records_with_anomalies = 0
    anomaly_counts: Counter[str] = Counter()
    anomaly_record_counts: Counter[str] = Counter()
    resolved_model_name = model_name or template_path.parent.name
    try:
        with (
            input_path.open(encoding="utf-8") as input_file,
            os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file,
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
                        record_index=converted_count,
                        model_name=resolved_model_name,
                        max_tokens=max_tokens,
                        instance_id=instance_id,
                        validate_roundtrip=validate_roundtrip,
                    )
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
                output_file.write(json.dumps(converted, ensure_ascii=False) + "\n")
                converted_count += 1
                tags = converted["conversion_tags"]
                if tags:
                    records_with_anomalies += 1
                    anomaly_record_counts.update(tags)
                    anomaly_counts.update(
                        anomaly["code"] for anomaly in converted["conversion_anomalies"]
                    )
                if limit is not None and converted_count >= limit:
                    break
        if converted_count == 0:
            raise ValueError(f"Input dataset is empty: {input_path}")
        os.replace(temporary_name, output_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    return {
        "input": str(input_path),
        "output": str(output_path),
        "records": converted_count,
        "records_with_anomalies": records_with_anomalies,
        "anomaly_counts": dict(sorted(anomaly_counts.items())),
        "anomaly_record_counts": dict(sorted(anomaly_record_counts.items())),
        "model": resolved_model_name,
        "chat_template": str(template_path),
        "roundtrip_validated": validate_roundtrip,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Rollout JSONL.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL.")
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
        help="Override instance_id for every row (default: source instance_id/task_id).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Convert only the first N non-empty records.",
    )
    parser.add_argument(
        "--no-roundtrip-validation",
        action="store_true",
        help="Allow non-canonical model output that cannot be re-rendered byte exactly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file atomically.",
    )
    args = parser.parse_args()

    convert_jsonl(
        args.input,
        args.output,
        model_path=args.model_path,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        instance_id=args.instance_id,
        limit=args.limit,
        validate_roundtrip=not args.no_roundtrip_validation,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
