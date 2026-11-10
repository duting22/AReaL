"""Evidence boundaries for offline Blender failure analysis."""

import json

import pytest

from examples.blender_overfit.diagnose import analyze_trajectory, decode_output


def test_native_output_unwraps_double_escaped_input_text():
    payload = {"stdout": "saved /scratch/scene.blend\n", "stderr": "", "exitCode": 0}
    escaped = json.dumps(json.dumps(payload))[1:-1]
    content = json.dumps([{"type": "inputText", "text": escaped}])

    assert decode_output(content) == [payload]


def test_save_instruction_in_tool_input_is_not_delivery_evidence():
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "write",
                        "name": "fs_create",
                        "input": {
                            "path": "/scratch/build.py",
                            "text": "import bpy\nbpy.ops.wm.save_as_mainfile(filepath='/work/scene.blend')",
                        },
                    }
                ]
            },
        }
    ]

    result = analyze_trajectory(events, {}, "/work")

    assert result["confirmed_save_logs"] == []
    assert result["blender_script_command_count"] == 0
    assert result["observed_stage"] == "blender_script_written_not_executed"


def test_saved_scratch_scene_is_distinguished_from_requested_delivery():
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "run",
                        "name": "run_command",
                        "input": {"command": "blender --python build.py"},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "run",
                        "is_error": False,
                        "content": json.dumps(
                            {
                                "stdout": 'saved /scratch/scene.blend\nRead blend: "/scratch/scene.blend"\n',
                                "stderr": "",
                            }
                        ),
                    }
                ]
            },
        },
    ]

    result = analyze_trajectory(events, {}, "/work")

    assert result["observed_stage"] == "saved_outside_delivery_directory"
    assert len(result["save_logs_outside_delivery"]) == 1
    assert len(result["confirmed_blend_read_logs"]) == 1


def test_reasoning_evidence_is_not_double_counted_or_equated_to_usage():
    text = "repeat this operation;" * 10
    events = [
        {"type": "core_item", "item": {"type": "reasoning", "content": [text]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
    ]

    result = analyze_trajectory(events, {"usage": {"outputTokens": 262144}}, "/work")

    assert result["retained_reasoning_items"] == 1
    assert result["largest_retained_reasoning_bytes"] == len(text.encode())
    assert result["output_tokens_completed_or_reported"] == 262144
    assert result["repeated_reasoning_fragments"][0]["count"] == 10


def test_streamed_judge_preserves_request_and_completed_json(tmp_path, monkeypatch):
    from examples.blender_overfit.verifier import call_judge

    frames = [
        {
            "model": "judge",
            "choices": [{"index": 0, "delta": {"reasoning_content": "reason"}}],
        },
        {"choices": [{"index": 0, "delta": {"content": '{"score":'}}]},
        {
            "choices": [
                {"index": 0, "delta": {"content": "50}"}, "finish_reason": "stop"}
            ]
        },
        {"choices": [], "usage": {"completion_tokens": 10}},
    ]
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            for frame in frames:
                yield ("data: " + json.dumps(frame) + "\n\n").encode()
            yield b"data: [DONE]\n\n"

    def urlopen(request, timeout):
        sent.append(json.loads(request.data))
        return Response()

    monkeypatch.setenv("BUILDING_VLM_STREAM", "1")
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    body = {
        "model": "judge",
        "messages": [{"role": "user", "content": "evidence"}],
        "response_format": {"type": "json_object"},
    }

    parsed, receipt = call_judge(
        body, "https://example.test/v1", "fixture", 10, tmp_path / "response.json"
    )

    assert parsed == {"score": 50}
    assert len(sent) == 1
    assert {
        k: v for k, v in sent[0].items() if k not in {"stream", "stream_options"}
    } == body
    assert receipt["usage"] == {"completion_tokens": 10}
    assert (
        json.loads((tmp_path / "response.json").read_text())["choices"][0]["message"][
            "reasoning_content"
        ]
        == "reason"
    )


def test_truncated_judge_stream_is_not_accepted_as_a_score():
    from examples.blender_overfit.verifier import read_judge_stream

    frames = [
        b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n'
    ]

    with pytest.raises(ValueError, match="Incomplete judge stream"):
        read_judge_stream(frames)


def test_short_blender_flag_and_relative_script_are_execution_evidence():
    events = [
        {
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "write",
                        "name": "fs_create",
                        "input": {"path": "/scratch/build.py", "text": "import bpy"},
                    },
                    {
                        "type": "tool_use",
                        "id": "run",
                        "name": "run_command",
                        "input": {"command": "blender -b -P build.py"},
                    },
                ]
            }
        }
    ]

    result = analyze_trajectory(events, {}, "/work")

    assert result["blender_script_command_count"] == 1
    assert result["observed_stage"] == "blender_executed_without_confirmed_scene_save"


def test_authorization_scope_failure_is_retained_separately_from_python_errors():
    error = {
        "code": "PERMISSION_DENIED",
        "message": "authorization directory identity changed; create a new scope",
    }
    events = [
        {
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "run",
                        "name": "run_command",
                        "input": {"command": "echo probe"},
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "run",
                        "content": json.dumps({"error": error}),
                    },
                ]
            }
        }
    ]

    result = analyze_trajectory(events, {}, "/work")

    assert result["tool_errors"] == [error]
    assert result["observed_errors"] == []
    assert result["observed_stage"] == "tool_scope_invalid_after_execution"
