"""CPU tests for paired SFT inputs, checkpoint pinning, and one-shot boundaries."""

import base64
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.blender_overfit.prepare import (
    detail_features,
    extract_code,
    generator_identity,
    latest_checkpoint,
    prepare,
    query_images,
)
from examples.blender_overfit.single_turn import generate, session_headers
from examples.blender_overfit.verifier import parse_votes, request_body


def _hair_instance(**changes):
    settings = dict(
        type="HAIR",
        render_type="PATH",
        count=120200,
        hair_length=1.0,
        root_radius=0.0,
        tip_radius=0.0,
        radius_scale=0.01,
        child_type="NONE",
        rendered_child_count=100,
        child_radius=0.2,
        kink="NO",
        kink_amplitude=0.2,
        clump_factor=0.0,
    )
    settings.update(changes)
    system = SimpleNamespace(
        settings=SimpleNamespace(**settings), use_hair_dynamics=False, is_edited=False
    )
    modifier = SimpleNamespace(
        type="PARTICLE_SYSTEM", show_render=True, particle_system=system
    )
    obj = SimpleNamespace(
        name="Emitter",
        modifiers=[modifier],
        bound_box=list(product([-0.1, 0.1], repeat=3)),
    )
    return SimpleNamespace(
        object=obj,
        matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    )


def test_legacy_hair_bounds_include_paths_beyond_emitter_mesh():
    from examples.blender_overfit.scene import particle_hair_bounds

    instance = _hair_instance()
    diagnostics = []

    corners = particle_hair_bounds(instance, diagnostics)

    assert len(corners) == 8
    assert min(p[0] for p in corners) == pytest.approx(-1.1)
    assert max(p[2] for p in corners) == pytest.approx(1.1)
    assert diagnostics[0]["count"] == 120200


def test_legacy_hair_bounds_apply_nonuniform_world_scale_and_translation():
    from examples.blender_overfit.scene import particle_hair_bounds

    instance = _hair_instance()
    instance.matrix_world = [[2, 0, 0, 10], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 1]]

    corners = particle_hair_bounds(instance)

    assert min(p[0] for p in corners) == pytest.approx(5.8)
    assert max(p[2] for p in corners) == pytest.approx(4.4)


@pytest.mark.parametrize(
    "changes,extent",
    [
        ({"tangent_factor": 0.88}, 4.62),
        ({"factor_random": 0.34}, 2.46),
        ({"object_align_factor": (0.1, -0.2, 0.3)}, 3.5),
        ({"timestep": 0.08}, 2.1),
    ],
)
def test_hair_bounds_include_independent_initial_velocities(changes, extent):
    from examples.blender_overfit.scene import particle_hair_bounds

    corners = particle_hair_bounds(_hair_instance(**changes))

    assert max(p[0] for p in corners) == pytest.approx(extent)


def test_hair_world_velocity_bound_does_not_shrink_with_emitter_scale():
    from examples.blender_overfit.scene import particle_hair_bounds

    instance = _hair_instance()
    instance.matrix_world = [
        [0.1, 0, 0, 0],
        [0, 0.1, 0, 0],
        [0, 0, 0.1, 0],
        [0, 0, 0, 1],
    ]

    corners = particle_hair_bounds(instance)

    assert max(p[0] for p in corners) == pytest.approx(1.01)


def test_hair_bounds_reject_nonfinite_timestep():
    from examples.blender_overfit.scene import particle_hair_bounds

    with pytest.raises(ValueError, match="Invalid legacy particle hair extent"):
        particle_hair_bounds(_hair_instance(timestep=float("nan")))


@pytest.mark.parametrize("name", ["brownian_factor", "object_factor", "damping"])
def test_forced_hair_requires_evaluated_bounds(name):
    from examples.blender_overfit.scene import particle_hair_bounds

    with pytest.raises(ValueError, match="requires evaluated path bounds"):
        particle_hair_bounds(_hair_instance(**{name: 0.1}))


@pytest.mark.parametrize("render_type,count", [("COLLECTION", 100), ("PATH", 0)])
def test_non_path_particles_do_not_change_existing_geometry_bounds(render_type, count):
    from examples.blender_overfit.scene import particle_hair_bounds

    assert (
        particle_hair_bounds(_hair_instance(render_type=render_type, count=count)) == []
    )


@pytest.mark.parametrize(
    "child_type,extent", [("NONE", 1.1), ("INTERPOLATED", 1.1), ("SIMPLE", 1.3)]
)
def test_child_bounds_only_include_offsets_used_by_child_type(child_type, extent):
    from examples.blender_overfit.scene import particle_hair_bounds

    corners = particle_hair_bounds(_hair_instance(child_type=child_type))

    assert max(p[0] for p in corners) == pytest.approx(extent)


@pytest.mark.parametrize("matching", [True, False])
def test_render_only_mode_binds_candidate_and_never_calls_visual_judge(
    tmp_path, monkeypatch, matching
):
    import hashlib

    from examples.blender_overfit.arena import reward_main

    agent = tmp_path / "agent"
    agent.mkdir()
    candidate = b"recorded candidate"
    (agent / "scene.blend").write_bytes(candidate)
    checksum = hashlib.sha256(candidate).hexdigest()
    config = tmp_path / "private-render-config.json"
    config.write_text(
        json.dumps(
            {"candidate_sha256": checksum if matching else "wrong", "view_indices": [1]}
        )
    )
    output = tmp_path / "reward"
    for name, value in {
        "ARENA_AGENT_OUTPUT_DIR": agent,
        "ARENA_REWARD_OUTPUT_DIR": output,
        "ARENA_REWARD_OUTPUT_FILE": output / "reward_result.json",
        "ARENA_QUERY_PATH": config,
        "ARENA_ORACLE_PATH": tmp_path / "oracle",
        "OVERFIT_RENDER_ONLY": "1",
    }.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.delenv("OVERFIT_SAVE_DISPLAY_SCENE", raising=False)
    monkeypatch.setattr(
        "examples.blender_overfit.arena.materialize_references",
        lambda _, root: (root / "manifest.json").write_text(
            json.dumps({"tasks": [{"task_id": "fixture"}]})
        ),
    )
    calls = []

    def render(command, **kwargs):
        calls.append(command)
        assert command[command.index("--views") + 1] == "1"
        evidence = Path(command[command.index("--output") + 1])
        evidence.mkdir()
        (evidence / "render-manifest.json").write_text(
            json.dumps({"candidate_sha256": checksum, "images": []})
        )

    monkeypatch.setattr("examples.blender_overfit.arena.subprocess.run", render)
    monkeypatch.setattr(
        "examples.blender_overfit.verifier.main",
        lambda: pytest.fail("Rendering-only task invoked a visual judge"),
    )
    if not matching:
        with pytest.raises(ValueError, match="different candidate"):
            reward_main()
        assert calls == []
        return

    reward_main()

    result = json.loads((output / "reward_result.json").read_text())
    assert result["diagnostic_only"] is True
    assert result["model_calls"] == 0
    assert len(calls) == 1


@pytest.mark.parametrize("hair_length", [float("nan"), -1.0])
def test_legacy_hair_bounds_reject_invalid_length(hair_length):
    from examples.blender_overfit.scene import particle_hair_bounds

    with pytest.raises(ValueError, match="Invalid legacy"):
        particle_hair_bounds(_hair_instance(hair_length=hair_length))


@pytest.mark.parametrize(
    "score,reason,delivered,terminal,expected",
    [
        (None, "REASONING_BYTE_LIMIT_1MIB", False, True, (0.0, True)),
        (0.4, "LLM_OUTPUT_TOKEN_LIMIT_EXCEEDED", True, True, (0.4, False)),
        (0.0, None, False, True, (0.0, False)),
        (None, "REASONING_BYTE_LIMIT_1MIB", False, False, (None, False)),
    ],
)
def test_budget_policy_preserves_deliveries_and_only_excludes_terminal_limits(
    score, reason, delivered, terminal, expected
):
    from examples.blender_overfit.report import apply_budget_policy

    assert apply_budget_policy(score, reason, delivered, terminal) == expected


def test_budget_exclusion_drops_same_question_from_both_modes():
    from examples.blender_overfit.report import paired_summary

    rows = [
        {
            "task_id": "limit",
            "sft_score": 0.7,
            "agent_score": 0.0,
            "agent_budget_without_scene": True,
        },
        {"task_id": "code_failure", "sft_score": 0.0, "agent_score": 0.4},
        {"task_id": "delivered", "sft_score": 0.9, "agent_score": 0.5},
    ]

    primary = paired_summary(rows)
    excluding = paired_summary(rows, exclude_budget=True)

    assert primary["count"] == 3
    assert excluding["task_ids"] == ["code_failure", "delivered"]
    assert excluding["sft_mean"] == pytest.approx(0.45)
    assert excluding["agent_mean"] == pytest.approx(0.45)


def test_paired_report_keeps_per_task_and_csv_budget_exclusions_consistent(
    tmp_path, monkeypatch
):
    from examples.blender_overfit.report import build_report

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    run = dataset / "run"
    run.mkdir()
    tasks = [
        {"task_id": "limit", "source_row_index": 0},
        {"task_id": "code_error", "source_row_index": 1},
    ]
    (dataset / "manifest.json").write_text(
        json.dumps({"tasks": tasks, "checkpoint": "fixture"})
    )
    records = []
    for task, mode, score, scene in [
        ("limit", "sft", 0.7, True),
        ("limit", "agent", None, False),
        ("code_error", "sft", 0, False),
        ("code_error", "agent", 0.4, True),
    ]:
        job = f"{task}-{mode}"
        artifacts = run / "tasks" / job / "artifacts"
        artifacts.mkdir(parents=True)
        if scene:
            (artifacts / "scene.blend").write_bytes(b"fixture")
        if task == "limit" and mode == "agent":
            (artifacts / "native-receipt.json").write_text(
                json.dumps({"error": [{"message": "reasoning exceeds 1 MiB"}]})
            )
        records.append(
            {
                "include_in_comparison": True,
                "source_task": task,
                "mode": mode,
                "job_id": job,
                "status": "HARNESS_FAILED" if score is None else "DONE",
                "score": score,
            }
        )
    (run / "state.json").write_text(
        json.dumps({"model": "fixture", "attempts": records})
    )
    monkeypatch.setattr(
        "examples.blender_overfit.gallery.build_gallery", lambda _: None
    )

    summary = build_report(dataset, run, "https://example.invalid")

    assert summary["paired"]["count"] == 2
    assert summary["paired_excluding_budget"]["task_ids"] == ["code_error"]
    for mode in ("sft", "agent"):
        result = json.loads(
            (dataset / "results" / mode / "limit/result.json").read_text()
        )
        assert result["paired_excluded_for_budget"] is True
        assert result["score_excluding_budget"] is None
    ordinary = json.loads((dataset / "results/sft/code_error/result.json").read_text())
    assert ordinary["score_excluding_budget"] == 0
    assert ordinary["paired_excluded_for_budget"] is False


@pytest.mark.parametrize("source_preview", [True, False])
def test_report_regrade_removes_superseded_evidence_keeps_source(
    tmp_path, monkeypatch, source_preview
):
    from examples.blender_overfit.report import build_report

    dataset = tmp_path / "dataset"
    run = dataset / "run"
    artifacts = run / "tasks/source-job/artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "scene.blend").write_bytes(b"original candidate")
    if source_preview:
        (artifacts / "preview.png").write_bytes(b"source presentation")
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "tasks": [{"task_id": "object", "source_row_index": 0}],
                "checkpoint": "fixture",
            }
        )
    )
    (run / "state.json").write_text(
        json.dumps(
            {
                "model": "fixture",
                "attempts": [
                    {
                        "source_task": "object",
                        "mode": "agent",
                        "include_in_comparison": True,
                        "job_id": "source-job",
                        "evaluation_job_id": "new-grade",
                        "evaluation_task_id": "new-grade-task",
                        "status": "EVALUATING",
                        "score": None,
                    }
                ],
            }
        )
    )
    target = dataset / "results/agent/object"
    (target / "display").mkdir(parents=True)
    (target / "verification").mkdir()
    (target / "verification/view-01.png").write_bytes(b"superseded evidence")
    current_evidence = run / "tasks/new-grade/artifacts"
    current_evidence.mkdir(parents=True)
    (current_evidence / "view-02.png").write_bytes(b"current evidence")
    (target / "result.json").write_text(
        json.dumps({"job_id": "source-job", "evaluation_task_id": "old-grade-task"})
    )
    for name in (
        "preview.png",
        "verification-preview.png",
        "display/scene.blend",
        "visual-score.json",
    ):
        (target / name).write_bytes(b"superseded")
    monkeypatch.setattr(
        "examples.blender_overfit.gallery.build_gallery", lambda _: None
    )

    build_report(dataset, run, "https://example.invalid")

    assert (target / "scene.blend").read_bytes() == b"original candidate"
    assert not (target / "verification-preview.png").exists()
    assert not (target / "display/scene.blend").exists()
    assert not (target / "visual-score.json").exists()
    assert not (target / "verification/view-01.png").exists()
    assert (target / "verification/view-02.png").read_bytes() == b"current evidence"
    if source_preview:
        assert (target / "preview.png").read_bytes() == b"source presentation"
    else:
        assert not (target / "preview.png").exists()


@pytest.mark.parametrize(
    "outcome,errors,expected",
    [
        (
            {"code": "LLM_OUTPUT_TOKEN_LIMIT_EXCEEDED"},
            [],
            "LLM_OUTPUT_TOKEN_LIMIT_EXCEEDED",
        ),
        (
            {"code": "HARNESS_INTERNAL_ERROR"},
            [{"message": "reasoning exceeds 1 MiB"}],
            "REASONING_BYTE_LIMIT_1MIB",
        ),
        (
            {
                "code": "LLM_RESPONSE_FAILED",
                "details": {"code": "tool_call_budget_exceeded"},
            },
            [],
            "TOOL_CALL_ARGUMENT_BYTES_LIMIT_EXCEEDED",
        ),
        (
            {"code": "HARNESS_INTERNAL_ERROR"},
            [{"message": "unrelated transport error"}],
            None,
        ),
    ],
)
def test_budget_receipt_requires_explicit_limit_evidence(
    tmp_path, outcome, errors, expected
):
    from examples.blender_overfit.report import budget_reason

    (tmp_path / "native-receipt.json").write_text(
        json.dumps({"raw": {"outcome": outcome}, "error": errors})
    )

    assert budget_reason({}, tmp_path) == expected


@pytest.mark.parametrize(
    "kind,delivered,expected",
    [("agent", True, 0), ("agent", False, 0), ("infrastructure", True, 1)],
)
def test_native_budget_failure_preserves_delivery_and_infrastructure_status(
    tmp_path, kind, delivered, expected
):
    from examples.blender_overfit.history32 import normalize_terminal

    output = tmp_path / "agent"
    output.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if delivered:
        (workspace / "scene.blend").write_bytes(b"recorded scene")
    path = tmp_path / "harness_result.json"
    result = {"status": "ERROR", "raw": {"outcome": {"class": kind, "code": "fixture"}}}
    path.write_text(json.dumps(result))

    actual = normalize_terminal(1, path, output, workspace)

    assert actual == expected
    assert (output / "model-failure.json").exists() == (
        kind == "agent" and not delivered
    )
    assert json.loads(path.read_text())["status"] == (
        "OK" if kind == "agent" else "ERROR"
    )
    if delivered:
        assert (workspace / "scene.blend").read_bytes() == b"recorded scene"


@pytest.mark.parametrize("configured,expected", [(None, 2), ("8", 8), ("0", None)])
def test_renderer_workers_preserve_default_and_require_positive_count(
    monkeypatch, configured, expected
):
    from examples.blender_overfit.arena import renderer_threads

    monkeypatch.delenv("OVERFIT_RENDER_THREADS", raising=False)
    if configured is not None:
        monkeypatch.setenv("OVERFIT_RENDER_THREADS", configured)

    if expected is None:
        with pytest.raises(ValueError, match="must be positive"):
            renderer_threads()
    else:
        assert renderer_threads() == expected


def test_streaming_one_shot_assembles_one_completion_and_usage(monkeypatch):
    import io

    calls = []
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": "```python\n"}}]},
        {"choices": [{"delta": {"content": "x = 1\n```"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 8}},
    ]

    def urlopen(request, **kwargs):
        calls.append(json.loads(request.data))
        return io.BytesIO(
            b"".join(
                b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks
            )
            + b"data: [DONE]\n\n"
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    messages = [{"role": "user", "content": "fixture"}]

    result = generate(
        messages,
        model="fixture",
        base_url="https://example.invalid",
        api_key="fixture",
        temperature=1,
        max_tokens=100,
        seed=42,
        timeout=1,
        stream=True,
    )

    assert len(calls) == 1
    assert calls[0]["messages"] == messages
    assert calls[0]["stream"] is True
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tools" not in calls[0]
    assert result["choices"][0]["message"]["content"] == "```python\nx = 1\n```"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 8


def test_recorded_delivery_replay_preserves_scene_without_generation(
    tmp_path, monkeypatch
):
    import hashlib

    from examples.blender_overfit.replay import main

    env = tmp_path / "env/evidence"
    env.mkdir(parents=True)
    scene = b"BLENDER-recorded-scene"
    (env / "scene.blend").write_bytes(scene)
    (env / "source.json").write_text(
        json.dumps(
            {
                "source_task_id": "fixture",
                "scene_sha256": hashlib.sha256(scene).hexdigest(),
            }
        )
    )
    output = tmp_path / "out"
    output.mkdir()
    for key, value in {
        "ARENA_ENV_DIR": env.parent,
        "ARENA_AGENT_OUTPUT_DIR": output / "agent",
        "ARENA_OUTPUT_DIR": output,
    }.items():
        monkeypatch.setenv(key, str(value))

    main()

    assert (output / "agent/scene.blend").read_bytes() == scene
    assert json.loads((output / "harness_result.json").read_text())["model_calls"] == 0


@pytest.mark.parametrize(
    "files,expected",
    [
        ({"scene-native.blend": b"native"}, "scene-native.blend"),
        ({"scene.blend": b"presented", "scene-native.blend": b"native"}, "scene.blend"),
        ({"scene.blend": b"", "scene-native.blend": b"native"}, "scene-native.blend"),
        ({"scene.blend": b""}, None),
    ],
)
def test_delivery_detection_preserves_raw_scene_after_presentation_failure(
    tmp_path, files, expected
):
    from examples.blender_overfit.replay import delivered_scene

    for name, content in files.items():
        (tmp_path / name).write_bytes(content)

    actual = delivered_scene(tmp_path)

    assert (actual.name if actual is not None else None) == expected


def test_invalid_judge_json_retains_raw_response(tmp_path, monkeypatch):
    import io

    from examples.blender_overfit.verifier import call_judge

    response = {
        "choices": [{"finish_reason": "stop", "message": {"content": '{"bad": }'}}]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(response).encode()),
    )
    path = tmp_path / "judge-response.json"

    with pytest.raises(json.JSONDecodeError):
        call_judge({}, "https://example.invalid/v1", "fixture-key", 1, path)

    assert json.loads(path.read_text()) == response


def test_invalid_model_program_reaches_grader_as_model_failure(tmp_path, monkeypatch):
    from examples.blender_overfit.arena import sft_main

    query = tmp_path / "query.json"
    query.write_text("[]")
    for key, value in {
        "ARENA_QUERY_PATH": query,
        "ARENA_OUTPUT_DIR": tmp_path / "out",
        "ARENA_AGENT_OUTPUT_DIR": tmp_path / "agent",
    }.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(
        "examples.blender_overfit.single_turn.generate",
        lambda *args, **kwargs: {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "No Python program"}}
            ]
        },
    )

    with pytest.raises(SystemExit) as caught:
        sft_main()

    assert caught.value.code == 0
    result = json.loads((tmp_path / "out/harness_result.json").read_text())
    assert result["status"] == "OK"
    assert result["model_status"] == "MODEL_FAILURE"
    assert (
        json.loads((tmp_path / "agent/model-failure.json").read_text())["status"]
        == "MODEL_FAILURE"
    )


@pytest.mark.parametrize("layout", ["archive", "mounted_archive", "directory"])
def test_reward_reference_materialization_handles_arena_mount(tmp_path, layout):
    import zipfile

    from examples.blender_overfit.arena import materialize_references

    mount = tmp_path / "oracle"
    mount.mkdir()
    oracle = mount / "query-reference.zip"
    with zipfile.ZipFile(oracle, "w") as archive:
        archive.writestr("manifest.json", '{"tasks": []}')
        archive.writestr("tasks/ref.jpg", b"reference")
    if layout == "mounted_archive":
        oracle = mount
    elif layout == "directory":
        with zipfile.ZipFile(oracle) as archive:
            archive.extractall(mount)
        oracle.unlink()
        oracle = mount
    destination = tmp_path / "inputs"

    materialize_references(oracle, destination)

    assert json.loads((destination / "manifest.json").read_text()) == {"tasks": []}
    assert (destination / "tasks/ref.jpg").read_bytes() == b"reference"


@pytest.mark.parametrize(
    "status,expected", [("HARNESS_FAILED", None), ("EVAL_FAILED", None), ("DONE", 0)]
)
def test_platform_failure_zero_remains_missing_evaluation(tmp_path, status, expected):
    from examples.blender_overfit.control import Evaluation

    class API:
        def request(self, method, path):
            return {"status": status}

        def get_optional(self, path):
            return {"score": 0}

    operation = Evaluation.__new__(Evaluation)
    operation.api = API()
    operation.root = tmp_path
    operation.state_path = tmp_path / "state.json"
    operation.state = {
        "attempts": [
            {"task_id": "task", "job_id": "job", "source_task": "source", "mode": "sft"}
        ]
    }

    operation.status()

    assert operation.state["attempts"][0]["score"] == expected
    assert operation.state["attempts"][0]["platform_score"] == 0


def make_checkpoint(root: Path, step: int, complete: bool = True) -> Path:
    path = root / f"epoch0epochstep{step}globalstep{step}"
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    index = {"weight_map": {"weight": "model-00001-of-00001.safetensors"}}
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    data = len(header).to_bytes(8, "little") + header + (b"\0" * (4 if complete else 1))
    (path / "model-00001-of-00001.safetensors").write_bytes(data)
    return path


def make_record(image: bytes, description: str) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "Blender Python. Output ONLY one python block.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(image).decode(),
                        },
                    },
                    {
                        "type": "text",
                        "text": "Write Blender Python.\n\nThe object: " + description,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": f"```python\nSECRET_TEACHER = 1\nTARGET_IMAGE_ID = {image!r}\n```",
            },
        ]
    }


def test_checkpoint_partial_new_export_uses_latest_complete(tmp_path):
    """A newer directory with a truncated shard must never become the shared model."""
    previous = make_checkpoint(tmp_path, 10)
    make_checkpoint(tmp_path, 20, complete=False)

    assert latest_checkpoint(tmp_path) == previous.resolve()


def test_checkpoint_partially_written_index_keeps_previous_export(tmp_path):
    """Checkpoint discovery remains usable while the trainer is writing the next export."""
    previous = make_checkpoint(tmp_path, 10)
    new = make_checkpoint(tmp_path, 20)
    (new / "model.safetensors.index.json").write_text('{"weight_map":')

    assert latest_checkpoint(tmp_path) == previous.resolve()


def test_generator_identity_seed_variants_share_one_sampling_unit():
    """A change to the factory seed alone must not consume another object slot."""
    first = (
        '```python\nimport os\nfactory_seed = int(os.environ.get("SEED", "11"))\n```'
    )
    second = first.replace('"11"', '"22"')
    different = second.replace("factory_seed =", "different_geometry =")

    assert generator_identity(first) == generator_identity(second)
    assert generator_identity(first) != generator_identity(different)


@pytest.mark.parametrize(
    "reasoning_field", ["reasoning_content", "reasoning", "reasoning_details"]
)
def test_sft_serving_returns_thinking_is_infrastructure_failure(
    tmp_path, monkeypatch, reasoning_field
):
    """A server that ignores the thinking switch must not give the model a zero score."""
    from examples.blender_overfit.arena import sft_main

    query = tmp_path / "query.json"
    query.write_text(json.dumps(make_record(b"image", "A cube")["messages"][:-1]))
    agent_output = tmp_path / "agent"
    output = tmp_path / "out"
    for key, value in {
        "ARENA_AGENT_OUTPUT_DIR": agent_output,
        "ARENA_OUTPUT_DIR": output,
        "ARENA_QUERY_PATH": query,
    }.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(
        "examples.blender_overfit.single_turn.generate",
        lambda *args, **kwargs: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        reasoning_field: "unexpected thinking",
                        "content": "```python\nx = 1\n```",
                    },
                }
            ],
        },
    )

    with pytest.raises(SystemExit) as exc:
        sft_main()

    assert exc.value.code == 1
    assert (
        json.loads((output / "harness_result.json").read_text())["status"]
        == "INFRA_ERROR"
    )
    assert not (agent_output / "model-failure.json").exists()


def test_prepare_duplicate_images_retains_exact_sft_input_and_private_teacher(tmp_path):
    """Pair distinct objects and keep all prompt bytes and image order intact."""
    source = tmp_path / "source.jsonl"
    rows = [
        make_record(b"image-a", "a cube"),
        make_record(b"image-a", "a large cube"),
        make_record(b"image-b", "a sphere"),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    checkpoint = make_checkpoint(tmp_path / "checkpoints", 10)
    output = tmp_path / "evaluation"

    result = prepare(source, checkpoint.parent, output, seed=42, count=2)

    assert result["checkpoint"] == str(checkpoint)
    assert result["distinct_reference_image_groups"] == 2
    assert len({task["image_group_sha256"] for task in result["tasks"]}) == 2
    for task in result["tasks"]:
        messages = json.loads((output / task["sft_messages"]).read_text())
        assert messages == rows[task["source_row_index"]]["messages"][:-1]
        agent = json.loads((output / task["agent_messages"]).read_text())
        assert query_images(agent) == query_images(messages)
        query = (output / task["query"]).read_text()
        assert task["description"] in query
        assert "scene.blend" in query
        assert not any(
            word in query.lower() for word in ["thinking", "agentic", "SECRET_TEACHER"]
        )
    assert not any(
        "SECRET_TEACHER" in path.read_text()
        for path in (output / "tasks").rglob("*.json")
    )
    assert len(list((output / "private").glob("*/teacher-response.txt"))) == 2


def test_prepare_existing_manifest_refuses_resampling(tmp_path):
    """An existing test set cannot change accidentally between evaluation modes."""
    output = tmp_path / "frozen"
    output.mkdir()
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "source", tmp_path / "checkpoints", output, 42, 16)


@pytest.mark.parametrize(
    "completion",
    [
        "Here is the code\n```python\nx = 1\n```",
        "```python\nx = 1\n```\n```python\nx = 2\n```",
        "```python\ndef broken(:\n```",
    ],
)
def test_extract_code_extra_text_multiple_blocks_or_invalid_python_rejected(completion):
    """Malformed output is recorded as failure, never repaired by another model turn."""
    with pytest.raises((ValueError, SyntaxError)):
        extract_code(completion)


def test_session_headers_multiple_clients_remain_isolated():
    """Gateway session mappings and unrelated headers remain per request."""
    first = session_headers(
        "X-Session-Id: one\nX-Session-Token: token-one\nX-Keepalive: yes"
    )
    second = session_headers("X-Session-Id: two\nX-Session-Token: token-two")
    second["X-Session-Id"] = "modified"

    assert first == {
        "X-Session-Id": "one",
        "X-Session-Token": "token-one",
        "X-Keepalive": "yes",
    }


@pytest.mark.parametrize(
    "block",
    [
        "X-Session-Id: one",
        "Authorization: replacement",
        "X-Header: a\nx-header: b",
    ],
)
def test_session_headers_missing_pair_or_authentication_override_rejected(block):
    """Incomplete capabilities and authentication overrides fail before any request."""
    with pytest.raises(ValueError):
        session_headers(block)


def test_generate_one_call_forwards_exact_messages_and_disables_thinking(monkeypatch):
    """The complete SFT messages reach the model without tools or a second turn."""
    calls = []
    messages = make_record(b"reference", "a cube")["messages"][:-1]

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"choices": []}'

    def urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    generate(
        messages,
        model="pinned-model",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        temperature=1.0,
        max_tokens=100,
        seed=42,
        timeout=1,
    )

    assert len(calls) == 1
    assert calls[0]["messages"] == messages
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "tools" not in calls[0]


def test_prepare_mixed_engines_excludes_cadquery_and_threejs(tmp_path):
    """Only the user-selected Blender subset may supply the 16 seen objects."""
    source = tmp_path / "source.jsonl"
    blender = make_record(b"blender", "a cube")
    cad = make_record(b"cad", "a part")
    cad["messages"][0]["content"] = "CadQuery"
    three = make_record(b"html", "a scene")
    three["messages"][0]["content"] = "three.js"
    source.write_text("".join(json.dumps(row) + "\n" for row in [cad, blender, three]))
    checkpoint = make_checkpoint(tmp_path / "checkpoints", 1)

    result = prepare(source, checkpoint.parent, tmp_path / "dataset", 42, 1)

    assert result["source_rows"] == 3
    assert result["eligible_rows"] == 1
    assert result["tasks"][0]["source_row_index"] == 1
    assert result["source_engine_counts"] == {"cadquery": 1, "blender": 1, "threejs": 1}


def make_votes():
    features = detail_features("A door with a carved border.")
    qa = {
        "samples": {
            "sample": {
                "quality": {"score": 80},
                "aesthetics": {"score": 50},
                "zoom_requests": [],
            }
        }
    }
    detail = {
        "sample_id": "sample",
        "items": [
            {
                "id": features[0]["id"],
                "status": "rated",
                "score": 50,
                "observed_regions": [
                    {"region": "border", "candidate_fact": "Visible carving"}
                ],
            }
        ],
    }
    return qa, detail, features


def test_visual_score_renormalizes_original_visual_weights_only():
    """Removing geometry preserves the relative Q/D/A contribution."""
    qa, detail, features = make_votes()

    result = parse_votes(qa, detail, features)

    assert result["status"] == "OK"
    assert result["score"] == pytest.approx(0.62)
    for channel in ["shape", "structure"]:
        assert result["components"][channel] == {
            "value": None,
            "weight": 0,
            "status": "NOT_APPLICABLE",
        }


def test_visual_score_unknown_detail_keeps_null_score():
    """Unobserved detail is neither a zero nor a reason to change channel weights."""
    qa, detail, features = make_votes()
    detail["items"][0].update(status="unknown", score=None)

    result = parse_votes(qa, detail, features)

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["score"] is None
    assert result["components"]["detail"]["weight"] == 0.4


@pytest.mark.parametrize("score", [True, -1, 101, 0.5, "50"])
def test_visual_score_invalid_judge_value_rejected(score):
    """The original integer rating schema is enforced before aggregation."""
    qa, detail, features = make_votes()
    qa["samples"]["sample"]["quality"]["score"] = score
    with pytest.raises(ValueError):
        parse_votes(qa, detail, features)


def test_verifier_reference_tampering_fails_before_request(tmp_path):
    """The grader must read exactly the same four images given to both candidates."""
    from examples.blender_overfit.prepare import digest

    image = tmp_path / "reference.jpg"
    image.write_bytes(b"changed")
    task = {
        "description": "A cube.",
        "visual_features": detail_features("A cube."),
        "reference_images": [
            {"path": image.name, "sha256": digest(b"original")} for _ in range(4)
        ],
    }
    with pytest.raises(ValueError, match="query image changed"):
        request_body(tmp_path, task, tmp_path / "evidence", "qa", "test-model")


def test_build_agent_package_excludes_grader_and_private_teacher(tmp_path):
    """The agent and reward see separate code/data packages in both evaluation modes."""
    import zipfile

    from examples.blender_overfit.build import build

    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(make_record(b"reference", "A carved cube.")) + "\n")
    checkpoint = make_checkpoint(tmp_path / "checkpoints", 10)
    dataset = tmp_path / "dataset"
    prepare(source, checkpoint.parent, dataset, 42, 1)
    env = tmp_path / "env.json"
    env.write_text(
        json.dumps(
            {
                "spec": {
                    "runtime": {
                        "kind": "image",
                        "image": "example.invalid/runtime:fixture",
                    },
                    "files": [
                        {"uri": "fixture", "dest": "native/task-policy.json"},
                        {"uri": "fixture", "dest": "native/task-skills/fine-building"},
                    ],
                    "hooks": {
                        "setup_script": 'python3 "$ARENA_ENV_DIR/official/collector.pyz" --setup'
                    },
                }
            }
        )
    )
    output = tmp_path / "materials"

    build(dataset, env, output)

    assert (
        (output / "blender-overfit-agent.pyz")
        .read_bytes()
        .startswith(b"#!/usr/bin/env python3\n")
    )
    with zipfile.ZipFile(output / "blender-overfit-agent.pyz") as archive:
        assert not any(
            "prompts/" in name or name.endswith("verifier.py")
            for name in archive.namelist()
        )
    with zipfile.ZipFile(output / "blender-overfit-reward.pyz") as archive:
        assert "examples/blender_overfit/prompts/qa.txt" in archive.namelist()
    for path in (output / "tasks").glob("*/query-reference.zip"):
        with zipfile.ZipFile(path) as archive:
            assert not any("teacher" in name for name in archive.namelist())
            assert b"SECRET_TEACHER" not in archive.read("manifest.json")
    blueprints = json.loads((output / "env-blueprints.json").read_text())
    assert len(blueprints) == 2
    assert all(
        not any(
            "task-policy" in item["dest"] or "task-skills" in item["dest"]
            for item in env["spec"]["files"]
        )
        for env in blueprints
    )
