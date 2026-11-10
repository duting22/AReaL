"""Verifier-defined reward observations and display names in training metrics."""

import pytest

from examples.swe.arena_reward_breakdown import SCHEMA, arena_reward_metrics


def test_reward_observation_v1_exports_supplied_component_fields_and_diagnostics():
    """Weights, contributions and per-result shares retain their distinct values."""
    observation = {
        "schema_version": 1,
        "verifier_version": "example-v1",
        "components": {
            "quality": {
                "value": 0.8,
                "weight": 0.9,
                "contribution": 0.72,
                "share": 0.87804878,
            },
            "format": {
                "value": 1.0,
                "weight": 0.1,
                "contribution": 0.1,
                "share": 0.12195122,
            },
        },
        "diagnostics": {"output_length": 1234},
    }

    metrics = arena_reward_metrics(
        sub_scores=None, raw={"reward_observation": observation}, score=0.82
    )

    assert metrics["reward_observation/present"] == 1.0
    assert metrics["reward_observation/valid"] == 1.0
    assert metrics["reward_observation/invalid_entries"] == 0.0
    for name, component in observation["components"].items():
        for field, value in component.items():
            assert metrics[f"reward_observation/components/{name}/{field}"] == value
    assert metrics["reward_observation/diagnostics/output_length"] == 1234.0
    assert "arena_score" not in metrics


@pytest.mark.parametrize("score", [None, 0.0, 0.82])
def test_reward_observation_missing_values_are_omitted_without_rescoring(score):
    """Nonlinear contributions and zero-total shares must not be fabricated."""
    raw = {
        "reward_observation": {
            "schema_version": 1,
            "components": {
                "structure/arch": {
                    "value": 0.8,
                    "weight": 0.9,
                    "contribution": None,
                    "share": None,
                },
                "unknown": {"value": None},
            },
            "diagnostics": {"unknown": None},
        }
    }

    metrics = arena_reward_metrics(sub_scores=None, raw=raw, score=score)

    assert metrics["reward_observation/valid"] == 1.0
    assert metrics["reward_observation/components/structure/arch/value"] == 0.8
    assert metrics["reward_observation/components/structure/arch/weight"] == 0.9
    assert not any(key.endswith(("/contribution", "/share")) for key in metrics)
    assert not any("/unknown/" in key or key.endswith("/unknown") for key in metrics)


def test_reward_observation_unicode_names_and_invalid_fields_are_independent():
    """A malformed optional measurement cannot suppress the remaining metrics."""
    raw = {
        "reward_observation": {
            "schema_version": 1,
            "components": {
                "主体形状": {"value": 0.8, "weight": True, "share": float("nan")},
                "shape/view1": {"value": 0.9, "contribution": "0.2"},
                "bad\nname": {"value": 0.1},
                "bad_component": [],
            },
            "diagnostics": {"输出长度": 1234, "invalid": float("inf")},
        }
    }

    metrics = arena_reward_metrics(sub_scores=None, raw=raw, score=0.82)

    assert metrics["reward_observation/components/主体形状/value"] == 0.8
    assert metrics["reward_observation/components/shape/view1/value"] == 0.9
    assert metrics["reward_observation/diagnostics/输出长度"] == 1234.0
    assert metrics["reward_observation/invalid_entries"] == 6.0
    assert "reward_observation/components/主体形状/weight" not in metrics


@pytest.mark.parametrize(
    "observation",
    [
        [],
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": 1, "components": []},
        {"schema_version": 1, "diagnostics": []},
        {"schema_version": 1, "components": {str(i): {} for i in range(257)}},
        {"schema_version": 1, "diagnostics": {str(i): 0 for i in range(257)}},
    ],
)
def test_invalid_reward_observation_preserves_legacy_telemetry(observation):
    """Unsupported or oversized observations remain nonfatal optional telemetry."""
    raw = {
        "reward_observation": observation,
        "reward_breakdown": {
            "schema": SCHEMA,
            "score_mode": "HARD_INVALID",
            "root": {"name": "total", "operator": "terminal", "value": 0.0},
        },
    }

    metrics = arena_reward_metrics(sub_scores={"quality": 0.25}, raw=raw, score=0.0)

    assert metrics["reward_sub_score/quality"] == 0.25
    assert metrics["reward_breakdown/valid"] == 1.0
    assert metrics["reward_observation/valid"] == 0.0
    assert metrics["reward_observation/invalid_entries"] == 1.0


@pytest.mark.parametrize("raw", [None, [], "unstructured", {}])
def test_missing_reward_observation_preserves_legacy_sub_scores(raw):
    """Opaque raw payloads and graders without observations remain supported."""
    metrics = arena_reward_metrics(sub_scores={"quality": 0.25}, raw=raw, score=0.5)

    assert metrics["reward_sub_score/quality"] == 0.25
    assert metrics["reward_observation/present"] == 0.0
    assert metrics["reward_observation/invalid_entries"] == 0.0


def test_sub_scores_chinese_display_names_resolve_to_declared_component_paths():
    """The fine115 display contract must not lose Chinese labels or nested IDs."""
    shape = "总项｜主体形状（占总分22.5%）"
    view = "轮廓细项｜整体轮廓视角1（占总分5.625%）"
    raw = {"reward_display": {"component_ids": {shape: "shape", view: "shape/view1"}}}

    metrics = arena_reward_metrics(
        sub_scores={shape: 0.8, view: 0.9}, raw=raw, score=0.5
    )

    assert metrics["reward_sub_score/shape"] == 0.8
    assert metrics["reward_sub_score/shape/view1"] == 0.9
    assert metrics["reward_sub_score/invalid_entries"] == 0.0


@pytest.mark.parametrize("display", [None, [], {"component_ids": []}])
def test_sub_scores_without_display_mapping_preserve_unicode_names(display):
    """Unicode display names also work when a grader provides no ID mapping."""
    name = "总项｜主体形状（占总分22.5%）"

    metrics = arena_reward_metrics(
        sub_scores={name: 0.8}, raw={"reward_display": display}, score=0.5
    )

    assert metrics[f"reward_sub_score/{name}"] == 0.8
    assert metrics["reward_sub_score/invalid_entries"] == 0.0


@pytest.mark.parametrize("name", ["", "bad\nname", "../escape", "a//b", "x" * 257])
def test_sub_scores_invalid_names_are_counted_without_dropping_valid_metrics(name):
    """Names remain bounded even though ASCII-only validation is removed."""
    metrics = arena_reward_metrics(
        sub_scores={name: 0.1, "quality": 0.25}, raw=None, score=0.5
    )

    assert metrics["reward_sub_score/quality"] == 0.25
    assert metrics["reward_sub_score/invalid_entries"] == 1.0


def test_sub_scores_duplicate_mapped_ids_do_not_silently_overwrite_values():
    """Conflicting aliases are observable instead of changing a recorded score."""
    metrics = arena_reward_metrics(
        sub_scores={"quality": 0.8, "质量": 0.2, "invalid_entries": 99},
        raw={"reward_display": {"component_ids": {"质量": "quality"}}},
        score=0.5,
    )

    assert metrics["reward_sub_score/quality"] == 0.8
    assert metrics["reward_sub_score/invalid_entries"] == 2.0
