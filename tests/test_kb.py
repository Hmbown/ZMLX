"""Tests for canonical knowledge base build and validation."""

from __future__ import annotations

import copy

import pytest

from zmlx.kb.builder import build_knowledge_base
from zmlx.kb.schema import (
    KBValidationError,
    canonical_dumps,
    load_knowledge_base,
    validate_knowledge_base,
)


@pytest.fixture(scope="module")
def kb_payload() -> dict:
    payload = build_knowledge_base()
    validate_knowledge_base(payload)
    return payload


def test_kb_build_has_expected_top_level_shape(kb_payload: dict) -> None:
    assert set(kb_payload.keys()) == {"metadata", "models", "definitions"}
    assert isinstance(kb_payload["models"], list)
    assert isinstance(kb_payload["definitions"], dict)
    assert isinstance(kb_payload["definitions"]["patterns"], list)
    assert isinstance(kb_payload["definitions"]["discovered_kernels"], list)


def test_kb_build_is_deterministic_bytes() -> None:
    payload_a = build_knowledge_base()
    payload_b = build_knowledge_base()
    assert canonical_dumps(payload_a) == canonical_dumps(payload_b)


def test_kb_artifact_matches_builder_output() -> None:
    artifact = load_knowledge_base("zmlx_knowledge_base.json")
    rebuilt = build_knowledge_base()
    assert canonical_dumps(artifact) == canonical_dumps(rebuilt)


def test_kb_pattern_policy_references_canonical_ids(kb_payload: dict) -> None:
    pattern_ids = {entry["id"] for entry in kb_payload["definitions"]["patterns"]}
    for model in kb_payload["models"]:
        expected = model["pattern_policy"]["expected"]
        excluded = model["pattern_policy"]["excluded"]
        for pattern_id in expected:
            assert pattern_id.startswith("pattern:")
            assert pattern_id in pattern_ids
        for pattern_id in excluded:
            assert pattern_id.startswith("pattern:")
            assert pattern_id in pattern_ids


def test_kb_specialized_kernel_refs_are_resolvable(kb_payload: dict) -> None:
    definition_ids = {
        *[entry["id"] for entry in kb_payload["definitions"]["patterns"]],
        *[entry["id"] for entry in kb_payload["definitions"]["discovered_kernels"]],
    }
    for model in kb_payload["models"]:
        for ref in model["specialized_kernels"]:
            assert ref["kernel_id"] in definition_ids


def test_validate_fails_on_duplicate_definition_id(kb_payload: dict) -> None:
    broken = copy.deepcopy(kb_payload)
    duplicate = copy.deepcopy(broken["definitions"]["patterns"][0])
    broken["definitions"]["patterns"].append(duplicate)
    with pytest.raises(KBValidationError, match="Duplicate definition id"):
        validate_knowledge_base(broken)


def test_validate_fails_on_broken_model_reference(kb_payload: dict) -> None:
    broken = copy.deepcopy(kb_payload)
    broken["models"][0]["specialized_kernels"][0]["kernel_id"] = "kernel:does_not_exist"
    with pytest.raises(KBValidationError, match="references unknown id"):
        validate_knowledge_base(broken)


def test_validate_fails_on_missing_required_field(kb_payload: dict) -> None:
    broken = copy.deepcopy(kb_payload)
    broken["definitions"]["patterns"][0].pop("description", None)
    with pytest.raises(KBValidationError, match="missing keys"):
        validate_knowledge_base(broken)


def test_validate_requires_missing_reason_for_null_field(kb_payload: dict) -> None:
    broken = copy.deepcopy(kb_payload)
    entry = broken["definitions"]["patterns"][0]
    entry["math_formula"] = None
    entry["missing_reason"].pop("math_formula", None)
    with pytest.raises(KBValidationError, match="math_formula is null"):
        validate_knowledge_base(broken)
