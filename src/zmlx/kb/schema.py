"""Canonical schema and validators for ``zmlx_knowledge_base.json``."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

KB_SCHEMA_VERSION = "2.0"
KB_FILE_NAME = "zmlx_knowledge_base.json"

DEFINITION_NULLABLE_FIELDS = (
    "math_formula",
    "implementation_notes",
    "source_symbol",
    "constraints",
    "env_flags",
    "compatibility",
    "evidence",
)

DEFINITION_REQUIRED_FIELDS = (
    "id",
    "name",
    "description",
    "math_formula",
    "implementation_notes",
    "source_path",
    "source_symbol",
    "constraints",
    "env_flags",
    "compatibility",
    "evidence",
    "missing_reason",
)

MODEL_KERNEL_NULLABLE_FIELDS = (
    "constraints",
    "evidence",
)

MODEL_KERNEL_REQUIRED_FIELDS = (
    "kernel_id",
    "applicability",
    "rationale",
    "constraints",
    "evidence",
    "missing_reason",
)

MODEL_REQUIRED_FIELDS = (
    "id",
    "display_name",
    "family",
    "zmlx_family",
    "architecture",
    "quant",
    "params",
    "storage_gb",
    "pattern_policy",
    "specialized_kernels",
)

KB_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ZMLX Knowledge Base",
    "type": "object",
    "additionalProperties": False,
    "required": ["metadata", "models", "definitions"],
    "properties": {
        "metadata": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema_version",
                "description",
                "generated_from",
                "generator",
                "deterministic",
                "model_count",
                "definition_count",
            ],
            "properties": {
                "schema_version": {"type": "string"},
                "description": {"type": "string"},
                "generated_from": {"type": "string"},
                "generator": {"type": "string"},
                "deterministic": {"type": "boolean"},
                "model_count": {"type": "integer"},
                "definition_count": {"type": "integer"},
            },
        },
        "models": {
            "type": "array",
            "items": {"$ref": "#/$defs/modelEntry"},
        },
        "definitions": {
            "type": "object",
            "additionalProperties": False,
            "required": ["patterns", "discovered_kernels"],
            "properties": {
                "patterns": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/definitionEntry"},
                },
                "discovered_kernels": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/definitionEntry"},
                },
            },
        },
    },
    "$defs": {
        "evidenceEntry": {
            "type": "object",
            "additionalProperties": False,
            "required": ["source_path", "source_symbol", "kind", "snippet"],
            "properties": {
                "source_path": {"type": "string"},
                "source_symbol": {"type": "string"},
                "kind": {"type": "string"},
                "snippet": {"type": "string"},
            },
        },
        "definitionEntry": {
            "type": "object",
            "additionalProperties": False,
            "required": list(DEFINITION_REQUIRED_FIELDS),
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "math_formula": {"type": ["string", "null"]},
                "implementation_notes": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "source_path": {"type": "string"},
                "source_symbol": {"type": ["string", "null"]},
                "constraints": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "env_flags": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "compatibility": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "evidence": {
                    "type": ["array", "null"],
                    "items": {"$ref": "#/$defs/evidenceEntry"},
                },
                "missing_reason": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
        },
        "modelKernelRef": {
            "type": "object",
            "additionalProperties": False,
            "required": list(MODEL_KERNEL_REQUIRED_FIELDS),
            "properties": {
                "kernel_id": {"type": "string"},
                "applicability": {"type": "string"},
                "rationale": {"type": "string"},
                "constraints": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "evidence": {
                    "type": ["array", "null"],
                    "items": {"$ref": "#/$defs/evidenceEntry"},
                },
                "missing_reason": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
        },
        "modelEntry": {
            "type": "object",
            "additionalProperties": False,
            "required": list(MODEL_REQUIRED_FIELDS),
            "properties": {
                "id": {"type": "string"},
                "display_name": {"type": "string"},
                "family": {"type": "string"},
                "zmlx_family": {"type": "string"},
                "architecture": {"type": "string"},
                "quant": {"type": "string"},
                "params": {"type": "string"},
                "storage_gb": {"type": "number"},
                "pattern_policy": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["expected", "excluded"],
                    "properties": {
                        "expected": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "excluded": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                },
                "specialized_kernels": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/modelKernelRef"},
                },
            },
        },
    },
}


class KBValidationError(ValueError):
    """Raised when KB payload validation fails."""


def canonical_dumps(data: Mapping[str, Any]) -> str:
    """Serialize JSON in a deterministic, byte-stable way."""
    return json.dumps(data, indent=2, ensure_ascii=True, sort_keys=True) + "\n"


def load_knowledge_base(path: str | Path) -> dict[str, Any]:
    """Load a knowledge base JSON file."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise KBValidationError("Top-level JSON value must be an object.")
    return data


def write_knowledge_base(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a KB JSON file with canonical deterministic formatting."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(canonical_dumps(data))


def coverage_report(data: Mapping[str, Any]) -> dict[str, int]:
    """Coverage report for formula and implementation-note population."""
    definitions = _all_definition_entries(data)
    total = len(definitions)
    non_null_formula = sum(1 for entry in definitions if _is_non_empty_str(entry.get("math_formula")))
    non_empty_notes = sum(
        1
        for entry in definitions
        if isinstance(entry.get("implementation_notes"), list) and bool(entry["implementation_notes"])
    )
    return {
        "definitions_total": total,
        "math_formula_non_null": non_null_formula,
        "implementation_notes_non_empty": non_empty_notes,
    }


def unresolved_gaps(data: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return explicit unresolved gaps with field-level reasons."""
    gaps: list[dict[str, str]] = []

    for entry in _all_definition_entries(data):
        entry_id = str(entry.get("id", "<missing-id>"))
        missing = entry.get("missing_reason")
        if not isinstance(missing, dict):
            continue
        for field in sorted(missing):
            reason = missing.get(field)
            if isinstance(reason, str) and reason.strip():
                gaps.append(
                    {
                        "scope": "definition",
                        "id": entry_id,
                        "field": field,
                        "reason": reason.strip(),
                    }
                )

    for model in data.get("models", []) if isinstance(data.get("models"), list) else []:
        model_id = str(model.get("id", "<missing-model-id>"))
        refs = model.get("specialized_kernels", [])
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            kernel_id = str(ref.get("kernel_id", "<missing-kernel-id>"))
            missing = ref.get("missing_reason")
            if not isinstance(missing, dict):
                continue
            for field in sorted(missing):
                reason = missing.get(field)
                if isinstance(reason, str) and reason.strip():
                    gaps.append(
                        {
                            "scope": "model_kernel_ref",
                            "id": f"{model_id}::{kernel_id}",
                            "field": field,
                            "reason": reason.strip(),
                        }
                    )

    gaps.sort(key=lambda item: (item["scope"], item["id"], item["field"]))
    return gaps


def validate_knowledge_base(data: Mapping[str, Any]) -> None:
    """Validate the KB payload and raise on schema, reference, or consistency drift."""
    errors: list[str] = []

    if not isinstance(data, Mapping):
        raise KBValidationError("Top-level value must be an object.")

    _validate_top_level(data, errors)

    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        _validate_metadata(metadata, errors)

    definition_ids: set[str] = set()
    pattern_ids: set[str] = set()
    definitions = data.get("definitions")
    if isinstance(definitions, Mapping):
        _validate_definition_bucket(
            definitions.get("patterns"),
            "definitions.patterns",
            errors,
            definition_ids,
            pattern_ids,
            require_pattern_prefix=True,
        )
        _validate_definition_bucket(
            definitions.get("discovered_kernels"),
            "definitions.discovered_kernels",
            errors,
            definition_ids,
            pattern_ids,
            require_pattern_prefix=False,
        )

    model_ids: set[str] = set()
    model_count = 0
    models = data.get("models")
    if not isinstance(models, list):
        errors.append("models must be an array.")
    else:
        model_count = len(models)
        for index, model in enumerate(models):
            path = f"models[{index}]"
            if not isinstance(model, Mapping):
                errors.append(f"{path} must be an object.")
                continue
            _validate_model_entry(model, path, errors, model_ids, definition_ids, pattern_ids)

    if isinstance(metadata, Mapping):
        expected_model_count = metadata.get("model_count")
        expected_definition_count = metadata.get("definition_count")
        if isinstance(expected_model_count, int) and expected_model_count != model_count:
            errors.append(
                "metadata.model_count mismatch: "
                f"expected {expected_model_count}, actual {model_count}."
            )
        if (
            isinstance(expected_definition_count, int)
            and expected_definition_count != len(definition_ids)
        ):
            errors.append(
                "metadata.definition_count mismatch: "
                f"expected {expected_definition_count}, actual {len(definition_ids)}."
            )

    if errors:
        raise KBValidationError("\n".join(errors))


def _validate_top_level(data: Mapping[str, Any], errors: list[str]) -> None:
    expected_keys = {"metadata", "models", "definitions"}
    actual_keys = set(data.keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
    if extra:
        errors.append(f"Unexpected top-level keys: {sorted(extra)}")


def _validate_metadata(metadata: Mapping[str, Any], errors: list[str]) -> None:
    required = {
        "schema_version",
        "description",
        "generated_from",
        "generator",
        "deterministic",
        "model_count",
        "definition_count",
    }
    missing = required - set(metadata.keys())
    extra = set(metadata.keys()) - required
    if missing:
        errors.append(f"metadata missing keys: {sorted(missing)}")
    if extra:
        errors.append(f"metadata has unexpected keys: {sorted(extra)}")

    schema_version = metadata.get("schema_version")
    if schema_version != KB_SCHEMA_VERSION:
        errors.append(
            f"metadata.schema_version must be {KB_SCHEMA_VERSION!r}, got {schema_version!r}."
        )

    if not _is_non_empty_str(metadata.get("description")):
        errors.append("metadata.description must be a non-empty string.")
    if not _is_non_empty_str(metadata.get("generated_from")):
        errors.append("metadata.generated_from must be a non-empty string.")
    if not _is_non_empty_str(metadata.get("generator")):
        errors.append("metadata.generator must be a non-empty string.")
    if not isinstance(metadata.get("deterministic"), bool):
        errors.append("metadata.deterministic must be boolean.")
    if not isinstance(metadata.get("model_count"), int):
        errors.append("metadata.model_count must be an integer.")
    if not isinstance(metadata.get("definition_count"), int):
        errors.append("metadata.definition_count must be an integer.")


def _validate_definition_bucket(
    entries: Any,
    path: str,
    errors: list[str],
    definition_ids: set[str],
    pattern_ids: set[str],
    *,
    require_pattern_prefix: bool,
) -> None:
    if not isinstance(entries, list):
        errors.append(f"{path} must be an array.")
        return

    for index, entry in enumerate(entries):
        entry_path = f"{path}[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{entry_path} must be an object.")
            continue

        _validate_required_keys(entry, DEFINITION_REQUIRED_FIELDS, entry_path, errors)
        entry_id = entry.get("id")
        if not _is_non_empty_str(entry_id):
            errors.append(f"{entry_path}.id must be a non-empty string.")
            continue

        assert isinstance(entry_id, str)
        if require_pattern_prefix and not entry_id.startswith("pattern:"):
            errors.append(f"{entry_path}.id must start with 'pattern:'.")
        if not require_pattern_prefix and not entry_id.startswith("kernel:"):
            errors.append(f"{entry_path}.id must start with 'kernel:'.")

        if entry_id in definition_ids:
            errors.append(f"Duplicate definition id detected: {entry_id!r}.")
        definition_ids.add(entry_id)
        if require_pattern_prefix:
            pattern_ids.add(entry_id)

        if not _is_non_empty_str(entry.get("name")):
            errors.append(f"{entry_path}.name must be a non-empty string.")
        if not _is_non_empty_str(entry.get("description")):
            errors.append(f"{entry_path}.description must be a non-empty string.")

        source_path = entry.get("source_path")
        if not _is_non_empty_str(source_path):
            errors.append(f"{entry_path}.source_path must be a non-empty string.")
        elif Path(str(source_path)).is_absolute():
            errors.append(f"{entry_path}.source_path must be repository-relative, not absolute.")

        _validate_nullable_string(entry.get("math_formula"), f"{entry_path}.math_formula", errors)
        _validate_nullable_string(entry.get("source_symbol"), f"{entry_path}.source_symbol", errors)
        _validate_nullable_str_list(
            entry.get("implementation_notes"), f"{entry_path}.implementation_notes", errors
        )
        _validate_nullable_str_list(entry.get("constraints"), f"{entry_path}.constraints", errors)
        _validate_nullable_str_list(entry.get("env_flags"), f"{entry_path}.env_flags", errors)
        _validate_nullable_str_list(entry.get("compatibility"), f"{entry_path}.compatibility", errors)
        _validate_nullable_evidence(entry.get("evidence"), f"{entry_path}.evidence", errors)

        missing_reason = entry.get("missing_reason")
        if not isinstance(missing_reason, Mapping):
            errors.append(f"{entry_path}.missing_reason must be an object.")
        else:
            _validate_missing_reason_object(
                entry,
                missing_reason,
                DEFINITION_NULLABLE_FIELDS,
                entry_path,
                errors,
            )


def _validate_model_entry(
    model: Mapping[str, Any],
    path: str,
    errors: list[str],
    model_ids: set[str],
    definition_ids: set[str],
    pattern_ids: set[str],
) -> None:
    _validate_required_keys(model, MODEL_REQUIRED_FIELDS, path, errors)

    model_id = model.get("id")
    if not _is_non_empty_str(model_id):
        errors.append(f"{path}.id must be a non-empty string.")
    else:
        assert isinstance(model_id, str)
        if model_id in model_ids:
            errors.append(f"Duplicate model id detected: {model_id!r}.")
        model_ids.add(model_id)

    for key in (
        "display_name",
        "family",
        "zmlx_family",
        "architecture",
        "quant",
        "params",
    ):
        if not _is_non_empty_str(model.get(key)):
            errors.append(f"{path}.{key} must be a non-empty string.")

    storage_gb = model.get("storage_gb")
    if not isinstance(storage_gb, (int, float)):
        errors.append(f"{path}.storage_gb must be numeric.")

    pattern_policy = model.get("pattern_policy")
    if not isinstance(pattern_policy, Mapping):
        errors.append(f"{path}.pattern_policy must be an object.")
    else:
        _validate_required_keys(pattern_policy, ("expected", "excluded"), f"{path}.pattern_policy", errors)
        expected = pattern_policy.get("expected")
        excluded = pattern_policy.get("excluded")
        if not isinstance(expected, list):
            errors.append(f"{path}.pattern_policy.expected must be an array.")
        else:
            seen_expected: set[str] = set()
            for index, kernel_id in enumerate(expected):
                item_path = f"{path}.pattern_policy.expected[{index}]"
                if not _is_non_empty_str(kernel_id):
                    errors.append(f"{item_path} must be a non-empty string.")
                    continue
                assert isinstance(kernel_id, str)
                if kernel_id in seen_expected:
                    errors.append(f"{item_path} duplicates kernel id {kernel_id!r}.")
                seen_expected.add(kernel_id)
                if kernel_id not in pattern_ids:
                    errors.append(f"{item_path} references unknown pattern id {kernel_id!r}.")

        if not isinstance(excluded, Mapping):
            errors.append(f"{path}.pattern_policy.excluded must be an object.")
        else:
            for kernel_id, reason in excluded.items():
                key_path = f"{path}.pattern_policy.excluded[{kernel_id!r}]"
                if not _is_non_empty_str(kernel_id):
                    errors.append(f"{key_path} key must be a non-empty string.")
                    continue
                if kernel_id not in pattern_ids:
                    errors.append(f"{key_path} references unknown pattern id {kernel_id!r}.")
                if not _is_non_empty_str(reason):
                    errors.append(f"{key_path} reason must be a non-empty string.")

    specialized = model.get("specialized_kernels")
    if not isinstance(specialized, list):
        errors.append(f"{path}.specialized_kernels must be an array.")
        return

    seen_kernel_refs: set[str] = set()
    for index, entry in enumerate(specialized):
        entry_path = f"{path}.specialized_kernels[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{entry_path} must be an object.")
            continue
        _validate_required_keys(entry, MODEL_KERNEL_REQUIRED_FIELDS, entry_path, errors)

        kernel_id = entry.get("kernel_id")
        if not _is_non_empty_str(kernel_id):
            errors.append(f"{entry_path}.kernel_id must be a non-empty string.")
        else:
            assert isinstance(kernel_id, str)
            if kernel_id not in definition_ids:
                errors.append(f"{entry_path}.kernel_id references unknown id {kernel_id!r}.")
            if kernel_id in seen_kernel_refs:
                errors.append(f"{entry_path}.kernel_id duplicates {kernel_id!r} within model.")
            seen_kernel_refs.add(kernel_id)

        if not _is_non_empty_str(entry.get("applicability")):
            errors.append(f"{entry_path}.applicability must be a non-empty string.")
        if not _is_non_empty_str(entry.get("rationale")):
            errors.append(f"{entry_path}.rationale must be a non-empty string.")

        _validate_nullable_str_list(entry.get("constraints"), f"{entry_path}.constraints", errors)
        _validate_nullable_evidence(entry.get("evidence"), f"{entry_path}.evidence", errors)

        missing_reason = entry.get("missing_reason")
        if not isinstance(missing_reason, Mapping):
            errors.append(f"{entry_path}.missing_reason must be an object.")
        else:
            _validate_missing_reason_object(
                entry,
                missing_reason,
                MODEL_KERNEL_NULLABLE_FIELDS,
                entry_path,
                errors,
            )


def _validate_required_keys(
    payload: Mapping[str, Any],
    required: tuple[str, ...],
    path: str,
    errors: list[str],
) -> None:
    missing = set(required) - set(payload.keys())
    extra = set(payload.keys()) - set(required)
    if missing:
        errors.append(f"{path} missing keys: {sorted(missing)}")
    if extra:
        errors.append(f"{path} has unexpected keys: {sorted(extra)}")


def _validate_missing_reason_object(
    payload: Mapping[str, Any],
    missing_reason: Mapping[str, Any],
    nullable_fields: tuple[str, ...],
    path: str,
    errors: list[str],
) -> None:
    for key, reason in missing_reason.items():
        if key not in nullable_fields:
            errors.append(f"{path}.missing_reason has unknown field {key!r}.")
        if not _is_non_empty_str(reason):
            errors.append(f"{path}.missing_reason[{key!r}] must be a non-empty string.")

    for field in nullable_fields:
        value = payload.get(field)
        has_reason = field in missing_reason
        if value is None and not has_reason:
            errors.append(f"{path}.{field} is null but missing_reason has no entry for it.")
        if value is not None and has_reason:
            errors.append(f"{path}.{field} is non-null but missing_reason includes it.")


def _validate_nullable_string(value: Any, path: str, errors: list[str]) -> None:
    if value is None:
        return
    if not _is_non_empty_str(value):
        errors.append(f"{path} must be null or a non-empty string.")


def _validate_nullable_str_list(value: Any, path: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        errors.append(f"{path} must be null or an array of non-empty strings.")
        return
    if not value:
        errors.append(f"{path} must not be an empty array; use null + missing_reason.")
        return
    for index, item in enumerate(value):
        if not _is_non_empty_str(item):
            errors.append(f"{path}[{index}] must be a non-empty string.")


def _validate_nullable_evidence(value: Any, path: str, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        errors.append(f"{path} must be null or an array of evidence entries.")
        return
    if not value:
        errors.append(f"{path} must not be an empty array; use null + missing_reason.")
        return
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{item_path} must be an object.")
            continue
        required = {"source_path", "source_symbol", "kind", "snippet"}
        missing = required - set(item.keys())
        extra = set(item.keys()) - required
        if missing:
            errors.append(f"{item_path} missing keys: {sorted(missing)}")
        if extra:
            errors.append(f"{item_path} has unexpected keys: {sorted(extra)}")
        if not _is_non_empty_str(item.get("source_path")):
            errors.append(f"{item_path}.source_path must be a non-empty string.")
        elif Path(str(item.get("source_path"))).is_absolute():
            errors.append(f"{item_path}.source_path must be repository-relative.")
        if not _is_non_empty_str(item.get("source_symbol")):
            errors.append(f"{item_path}.source_symbol must be a non-empty string.")
        if not _is_non_empty_str(item.get("kind")):
            errors.append(f"{item_path}.kind must be a non-empty string.")
        if not _is_non_empty_str(item.get("snippet")):
            errors.append(f"{item_path}.snippet must be a non-empty string.")


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _all_definition_entries(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    definitions = data.get("definitions")
    if not isinstance(definitions, Mapping):
        return []
    entries: list[Mapping[str, Any]] = []
    for key in ("patterns", "discovered_kernels"):
        bucket = definitions.get(key)
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if isinstance(item, Mapping):
                entries.append(item)
    return entries


__all__ = [
    "KB_FILE_NAME",
    "KB_JSON_SCHEMA",
    "KB_SCHEMA_VERSION",
    "KBValidationError",
    "canonical_dumps",
    "coverage_report",
    "load_knowledge_base",
    "unresolved_gaps",
    "validate_knowledge_base",
    "write_knowledge_base",
]
