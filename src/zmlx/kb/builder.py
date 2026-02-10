"""Deterministic source-driven builder for ``zmlx_knowledge_base.json``."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from zmlx.matrix.models import load_catalog

from .schema import DEFINITION_NULLABLE_FIELDS, KB_SCHEMA_VERSION, validate_knowledge_base

_PATTERNS_DIR = Path("src/zmlx/patch/patterns")
_DISCOVERED_DIR = Path("src/zmlx/kernels/discovered")
_MODELS_POLICY_SOURCE = "src/zmlx/matrix/models.py"

_ENV_FLAG_RE = re.compile(r"\bZMLX_[A-Z0-9_]+\b")

_FORMULA_HINT_RE = re.compile(
    r"(=|->|\bsilu\b|\bgelu\b|\bsigmoid\b|\bsoftmax\b|\brsqrt\b|\bnormalize\b|\btop-?k\b|\bargpartition\b|\bconcat\b|\bsum\b)",
    re.IGNORECASE,
)

_CONSTRAINT_HINT_RE = re.compile(
    r"(\bonly\b|\brequire|\bsupport|\bfallback\b|\bopt-in\b|\bdecode\b|\bprefill\b|\bbits?\b|\bgroup\b|\bshape\b|\bwhen\b|\bD\s*=\s*\d+|\bK\s*=\s*\d+)",
    re.IGNORECASE,
)

_FAMILY_HINTS: dict[str, str] = {
    "glm": "model_family:glm",
    "qwen": "model_family:qwen",
    "deepseek": "model_family:deepseek",
    "kimi": "model_family:kimi",
    "mixtral": "model_family:mixtral",
    "gpt_oss": "model_family:gpt_oss",
    "gpt-oss": "model_family:gpt_oss",
    "lfm": "model_family:lfm",
    "llama": "model_family:llama",
    "gemma": "model_family:gemma",
    "phi": "model_family:phi",
    "mistral": "model_family:mistral",
    "nemotron": "model_family:nemotron",
}


def build_knowledge_base(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Build the canonical knowledge base payload from repository sources."""
    root = _resolve_repo_root(repo_root)

    pattern_entries = _build_pattern_entries(root)
    discovered_entries = _build_discovered_entries(root)
    definition_lookup = {entry["id"]: entry for entry in [*pattern_entries, *discovered_entries]}

    models = _build_model_entries(definition_lookup)

    payload: dict[str, Any] = {
        "metadata": {
            "schema_version": KB_SCHEMA_VERSION,
            "description": "ZMLX model catalog and kernel knowledge base.",
            "generated_from": "src/zmlx",
            "generator": "python -m zmlx.kb build",
            "deterministic": True,
            "model_count": len(models),
            "definition_count": len(pattern_entries) + len(discovered_entries),
        },
        "models": models,
        "definitions": {
            "patterns": pattern_entries,
            "discovered_kernels": discovered_entries,
        },
    }

    validate_knowledge_base(payload)
    return payload


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    # Try __file__-based root first (works for editable installs / running from source)
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "src" / "zmlx").is_dir():
        return candidate
    # Fall back to cwd (works for non-editable installs run from checkout)
    cwd = Path.cwd()
    if (cwd / "src" / "zmlx").is_dir():
        return cwd
    return candidate


def _build_pattern_entries(repo_root: Path) -> list[dict[str, Any]]:
    pattern_dir = repo_root / _PATTERNS_DIR
    entries: list[dict[str, Any]] = []
    for path in sorted(pattern_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        module_doc = ast.get_docstring(tree) or ""
        pattern_name, symbol = _extract_pattern_name_and_symbol(tree, path.stem)
        description = _first_paragraph(module_doc) or f"Patch pattern: {pattern_name}"

        formula = _extract_math_formula(module_doc, source)
        notes = _extract_implementation_notes(module_doc, source)
        constraints = _extract_constraints(module_doc, source)
        env_flags = sorted(set(_ENV_FLAG_RE.findall(source))) or None
        compatibility = _extract_compatibility(module_doc, source)
        evidence = _extract_evidence(
            source_path=_relative_path(repo_root, path),
            source_symbol=symbol or pattern_name,
            module_doc=module_doc,
            formula=formula,
            notes=notes,
        )

        entry = {
            "id": f"pattern:{pattern_name}",
            "name": pattern_name,
            "description": description,
            "math_formula": formula,
            "implementation_notes": notes,
            "source_path": _relative_path(repo_root, path),
            "source_symbol": symbol,
            "constraints": constraints,
            "env_flags": env_flags,
            "compatibility": compatibility,
            "evidence": evidence,
        }
        _attach_missing_reason(entry)
        entries.append(entry)

    entries.sort(key=lambda item: item["id"])
    return entries


def _build_discovered_entries(repo_root: Path) -> list[dict[str, Any]]:
    discovered_dir = repo_root / _DISCOVERED_DIR
    entries: list[dict[str, Any]] = []
    for path in sorted(discovered_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        module_doc = ast.get_docstring(tree) or ""

        source_symbol = _extract_primary_callable_symbol(tree)
        name = path.stem
        description = _first_paragraph(module_doc) or f"Discovered kernel: {name}"

        formula = _extract_math_formula(module_doc, source)
        notes = _extract_implementation_notes(module_doc, source)
        constraints = _extract_constraints(module_doc, source)
        env_flags = sorted(set(_ENV_FLAG_RE.findall(source))) or None
        compatibility = _extract_compatibility(module_doc, source)
        evidence = _extract_evidence(
            source_path=_relative_path(repo_root, path),
            source_symbol=source_symbol or name,
            module_doc=module_doc,
            formula=formula,
            notes=notes,
        )

        entry = {
            "id": f"kernel:{name}",
            "name": name,
            "description": description,
            "math_formula": formula,
            "implementation_notes": notes,
            "source_path": _relative_path(repo_root, path),
            "source_symbol": source_symbol,
            "constraints": constraints,
            "env_flags": env_flags,
            "compatibility": compatibility,
            "evidence": evidence,
        }
        _attach_missing_reason(entry)
        entries.append(entry)

    entries.sort(key=lambda item: item["id"])
    return entries


def _build_model_entries(definition_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    pattern_ids = sorted(def_id for def_id in definition_lookup if def_id.startswith("pattern:"))
    pattern_id_set = set(pattern_ids)
    kernel_ids = sorted(def_id for def_id in definition_lookup if def_id.startswith("kernel:"))

    catalog = sorted(load_catalog(), key=lambda item: item.model_id)
    model_entries: list[dict[str, Any]] = []
    for model in catalog:
        expected_ids = [
            pattern_id
            for pattern_id in sorted({f"pattern:{name}" for name in model.expected_patterns})
            if pattern_id in pattern_id_set
        ]
        excluded = {
            f"pattern:{name}": reason
            for name, reason in sorted(model.excluded_patterns.items())
            if f"pattern:{name}" in pattern_id_set
        }
        specialized = _build_specialized_kernel_refs(
            family=model.family,
            zmlx_family=model.zmlx_family,
            architecture=model.architecture,
            expected_pattern_ids=expected_ids,
            excluded_pattern_ids=excluded,
            kernel_ids=kernel_ids,
            definition_lookup=definition_lookup,
        )

        model_entry = {
            "id": model.model_id,
            "display_name": model.display_name,
            "family": model.family or "unknown",
            "zmlx_family": model.zmlx_family or "unknown",
            "architecture": model.architecture or "unknown",
            "quant": model.quant or "unknown",
            "params": model.total_params or "unknown",
            "storage_gb": float(model.storage_gb),
            "pattern_policy": {
                "expected": expected_ids,
                "excluded": excluded,
            },
            "specialized_kernels": specialized,
        }
        model_entries.append(model_entry)

    return model_entries


def _build_specialized_kernel_refs(
    *,
    family: str,
    zmlx_family: str,
    architecture: str,
    expected_pattern_ids: list[str],
    excluded_pattern_ids: dict[str, str],
    kernel_ids: list[str],
    definition_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    policy_evidence = [{
        "source_path": _MODELS_POLICY_SOURCE,
        "source_symbol": "_compute_expected_patterns",
        "kind": "policy",
        "snippet": "Pattern policy derives expected/excluded defaults from model family safety tables.",
    }]

    for kernel_id in expected_pattern_ids:
        ref = {
            "kernel_id": kernel_id,
            "applicability": "default_enabled",
            "rationale": "Enabled by model-aware default patch policy.",
            "constraints": [f"architecture:{architecture}"],
            "evidence": policy_evidence,
        }
        _attach_model_ref_missing_reason(ref)
        refs.append(ref)

    for kernel_id, reason in excluded_pattern_ids.items():
        ref = {
            "kernel_id": kernel_id,
            "applicability": "default_excluded",
            "rationale": f"Excluded by model-aware default patch policy ({reason}).",
            "constraints": [f"excluded_reason:{reason}", f"architecture:{architecture}"],
            "evidence": policy_evidence,
        }
        _attach_model_ref_missing_reason(ref)
        refs.append(ref)

    model_tags = {family.lower(), zmlx_family.lower()}
    matched_kernel_ids = [
        kernel_id
        for kernel_id in kernel_ids
        if _matches_kernel_family(kernel_id, model_tags)
    ]

    for kernel_id in matched_kernel_ids:
        definition = definition_lookup[kernel_id]
        constraints = definition.get("constraints")
        if isinstance(constraints, list):
            constraints = constraints[:3]
        evidence = definition.get("evidence")
        if isinstance(evidence, list):
            evidence = evidence[:2]
        ref = {
            "kernel_id": kernel_id,
            "applicability": "optional_family_specific",
            "rationale": "Discovered kernel target prefix matches model family.",
            "constraints": constraints if isinstance(constraints, list) and constraints else None,  # type: ignore[dict-item]
            "evidence": evidence if isinstance(evidence, list) and evidence else None,  # type: ignore[dict-item]
        }
        _attach_model_ref_missing_reason(ref)
        refs.append(ref)

    refs.sort(key=lambda item: (item["kernel_id"], item["applicability"], item["rationale"]))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        key = (ref["kernel_id"], ref["applicability"])
        if key in seen:
            continue
        seen.add(key)  # type: ignore[arg-type]
        deduped.append(ref)

    # No fallback — if no specialized kernels match, return empty rather than
    # fabricating a reference to a definition that may not exist.
    return deduped


def _matches_kernel_family(kernel_id: str, model_tags: set[str]) -> bool:
    if not kernel_id.startswith("kernel:"):
        return False
    stem = kernel_id.split(":", 1)[1]
    for tag in model_tags:
        if not tag or tag == "unknown":
            continue
        if stem.startswith(f"{tag}_"):
            return True
    return False


def _extract_pattern_name_and_symbol(tree: ast.Module, fallback_stem: str) -> tuple[str, str | None]:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name != "name":
                continue
            for stmt in ast.walk(item):
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant):
                    value = stmt.value.value
                    if isinstance(value, str) and value:
                        return value, node.name
    return fallback_stem, None


def _extract_primary_callable_symbol(tree: ast.Module) -> str | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            return node.name
    return None


def _extract_math_formula(module_doc: str, source: str) -> str | None:
    candidates: list[str] = []
    for line in _iter_signal_lines(module_doc, source):
        if _FORMULA_HINT_RE.search(line):
            candidates.append(line)
    unique = _dedupe(candidates)
    if not unique:
        return None
    return "; ".join(unique[:2])


def _extract_implementation_notes(module_doc: str, source: str) -> list[str] | None:
    candidates: list[str] = []
    for line in _iter_signal_lines(module_doc, source):
        if _FORMULA_HINT_RE.search(line):
            continue
        if len(line.split()) < 4:
            continue
        candidates.append(line)
    unique = _dedupe(candidates)
    if not unique:
        return None
    return unique[:8]


def _extract_constraints(module_doc: str, source: str) -> list[str] | None:
    candidates: list[str] = []
    for line in _iter_signal_lines(module_doc, source):
        if _CONSTRAINT_HINT_RE.search(line):
            candidates.append(line)
    unique = _dedupe(candidates)
    if not unique:
        return None
    return unique[:8]


def _extract_compatibility(module_doc: str, source: str) -> list[str] | None:
    text = f"{module_doc}\n{source}".lower()
    tags: list[str] = []
    for token, label in _FAMILY_HINTS.items():
        if token in text:
            tags.append(label)
    if "moe" in text:
        tags.append("architecture:moe")
    if "dense" in text:
        tags.append("architecture:dense")
    deduped = sorted(set(tags))
    return deduped or None


def _extract_evidence(
    *,
    source_path: str,
    source_symbol: str,
    module_doc: str,
    formula: str | None,
    notes: list[str] | None,
) -> list[dict[str, str]] | None:
    evidence: list[dict[str, str]] = []
    first_para = _first_paragraph(module_doc)
    if first_para:
        evidence.append(
            {
                "source_path": source_path,
                "source_symbol": source_symbol,
                "kind": "module_docstring",
                "snippet": first_para,
            }
        )
    if formula:
        evidence.append(
            {
                "source_path": source_path,
                "source_symbol": source_symbol,
                "kind": "formula_hint",
                "snippet": formula,
            }
        )
    if notes:
        for note in notes[:2]:
            evidence.append(
                {
                    "source_path": source_path,
                    "source_symbol": source_symbol,
                    "kind": "implementation_note",
                    "snippet": note,
                }
            )

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in evidence:
        key = (item["kind"], item["snippet"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped or None


def _iter_signal_lines(module_doc: str, source: str) -> list[str]:
    lines: list[str] = []

    for raw in module_doc.splitlines():
        line = _normalize_doc_line(raw)
        if line:
            lines.append(line)

    for raw in source.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            line = _normalize_doc_line(stripped.lstrip("#"))
            if line:
                lines.append(line)
        elif stripped.startswith("//"):
            line = _normalize_doc_line(stripped.lstrip("/"))
            if line:
                lines.append(line)

    return lines


def _normalize_doc_line(raw: str) -> str:
    line = raw.strip()
    if not line:
        return ""
    for prefix in ("- ", "* ", "1) ", "2) ", "3) ", "4) ", "5) "):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
    line = re.sub(r"\s+", " ", line).strip()
    if not line:
        return ""
    if set(line) <= {"-"}:
        return ""
    return line


def _first_paragraph(doc: str) -> str:
    lines: list[str] = []
    for raw in doc.splitlines():
        line = raw.strip()
        if not line:
            if lines:
                break
            continue
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _relative_path(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _attach_missing_reason(entry: dict[str, Any]) -> None:
    missing_reason: dict[str, str] = {}
    reasons = {
        "math_formula": "No explicit mathematical expression found in source comments/docstrings.",
        "implementation_notes": "No implementation notes could be extracted from source comments/docstrings.",
        "source_symbol": "No canonical class/function symbol was identified for this definition.",
        "constraints": "No explicit constraints were detected from source guards/docstrings.",
        "env_flags": "No ZMLX_* environment flags are referenced by this definition.",
        "compatibility": "No model-family compatibility hints were found in source text.",
        "evidence": "No source snippets were captured to support extracted metadata.",
    }
    for field in DEFINITION_NULLABLE_FIELDS:
        value = entry.get(field)
        if isinstance(value, list) and not value:
            value = None
            entry[field] = None
        if value is None:
            missing_reason[field] = reasons[field]
    entry["missing_reason"] = missing_reason


def _attach_model_ref_missing_reason(ref: dict[str, Any]) -> None:
    missing: dict[str, str] = {}
    if ref.get("constraints") is None:
        missing["constraints"] = "No additional model-specific constraints were derived."
    if ref.get("evidence") is None:
        missing["evidence"] = "No model-specific evidence snippets were attached."
    ref["missing_reason"] = missing


__all__ = ["build_knowledge_base"]
