"""CLI for generating and validating the ZMLX knowledge base."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .builder import build_knowledge_base
from .schema import (
    KB_FILE_NAME,
    KB_JSON_SCHEMA,
    KBValidationError,
    canonical_dumps,
    coverage_report,
    load_knowledge_base,
    unresolved_gaps,
    validate_knowledge_base,
    write_knowledge_base,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m zmlx.kb",
        description="Build and validate the canonical ZMLX knowledge base JSON artifact.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build and write zmlx_knowledge_base.json")
    build_p.add_argument("--output", type=Path, default=Path(KB_FILE_NAME))
    build_p.add_argument(
        "--check-deterministic",
        action="store_true",
        help="Build twice and fail if canonical bytes differ.",
    )

    validate_p = sub.add_parser("validate", help="Validate an existing KB JSON")
    validate_p.add_argument("--input", type=Path, default=Path(KB_FILE_NAME))

    report_p = sub.add_parser("report", help="Print coverage + unresolved gaps")
    report_p.add_argument("--input", type=Path, default=Path(KB_FILE_NAME))

    schema_p = sub.add_parser("schema", help="Print JSON schema used by validator")
    schema_p.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.command == "build":
        payload = build_knowledge_base()
        if args.check_deterministic:
            payload_2 = build_knowledge_base()
            if canonical_dumps(payload) != canonical_dumps(payload_2):
                print("Determinism check failed: repeated build produced different bytes.", file=sys.stderr)
                return 1
        write_knowledge_base(args.output, payload)
        print(f"Wrote {args.output}")
        return 0

    if args.command == "validate":
        try:
            payload = load_knowledge_base(args.input)
            validate_knowledge_base(payload)
        except (OSError, json.JSONDecodeError, KBValidationError) as exc:
            print(f"Validation failed: {exc}", file=sys.stderr)
            return 1
        print(f"Valid: {args.input}")
        return 0

    if args.command == "report":
        try:
            payload = load_knowledge_base(args.input)
            validate_knowledge_base(payload)
        except (OSError, json.JSONDecodeError, KBValidationError) as exc:
            print(f"Cannot report due to invalid KB: {exc}", file=sys.stderr)
            return 1

        coverage = coverage_report(payload)
        gaps = unresolved_gaps(payload)
        print("Coverage")
        print(
            f"- math_formula non-null: {coverage['math_formula_non_null']}/{coverage['definitions_total']}"
        )
        print(
            "- implementation_notes non-empty: "
            f"{coverage['implementation_notes_non_empty']}/{coverage['definitions_total']}"
        )
        print(f"- unresolved gaps: {len(gaps)}")
        if gaps:
            print("Unresolved Gap Details")
            for gap in gaps:
                print(
                    f"- [{gap['scope']}] {gap['id']} :: {gap['field']} -> {gap['reason']}"
                )
        return 0

    if args.command == "schema":
        rendered = canonical_dumps(KB_JSON_SCHEMA)
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.write_text(rendered, encoding="utf-8")
            print(f"Wrote {args.output}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
