"""CLI entry point: ``python -m zmlx.matrix``."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m zmlx.matrix",
        description="ZMLX kernel test matrix: catalog, run, report.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- catalog ---
    sub.add_parser("catalog", help="Print model catalog")

    # --- run ---
    run_p = sub.add_parser("run", help="Run validation for one or more models")
    run_p.add_argument("model", nargs="?", help="HuggingFace model ID")
    run_p.add_argument("--all", action="store_true", help="Run all models that fit in RAM")
    run_p.add_argument("--family", type=str, default=None, help="Run all models in a family")
    run_p.add_argument("--runs", type=int, default=3, help="Timed runs per config")
    run_p.add_argument("--max-tokens", type=int, default=200, help="Tokens to generate")
    run_p.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help="ZMLX patch patterns to apply (default: model-aware patch defaults)",
    )
    run_p.add_argument(
        "--patch-profile",
        type=str,
        default=None,
        help="Patch profile to apply (e.g., qwen3)",
    )
    run_p.add_argument("--notes", type=str, default="", help="Freeform notes to store")
    run_p.add_argument("--ledger", type=str, default=None, help="Path to JSONL ledger")

    # --- report ---
    report_p = sub.add_parser("report", help="Show latest results as heatmap table")
    report_p.add_argument("--ledger", type=str, default=None, help="Path to JSONL ledger")
    report_p.add_argument("--hardware", type=str, default=None, help="Filter by hardware")

    # --- history ---
    hist_p = sub.add_parser("history", help="Show result history for a model")
    hist_p.add_argument("model", help="HuggingFace model ID or substring")
    hist_p.add_argument("--ledger", type=str, default=None, help="Path to JSONL ledger")
    hist_p.add_argument("--hardware", type=str, default=None, help="Filter by hardware")

    # --- csv ---
    csv_p = sub.add_parser("csv", help="Export results as CSV")
    csv_p.add_argument("--ledger", type=str, default=None, help="Path to JSONL ledger")

    # --- html ---
    html_p = sub.add_parser("html", help="Export results as self-contained HTML")
    html_p.add_argument("--ledger", type=str, default=None, help="Path to JSONL ledger")

    # --- diff ---
    diff_p = sub.add_parser("diff", help="Compare two JSONL ledger files")
    diff_p.add_argument("file1", help="First JSONL file")
    diff_p.add_argument("file2", help="Second JSONL file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "catalog":
        from .models import print_catalog
        print_catalog()

    elif args.command == "run":
        from .runner import run_all, run_one

        if args.patterns is not None and args.patch_profile is not None:
            print("Use either --patterns or --patch-profile, not both.", file=sys.stderr)
            sys.exit(1)

        # Convenience: allow `--patterns []` / `--patterns none` to mean "no patches".
        patterns = args.patterns
        if patterns is not None and len(patterns) == 1:
            raw = patterns[0].strip().lower()
            if raw in {"[]", "none", "no", "off"}:
                patterns = []

        if args.model:
            entry = run_one(
                args.model,
                patterns=patterns,
                profile=args.patch_profile,
                runs=args.runs,
                max_tokens=args.max_tokens,
                notes=args.notes,
            )
            from .storage import append
            append(entry, args.ledger)
            print(f"\n[matrix] Result: {entry.fidelity} | "
                  f"decode {entry.decode_speedup:.3f}x | "
                  f"{entry.decode_tps_patched:.1f} tok/s")
        elif args.all or args.family:
            run_all(
                runs=args.runs,
                max_tokens=args.max_tokens,
                family_filter=args.family,
                ledger_path=args.ledger,
            )
        else:
            print("Specify a model ID, --all, or --family", file=sys.stderr)
            sys.exit(1)

    elif args.command == "report":
        from .report import print_heatmap
        print_heatmap(ledger_path=args.ledger, hardware_filter=args.hardware)

    elif args.command == "history":
        _history(args.model, ledger=args.ledger, hardware=args.hardware)

    elif args.command == "csv":
        from .report import to_csv
        print(to_csv(ledger_path=args.ledger))

    elif args.command == "html":
        from .report import to_html
        print(to_html(ledger_path=args.ledger))

    elif args.command == "diff":
        _diff(args.file1, args.file2)


def _history(model: str, *, ledger: str | None = None, hardware: str | None = None) -> None:
    """Print history for a given model (substring match)."""
    from .storage import load_all

    query = (model or "").strip().lower()
    if not query:
        print("Model query must be non-empty.", file=sys.stderr)
        return

    entries = load_all(ledger)
    filtered = []
    for e in entries:
        mid = (e.model_id or "").lower()
        suffix = mid.rsplit("/", 1)[-1]
        if query not in mid and query not in suffix:
            continue
        if hardware and e.hardware != hardware:
            continue
        filtered.append(e)

    if not filtered:
        print("\n[matrix] No matching entries.")
        return

    filtered.sort(key=lambda e: e.timestamp)
    print()
    print(f"[matrix] History for {model!r} ({len(filtered)} entries)")
    if hardware:
        print(f"[matrix] Hardware: {hardware}")

    print()
    header = (
        f"{'Timestamp':<20} {'Decode':>8} {'Base':>8} {'Patched':>8} "
        f"{'Fid':>6} {'Patterns':<28} Notes"
    )
    print(header)
    print("-" * len(header))
    for e in filtered:
        ts = (e.timestamp or "")[:19]
        patt = ",".join(e.patterns_applied) if e.patterns_applied else "(none)"
        notes = (e.notes or "").replace("\n", " ").strip()
        print(
            f"{ts:<20} {e.decode_speedup:>8.3f} {e.decode_tps_baseline:>8.1f} "
            f"{e.decode_tps_patched:>8.1f} {e.fidelity:>6} {patt:<28} {notes}"
        )
    print()


def _diff(file1: str, file2: str) -> None:
    """Compare two JSONL ledger files and show regressions/improvements."""
    from .storage import latest as load_latest

    lat1 = load_latest(file1)
    lat2 = load_latest(file2)

    all_keys = sorted(set(lat1.keys()) | set(lat2.keys()))
    if not all_keys:
        print("No entries to compare.")
        return

    print(f"\n{'Model':<40} {'Old':>10} {'New':>10} {'Change':>10}")
    print("-" * 72)

    for key in all_keys:
        e1 = lat1.get(key)
        e2 = lat2.get(key)
        model_name = key[0].rsplit("/", 1)[-1] if "/" in key[0] else key[0]

        old_tps = f"{e1.decode_tps_patched:.1f}" if e1 else "--"
        new_tps = f"{e2.decode_tps_patched:.1f}" if e2 else "--"

        if e1 and e2 and e1.decode_tps_patched > 0:
            change = (e2.decode_tps_patched / e1.decode_tps_patched - 1.0) * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = "--"

        print(f"{model_name:<40} {old_tps:>10} {new_tps:>10} {change_str:>10}")

    print()


if __name__ == "__main__":
    main()
