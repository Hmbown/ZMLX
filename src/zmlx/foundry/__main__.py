"""CLI entry point for the ZMLX foundry.

Usage::

    python -m zmlx.foundry run    [options]   # generate kernel dataset
    python -m zmlx.foundry report [options]   # coverage + pareto reports
    python -m zmlx.foundry export [options]   # export training JSONL
    python -m zmlx.foundry export-sft [options]  # export chat SFT JSONL
    python -m zmlx.foundry list               # list registered ops

Example::

    python -m zmlx.foundry run --ops rmsnorm swiglu --n 500 --workers 4
    python -m zmlx.foundry report sessions/my_run
    python -m zmlx.foundry export sessions/my_run --out training_data/
    python -m zmlx.foundry export-sft --out training_data/kernel_sft
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> None:
    """Run a foundry session."""
    from .ops import get_registry
    from .scheduler import flatten_curriculum
    from .workers import spawn_workers

    registry = get_registry()

    if args.ops:
        ops = args.ops
        unknown = [o for o in ops if o not in registry]
        if unknown:
            print(f"Unknown ops: {', '.join(unknown)}", file=sys.stderr)
            print(f"Available: {', '.join(sorted(registry))}", file=sys.stderr)
            sys.exit(1)
    else:
        ops = flatten_curriculum()

    # Session directory
    if args.session_dir:
        session_dir = args.session_dir
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        session_dir = str(Path("sessions") / f"foundry_{ts}")

    print(f"Session: {session_dir}")
    print(f"Ops: {', '.join(ops)}")
    print(f"Attempts: {args.n}, Workers: {args.workers}, Mode: {args.mode}")
    print(f"Backend: {args.backend}, Stage: {args.stage}")
    print()

    result = spawn_workers(
        session_dir=session_dir,
        ops=ops,
        n_attempts=args.n,
        num_workers=args.workers,
        mode=args.mode,
        seed=args.seed,
        backend=args.backend,
        correctness_tests=args.correctness_tests,
        warmup=args.warmup,
        repeats=args.repeats,
        bench_timeout_s=args.bench_timeout,
        stage=args.stage,
    )

    print(f"\nDone in {result['total_elapsed_s']:.1f}s")
    for w in result["workers"]:
        print(
            f"  worker {w['worker_id']}: "
            f"{w['n_written']} written, "
            f"{w['n_skipped']} skipped, "
            f"{w['elapsed_s']:.1f}s"
        )
    print(f"Merged log: {result['merged_path']}")


def _cmd_report(args: argparse.Namespace) -> None:
    """Generate coverage and pareto reports."""
    from .reports import build_coverage, extract_pareto_front, write_coverage_reports
    from .reports.pareto import load_attempts

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        print(f"Session directory not found: {session_dir}", file=sys.stderr)
        sys.exit(1)

    md_path, json_path = write_coverage_reports(session_dir)
    print(f"Coverage report: {md_path}")
    print(f"Coverage data:   {json_path}")

    # Print summary
    cov = build_coverage(session_dir)
    n_ops = len(cov["ops"])
    total = sum(d["attempts"] for d in cov["ops"].values())
    total_ok = sum(d["ok"] for d in cov["ops"].values())
    rate = (total_ok / total * 100) if total > 0 else 0
    print(f"\n{n_ops} ops, {total} attempts, {total_ok} passed ({rate:.1f}%)")

    for op, d in sorted(cov["ops"].items()):
        best = d["best_template_latency_ms"]
        best_s = f"{best:.4f} ms" if isinstance(best, (int, float)) else "-"
        print(f"  {op:30s}  {d['ok']:4d}/{d['attempts']:<4d}  best: {best_s}")

    # Pareto front
    attempts = load_attempts(str(session_dir))
    pareto = extract_pareto_front(attempts)
    if pareto:
        print(f"\nPareto front: {len(pareto)} kernels")
        for p in pareto[:5]:
            print(f"  {p['op']}/{p.get('template_id', '?')}: {p.get('p50_ms', '?')} ms")
    else:
        print("\nNo pareto front (no successful benchmarked attempts)")


def _cmd_export(args: argparse.Namespace) -> None:
    """Export training JSONL."""
    from .export.training import export_training_jsonl

    result = export_training_jsonl(
        session_dir=args.session_dir,
        out_dir=args.out,
        ops=args.ops or None,
        min_p50_ms=args.min_p50,
        max_p50_ms=args.max_p50,
    )

    print(f"Exported {result['n_records']} records ({result['n_skipped']} skipped)")
    print(f"Output: {result['out_path']}")


def _cmd_export_sft(args: argparse.Namespace) -> None:
    """Export chat-style SFT JSONL from foundry/discovery artifacts."""
    from .export.sft import export_kernel_sft_jsonl

    result = export_kernel_sft_jsonl(
        out_dir=args.out,
        sessions_root=args.sessions_root,
        runs_root=args.runs_root,
        discover_root=args.discover_root,
        train_fraction=args.train_fraction,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        max_examples=args.max_examples,
        include_failed_kd=args.include_failed_kd,
    )

    counts = result["counts"]
    print(
        "Exported SFT dataset "
        f"(total={counts['total']}, train={counts['train']}, "
        f"valid={counts['valid']}, test={counts['test']})"
    )
    print(f"Manifest: {args.out}/manifest.json")


def _cmd_list(_args: argparse.Namespace) -> None:
    """List registered ops."""
    from .ops import get_registry
    from .scheduler import DEFAULT_CURRICULUM
    from .templates import list_templates

    registry = get_registry()

    # Build stage map
    stage_map = {}
    for stage_idx, stage_ops in enumerate(DEFAULT_CURRICULUM):
        for op in stage_ops:
            stage_map[op] = stage_idx

    print(f"{'Op':30s}  {'Class':12s}  {'Stage':6s}  {'Templates':20s}")
    print("-" * 75)
    for name, op in sorted(registry.items()):
        try:
            kc = op.spec().kernel_class.value
        except Exception:
            kc = "?"
        stage = stage_map.get(name, "?")
        templates = list_templates(name)
        tpl_str = ", ".join(templates) if templates else "(ref only)"
        print(f"{name:30s}  {str(kc):12s}  {str(stage):6s}  {tpl_str}")

    print(f"\n{len(registry)} ops registered")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zmlx-foundry",
        description="ZMLX Foundry -- Metal kernel template evaluation and dataset generation",
    )
    sub = parser.add_subparsers(dest="command")

    # -- run ---------------------------------------------------------------
    p_run = sub.add_parser("run", help="Run a foundry evaluation session")
    p_run.add_argument("--ops", nargs="+", help="Ops to evaluate (default: all curriculum)")
    p_run.add_argument("-n", type=int, default=100, help="Number of attempts per op (default: 100)")
    p_run.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    p_run.add_argument("--mode", default="mix", choices=["random", "coverage", "mutation", "mix"],
                        help="Sampling mode (default: mix)")
    p_run.add_argument("--seed", type=int, default=42, help="Base random seed")
    p_run.add_argument("--backend", default="mlx", choices=["mlx", "mock"],
                        help="Evaluation backend (default: mlx)")
    p_run.add_argument("--stage", type=int, default=4,
                        help="Curriculum stage (0-4, default: 4 = all ops)")
    p_run.add_argument("--session-dir", help="Session directory (default: auto-generated)")
    p_run.add_argument("--correctness-tests", type=int, default=3)
    p_run.add_argument("--warmup", type=int, default=10)
    p_run.add_argument("--repeats", type=int, default=50)
    p_run.add_argument("--bench-timeout", type=float, default=10.0)

    # -- report ------------------------------------------------------------
    p_report = sub.add_parser("report", help="Generate coverage/pareto reports")
    p_report.add_argument("session_dir", help="Session directory to analyze")

    # -- export ------------------------------------------------------------
    p_export = sub.add_parser("export", help="Export training JSONL from a session")
    p_export.add_argument("session_dir", help="Session directory")
    p_export.add_argument("--out", default="training_data", help="Output directory")
    p_export.add_argument("--ops", nargs="+", help="Filter to specific ops")
    p_export.add_argument("--min-p50", type=float, help="Min p50 latency filter (ms)")
    p_export.add_argument("--max-p50", type=float, help="Max p50 latency filter (ms)")

    # -- export-sft --------------------------------------------------------
    p_export_sft = sub.add_parser(
        "export-sft",
        help="Export chat SFT JSONL from sessions/runs/discover artifacts",
    )
    p_export_sft.add_argument(
        "--out",
        default="training_data/kernel_sft",
        help="Output directory for train/valid/test JSONL",
    )
    p_export_sft.add_argument(
        "--sessions-root",
        default="sessions",
        help="Root directory containing foundry session attempts",
    )
    p_export_sft.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory containing KD run logs",
    )
    p_export_sft.add_argument(
        "--discover-root",
        default="discover_sessions",
        help="Root directory containing discover session JSON files",
    )
    p_export_sft.add_argument(
        "--train-fraction",
        type=float,
        default=0.96,
        help="Fraction of examples assigned to train split",
    )
    p_export_sft.add_argument(
        "--valid-fraction",
        type=float,
        default=0.03,
        help="Fraction of examples assigned to valid split",
    )
    p_export_sft.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed",
    )
    p_export_sft.add_argument(
        "--max-examples",
        type=int,
        help="Optional cap on total examples after shuffle",
    )
    p_export_sft.add_argument(
        "--include-failed-kd",
        action="store_true",
        help="Include KD failed candidates as negative examples",
    )

    # -- list --------------------------------------------------------------
    sub.add_parser("list", help="List registered ops and their metadata")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "run": _cmd_run,
        "report": _cmd_report,
        "export": _cmd_export,
        "export-sft": _cmd_export_sft,
        "list": _cmd_list,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
