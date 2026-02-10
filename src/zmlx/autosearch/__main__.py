"""CLI for ZMLX Autosearch: structured kernel autotuner.

Usage:
    python -m zmlx.autosearch run swiglu --D 1536 --generations 10
    python -m zmlx.autosearch run rmsnorm --D 2048 --generations 10
    python -m zmlx.autosearch run moe_combine --D 2048 --K 4 --exhaustive
    python -m zmlx.autosearch list
    python -m zmlx.autosearch export swiglu --D 1536 --output /tmp/best_swiglu.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any


def _run(args: argparse.Namespace) -> None:
    """Run autosearch on a kernel template."""
    from .harness import Harness
    from .search import _fmt_config, auto_search
    from .templates import get_space, get_target_name, get_template_fn

    template_name = args.template

    # Map template name to discover target
    target_name = get_target_name(template_name)
    D = args.D
    K = args.K

    print(f"ZMLX Autosearch: {template_name} (D={D}, K={K})")
    print(f"  Target: {target_name}")

    # Config space
    space = get_space(template_name, D, K)
    valid_count = len(space.enumerate_all())
    print(f"  Config space: {space.total_configs} total, {valid_count} valid")

    # Template function
    template_fn = get_template_fn(template_name, D, K)

    # Harness
    harness = Harness(
        target_name=target_name,
        D=D,
        K=K,
        template_fn=template_fn,
    )
    print("  Setting up harness...")
    harness.setup(warmup=args.warmup, iters=args.iters)
    print(f"  Baseline: {harness.baseline_us:.1f} us")

    # Evaluate function
    def evaluate_fn(
        config: dict[str, Any],
        source: str,
        grid: tuple[int, int, int],
        tg: tuple[int, int, int],
    ) -> Any:
        return harness.evaluate(
            config, source, grid, tg,
            warmup=args.warmup, iters=args.iters, timeout_s=args.timeout,
        )

    # Search
    t0 = time.time()

    if args.exhaustive:
        from .search import ExhaustiveSearch

        searcher = ExhaustiveSearch(space, template_fn)
        results = searcher.run(evaluate_fn, verbose=args.verbose)
    else:
        results = auto_search(
            space, template_fn, evaluate_fn,
            generations=args.generations,
            population_size=args.population,
            seed=args.seed,
            verbose=args.verbose,
        )

    elapsed = time.time() - t0

    # Results table
    valid_results = [r for r in results if r.eval_result.correct]
    print(f"\n{'=' * 70}")
    print(f"  AUTOSEARCH RESULTS — {template_name} D={D}")
    print(f"{'=' * 70}")
    print(f"  Evaluated: {len(results)} configs ({len(valid_results)} correct)")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Baseline: {harness.baseline_us:.1f} us")

    if valid_results:
        print(f"\n  {'Rank':>4s}  {'Speedup':>8s}  {'Median':>10s}  Config")
        print(f"  {'-' * 4}  {'-' * 8}  {'-' * 10}  {'-' * 40}")
        for i, r in enumerate(valid_results[:10]):
            print(f"  {i + 1:4d}  {r.speedup:7.2f}x  {r.eval_result.median_us:9.1f}us  "
                  f"{_fmt_config(r.config)}")
    else:
        print("  No valid configs found!")

    print(f"{'=' * 70}\n")


def _list(_args: argparse.Namespace) -> None:
    """List available templates."""
    from .templates import TEMPLATES, get_space

    print("Available autosearch templates:")
    for name, (_, target) in TEMPLATES.items():
        space = get_space(name, D=2048, K=4)
        valid = len(space.enumerate_all())
        print(f"  {name:15s}  discover target: {target:15s}  {valid:4d} valid configs")


def _export(args: argparse.Namespace) -> None:
    """Run search and export best kernel as a Python module."""
    from .harness import Harness
    from .search import auto_search
    from .templates import get_space, get_target_name, get_template_fn

    template_name = args.template
    target_name = get_target_name(template_name)
    D = args.D
    K = args.K

    space = get_space(template_name, D, K)
    template_fn = get_template_fn(template_name, D, K)

    harness = Harness(target_name=target_name, D=D, K=K, template_fn=template_fn)
    harness.setup(warmup=args.warmup, iters=args.iters)

    def evaluate_fn(
        config: dict[str, Any],
        source: str,
        grid: tuple[int, int, int],
        tg: tuple[int, int, int],
    ) -> Any:
        return harness.evaluate(config, source, grid, tg,
                                warmup=args.warmup, iters=args.iters)

    print(f"Searching {template_name} D={D}...")
    results = auto_search(
        space, template_fn, evaluate_fn,
        generations=args.generations,
        population_size=args.population,
        seed=args.seed,
        verbose=True,
    )

    valid = [r for r in results if r.eval_result.correct]
    if not valid:
        print("No valid configs found — cannot export.")
        sys.exit(1)

    best = valid[0]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build input/output names from discover target
    import inspect

    from ..discover.targets import TARGETS
    sig = inspect.signature(TARGETS[target_name])
    kwargs: dict[str, Any] = {}
    if "D" in sig.parameters:
        kwargs["D"] = D
    if "K" in sig.parameters:
        kwargs["K"] = K
    ss = TARGETS[target_name](**kwargs)
    input_names = list(ss.input_names)
    output_names = list(ss.output_names)

    code = f'''"""Auto-generated kernel from ZMLX Autosearch.

Template: {template_name}
Config: {best.config}
Speedup: {best.speedup:.2f}x
Baseline: {harness.baseline_us:.1f}us -> {best.eval_result.median_us:.1f}us
"""

from __future__ import annotations

from functools import cache
from typing import Any

from zmlx.metal import kernel as metal_kernel
from zmlx.msl import DEFAULT_HEADER


@cache
def _kernel() -> Any:
    """Build the autosearch-discovered kernel."""
    source = """{best.source}"""

    return metal_kernel(
        name="kk_autosearch_{template_name}",
        input_names={input_names!r},
        output_names={output_names!r},
        source=source,
        header=DEFAULT_HEADER,
        cache=True,
    )
'''

    output_path.write_text(code)
    print(f"Exported best kernel to {output_path}")
    print(f"  Config: {best.config}")
    print(f"  Speedup: {best.speedup:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="zmlx.autosearch",
        description="Structured kernel autotuner for ZMLX (no LLM required)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    sp = subparsers.add_parser("run", help="Run autosearch on a template")
    sp.add_argument("template", choices=["swiglu", "rmsnorm", "moe_combine"])
    sp.add_argument("--D", type=int, default=2048, help="Hidden dimension")
    sp.add_argument("--K", type=int, default=2, help="Number of experts (moe_combine)")
    sp.add_argument("--generations", type=int, default=10)
    sp.add_argument("--population", type=int, default=24)
    sp.add_argument("--exhaustive", action="store_true",
                    help="Try all valid configs instead of evolutionary search")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--warmup", type=int, default=5)
    sp.add_argument("--iters", type=int, default=20)
    sp.add_argument("--timeout", type=float, default=10.0)
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.set_defaults(func=_run)

    # list
    lp = subparsers.add_parser("list", help="List available templates")
    lp.set_defaults(func=_list)

    # export
    ep = subparsers.add_parser("export", help="Search and export best kernel")
    ep.add_argument("template", choices=["swiglu", "rmsnorm", "moe_combine"])
    ep.add_argument("--output", "-o", required=True, help="Output file path")
    ep.add_argument("--D", type=int, default=2048)
    ep.add_argument("--K", type=int, default=2)
    ep.add_argument("--generations", type=int, default=10)
    ep.add_argument("--population", type=int, default=24)
    ep.add_argument("--seed", type=int, default=42)
    ep.add_argument("--warmup", type=int, default=5)
    ep.add_argument("--iters", type=int, default=20)
    ep.set_defaults(func=_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
