"""Report renderers: terminal heatmap table, CSV export, HTML export."""

from __future__ import annotations

import csv
import io
import sys
from datetime import datetime

from .models import ModelInfo, load_catalog
from .schema import MatrixEntry
from .storage import latest, snapshot

# ---------------------------------------------------------------------------
# Color helpers (ANSI)
# ---------------------------------------------------------------------------

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _color_speedup(speedup: float, use_color: bool = True) -> str:
    """Format a speedup value with color coding."""
    if speedup == 0.0:
        return _dim("--", use_color)
    pct = (speedup - 1.0) * 100
    text = f"{pct:+.1f}%"
    if not use_color:
        return text
    if pct > 2.0:
        return f"{_GREEN}{text}{_RESET}"
    if pct < -2.0:
        return f"{_RED}{text}{_RESET}"
    return f"{_YELLOW}{text}{_RESET}"


def _color_fidelity(fidelity: str, use_color: bool = True) -> str:
    if not use_color:
        return fidelity
    if fidelity == "PASS":
        return f"{_GREEN}{fidelity}{_RESET}"
    if fidelity == "FAIL":
        return f"{_RED}{fidelity}{_RESET}"
    return f"{_DIM}{fidelity}{_RESET}"


def _dim(text: str, use_color: bool = True) -> str:
    if not use_color:
        return text
    return f"{_DIM}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Terminal heatmap table
# ---------------------------------------------------------------------------

def _pattern_cell(
    entry: MatrixEntry | None,
    pattern: str,
    model_info: ModelInfo | None,
    use_color: bool,
) -> str:
    """Render one cell in the pattern columns."""
    # Check if pattern is excluded for this model
    if model_info and pattern in model_info.excluded_patterns:
        reason = model_info.excluded_patterns[pattern]
        return _dim(f"EXCL({reason[0]})", use_color)

    # Check if pattern doesn't apply (e.g. moe_mlp on dense model)
    if model_info:
        if pattern == "moe_mlp" and model_info.architecture != "moe":
            return _dim("--", use_color)

    if entry is None:
        return _dim("--", use_color)

    # We have a result — show decode speedup
    return _color_speedup(entry.decode_speedup, use_color)


def print_heatmap(
    ledger_path: str | None = None,
    hardware_filter: str | None = None,
) -> None:
    """Print the terminal heatmap table from the JSONL ledger."""
    use_color = _supports_color()
    snap = snapshot(ledger_path, hardware=hardware_filter)
    catalog = load_catalog()

    # Merge: show all catalog models, with results overlaid
    entry_by_model: dict[str, MatrixEntry] = {}
    for e in snap.entries:
        entry_by_model[e.model_id] = e

    # Collect environment info from any entry
    env_line = ""
    if snap.entries:
        e0 = snap.entries[0]
        parts = []
        if e0.mlx_version:
            parts.append(f"MLX {e0.mlx_version}")
        if e0.zmlx_version:
            parts.append(f"ZMLX {e0.zmlx_version}")
        if e0.custom_mlx:
            parts.append("Custom MLX: Yes")
        else:
            parts.append("Custom MLX: No")
        env_line = " | ".join(parts)

    hw = snap.hardware or "(all hardware)"
    date = snap.date[:10] if snap.date else datetime.now().strftime("%Y-%m-%d")

    print()
    if use_color:
        print(f"{_BOLD}ZMLX Test Matrix{_RESET} -- {hw} -- {date}")
    else:
        print(f"ZMLX Test Matrix -- {hw} -- {date}")
    if env_line:
        print(env_line)
    print()

    # Patterns to show as columns
    pattern_cols = ["moe_mlp", "swiglu_mlp", "geglu_mlp"]

    # Header
    hdr = (
        f"{'Family':<10} {'Model':<36} {'Arch':<5} {'Quant':<9} "
        f"{'Size':>6}  "
    )
    for p in pattern_cols:
        hdr += f"{p:>12}  "
    hdr += f"{'Fidelity':>10}"
    print(hdr)
    print("-" * len(hdr))

    for m in catalog:
        entry = entry_by_model.get(m.model_id)

        size_str = f"{m.storage_gb:.0f}GB" if m.storage_gb >= 1 else f"{m.storage_gb:.1f}GB"

        row = f"{m.family:<10} {m.display_name:<36} {m.architecture:<5} {m.quant:<9} {size_str:>6}  "

        for p in pattern_cols:
            cell = _pattern_cell(entry, p, m, use_color)
            # Need to account for ANSI codes in alignment
            visible_len = len(cell.replace(_GREEN, "").replace(_YELLOW, "")
                              .replace(_RED, "").replace(_DIM, "")
                              .replace(_BOLD, "").replace(_RESET, ""))
            padding = 12 - visible_len
            row += " " * max(0, padding) + cell + "  "

        if entry:
            row += _color_fidelity(entry.fidelity, use_color).rjust(10)
        else:
            avail = _available_for_model(m)
            if avail:
                row += _dim("(pending)", use_color).rjust(10)
            else:
                row += _dim("SKIP(RAM)", use_color).rjust(10)

        print(row)

    print()

    # Legend
    if use_color:
        print(f"  {_GREEN}Green{_RESET}: >2% speedup  "
              f"{_YELLOW}Yellow{_RESET}: neutral  "
              f"{_RED}Red{_RESET}: >2% regression  "
              f"{_DIM}Grey{_RESET}: skip/excluded")
    print()


def _available_for_model(m: ModelInfo) -> bool:
    """Quick check if model could fit on any known tier."""
    return bool(m.fits_on)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def to_csv(ledger_path: str | None = None, output: io.TextIOBase | None = None) -> str:
    """Export all ledger entries as CSV. Returns CSV string and optionally writes to output."""
    entries = list(latest(ledger_path).values())
    entries.sort(key=lambda e: (e.model_family, e.model_id))

    buf = io.StringIO()
    if not entries:
        # Still write header from catalog
        catalog = load_catalog()
        writer = csv.writer(buf)
        writer.writerow([
            "model_id", "family", "architecture", "quant", "storage_gb",
            "expected_patterns", "excluded_patterns", "status",
        ])
        for m in catalog:
            writer.writerow([
                m.model_id, m.family, m.architecture, m.quant, m.storage_gb,
                ";".join(m.expected_patterns),
                ";".join(f"{k}={v}" for k, v in m.excluded_patterns.items()),
                "pending",
            ])
    else:
        fieldnames = [
            "model_id", "model_family", "architecture", "patterns_applied",
            "hardware", "fidelity", "fidelity_detail", "modules_patched",
            "decode_tps_baseline", "decode_tps_patched", "decode_speedup",
            "prefill_tps_baseline", "prefill_tps_patched", "prefill_change",
            "peak_mem_baseline_gb", "peak_mem_patched_gb",
            "max_tokens", "gen_tokens", "runs",
            "mlx_version", "zmlx_version", "custom_mlx", "timestamp",
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            d = e.to_dict()
            d["patterns_applied"] = ";".join(d["patterns_applied"])
            row = {k: d.get(k, "") for k in fieldnames}
            writer.writerow(row)

    result = buf.getvalue()
    if output is not None:
        output.write(result)  # type: ignore[arg-type]
    return result


# ---------------------------------------------------------------------------
# HTML export (self-contained)
# ---------------------------------------------------------------------------

def to_html(ledger_path: str | None = None) -> str:
    """Generate a self-contained HTML page with the matrix table."""
    snap = snapshot(ledger_path)
    catalog = load_catalog()
    entry_by_model = {e.model_id: e for e in snap.entries}

    date = snap.date[:10] if snap.date else datetime.now().strftime("%Y-%m-%d")

    env_parts = []
    if snap.entries:
        e0 = snap.entries[0]
        if e0.mlx_version:
            env_parts.append(f"MLX {e0.mlx_version}")
        if e0.zmlx_version:
            env_parts.append(f"ZMLX {e0.zmlx_version}")
        env_parts.append(f"Custom MLX: {'Yes' if e0.custom_mlx else 'No'}")

    pattern_cols = ["moe_mlp", "swiglu_mlp", "geglu_mlp"]

    rows_html = []
    for m in catalog:
        entry = entry_by_model.get(m.model_id)
        size = f"{m.storage_gb:.0f}GB" if m.storage_gb >= 1 else f"{m.storage_gb:.1f}GB"

        cells = [
            f"<td>{m.family}</td>",
            f"<td>{m.display_name}</td>",
            f"<td>{m.architecture}</td>",
            f"<td>{m.quant}</td>",
            f"<td class='num'>{size}</td>",
        ]

        for p in pattern_cols:
            if m.excluded_patterns and p in m.excluded_patterns:
                cells.append("<td class='excl'>EXCL</td>")
            elif p == "moe_mlp" and m.architecture != "moe":
                cells.append("<td class='na'>--</td>")
            elif entry and entry.decode_speedup:
                pct = (entry.decode_speedup - 1.0) * 100
                cls = "good" if pct > 2 else ("bad" if pct < -2 else "neutral")
                cells.append(f"<td class='{cls}'>{pct:+.1f}%</td>")
            else:
                cells.append("<td class='na'>--</td>")

        if entry:
            cls = "pass" if entry.fidelity == "PASS" else "fail"
            cells.append(f"<td class='{cls}'>{entry.fidelity}</td>")
        else:
            cells.append("<td class='na'>pending</td>")

        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    th_patterns = "".join(f"<th>{p}</th>" for p in pattern_cols)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>ZMLX Test Matrix - {date}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2em; }}
h1 {{ font-size: 1.4em; }}
.env {{ color: #666; margin-bottom: 1em; }}
table {{ border-collapse: collapse; font-size: 0.9em; }}
th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; }}
th {{ background: #f5f5f5; position: sticky; top: 0; }}
.num {{ text-align: right; }}
.good {{ background: #d4edda; text-align: right; }}
.bad {{ background: #f8d7da; text-align: right; }}
.neutral {{ background: #fff3cd; text-align: right; }}
.excl {{ background: #e9ecef; color: #666; text-align: center; }}
.na {{ color: #999; text-align: center; }}
.pass {{ color: #28a745; font-weight: bold; }}
.fail {{ color: #dc3545; font-weight: bold; }}
</style></head><body>
<h1>ZMLX Test Matrix</h1>
<div class="env">{' | '.join(env_parts)} | {date}</div>
<table>
<thead><tr>
<th>Family</th><th>Model</th><th>Arch</th><th>Quant</th><th>Size</th>
{th_patterns}
<th>Fidelity</th>
</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
</body></html>"""
    return html
