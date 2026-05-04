#!/usr/bin/env python3
"""Full per-rank kernel breakdown for one NVTX iteration in a multi-rank nsys trace.

Usage:
    breakdown_full.py <report-name> [<nvtx-range>] [<rank>] [--min-pct N] [--no-color]

Examples:
    breakdown_full.py vllm_omni_pp8_encoder_compile_cl3d
    breakdown_full.py vllm_omni_pp8_encoder_compile_cl3d encode_iter_1 0
    breakdown_full.py vllm_omni_pp8_decoder_compile_cl3d decode_iter_2 3 --min-pct 0.05

Reads `<repo>/reports-fp32/<report-name>.{nsys-rep,sqlite}`. If the .sqlite is
missing it is auto-generated.

Columns: Time%, Total, Time(s), GEMM, Attention, Others, Instances,
Avg, Med, Min, Max, StdDev, Name. Per-kernel time goes into exactly one of
GEMM / Attention / Others depending on a name-pattern match.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports-fp32"


def fmt_t(ns: float) -> str:
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns/1e3:.3f} μs"
    if ns < 1e9:
        return f"{ns/1e6:.3f} ms"
    return f"{ns/1e9:.3f} s"


def categorize(name: str) -> str:
    n = name.lower()
    if "fmha" in n or "attention" in n or "flash_attn" in n or "flash_fwd" in n:
        return "attention"
    if "gemm" in n or "cutlass" in n:
        return "gemm"
    return "other"


def ensure_sqlite(rep: Path, sq: Path) -> None:
    if sq.exists() and sq.stat().st_mtime >= rep.stat().st_mtime:
        return
    nsys = shutil.which("nsys")
    if nsys is None:
        sys.exit("nsys not on PATH; cannot export sqlite")
    print(f"[exporting {rep.name} -> sqlite ...]", file=sys.stderr)
    subprocess.run(
        [nsys, "export", "--type", "sqlite", "--force-overwrite=true",
         "--output", str(sq), str(rep)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("nvtx", nargs="?", default="encode_iter_1")
    ap.add_argument("rank", nargs="?", type=int, default=0)
    ap.add_argument("--min-pct", type=float, default=0.0,
                    help="Drop rows below this %% of total (default 0 = show all).")
    ap.add_argument("--csv", nargs="?", const="<auto>", default=None,
                    help="Write CSV (every kernel). With no value, auto-name "
                         "<report>.<nvtx>.rank<N>.csv next to the report.")
    args = ap.parse_args()

    rep = REPORTS / f"{args.report}.nsys-rep"
    sq = REPORTS / f"{args.report}.sqlite"
    if not rep.exists():
        sys.exit(f"no such report: {rep}")
    ensure_sqlite(rep, sq)

    con = sqlite3.connect(str(sq))
    con.row_factory = sqlite3.Row

    rows = con.execute(
        """
        SELECT globalTid AS gtid, start, end
        FROM NVTX_EVENTS
        WHERE text = ?
        ORDER BY globalTid
        """,
        (args.nvtx,),
    ).fetchall()
    if not rows:
        sys.exit(f"no NVTX range named {args.nvtx!r} in {rep.name}")
    if args.rank >= len(rows):
        sys.exit(f"rank {args.rank} out of range (have {len(rows)} ranks)")

    iter_row = rows[args.rank]
    s, e = iter_row["start"], iter_row["end"]
    pid_high = iter_row["gtid"] >> 24
    wall_ns = e - s

    kernels = con.execute(
        """
        SELECT s.value AS name, k.end - k.start AS dur
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        WHERE k.start >= ? AND k.end <= ?
          AND (k.globalPid >> 24) = ?
        """,
        (s, e, pid_high),
    ).fetchall()
    if not kernels:
        sys.exit(f"no kernels found in {args.nvtx} on rank {args.rank}")

    grouped: dict[str, list[int]] = {}
    for r in kernels:
        grouped.setdefault(r["name"], []).append(r["dur"])

    total_ns = sum(sum(v) for v in grouped.values())
    rows_out = []
    for name, durs in grouped.items():
        tot = sum(durs)
        rows_out.append({
            "pct": 100.0 * tot / total_ns,
            "tot_ns": tot,
            "n": len(durs),
            "avg": tot / len(durs),
            "med": statistics.median(durs),
            "min": min(durs),
            "max": max(durs),
            "std": statistics.pstdev(durs) if len(durs) > 1 else 0,
            "name": name,
        })
    rows_out.sort(key=lambda r: -r["tot_ns"])

    print(f"=== {args.report} :: {args.nvtx} :: rank {args.rank} ===")
    print(f"wall:     {fmt_t(wall_ns)}")
    print(f"GPU busy: {fmt_t(total_ns)}  ({100*total_ns/wall_ns:.1f}% of wall)")
    print(f"idle gap: {fmt_t(wall_ns - total_ns)}  "
          f"({100*(wall_ns - total_ns)/wall_ns:.1f}% of wall)")
    print(f"distinct kernels: {len(rows_out)}, total launches: {sum(r['n'] for r in rows_out)}")
    print()

    if args.csv is not None:
        out = (REPORTS / f"{args.report}.{args.nvtx}.rank{args.rank}.csv"
               if args.csv == "<auto>" else Path(args.csv))
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "time", "total_time", "instances",
                "avg", "med", "min", "max", "stddev", "name",
            ])
            for r in rows_out:
                w.writerow([
                    f"{r['pct']:.2f}%", fmt_t(r['tot_ns']), r['n'],
                    fmt_t(r['avg']), fmt_t(r['med']),
                    fmt_t(r['min']), fmt_t(r['max']), fmt_t(r['std']),
                    r['name'],
                ])
        print(f"[wrote {out}  ({len(rows_out)} rows)]", file=sys.stderr)
        return 0

    headers = ["Time", "Total Time", "Instances", "Avg", "Med",
               "Min", "Max", "StdDev", "Name"]
    fmt_row = lambda r: [
        f"{r['pct']:.2f}%",
        fmt_t(r['tot_ns']),
        f"{r['n']}",
        fmt_t(r['avg']),
        fmt_t(r['med']),
        fmt_t(r['min']),
        fmt_t(r['max']),
        fmt_t(r['std']),
        r['name'],
    ]
    table = [headers] + [fmt_row(r) for r in rows_out
                         if r["pct"] >= args.min_pct]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers) - 1)]
    for row in table:
        out = "  ".join(c.ljust(w) for c, w in zip(row[:-1], widths))
        print(f"{out}  {row[-1]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
