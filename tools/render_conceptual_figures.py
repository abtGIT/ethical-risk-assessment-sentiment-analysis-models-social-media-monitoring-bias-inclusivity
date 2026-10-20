"""Render published Graphviz diagrams from the checked-in DOT sources.

By default, print the planned outputs without modifying any assets.
Use --write to render them, and review the resulting diffs before committing.
Figure 3.4 is Matplotlib-based; see figures/diagrams/source/fig34.py.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures" / "diagrams" / "source"
OUTPUT = ROOT / "figures" / "diagrams"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", help="One figure ID, e.g. Fig3_10; default: all DOT sources")
    ap.add_argument("--write", action="store_true", help="Actually overwrite PNG/SVG assets")
    args = ap.parse_args()

    dots = sorted(SOURCE.glob("Fig*.dot"))
    if args.figure:
        dots = [path for path in dots if path.stem == args.figure]
    if not dots:
        ap.error("No matching Graphviz source (.dot) files")

    if args.write and not shutil.which("dot"):
        ap.error("Graphviz 'dot' is required for --write")

    for src in dots:
        for fmt in ("png", "svg"):
            dst = OUTPUT / (src.stem + "." + fmt)
            cmd = ["dot", f"-T{fmt}", str(src), "-o", str(dst)]
            if fmt == "png":
                cmd.insert(2, "-Gdpi=300")
            if not args.write:
                print(f"PLAN {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
            else:
                subprocess.run(cmd, check=True)
                print(f"WROTE {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
