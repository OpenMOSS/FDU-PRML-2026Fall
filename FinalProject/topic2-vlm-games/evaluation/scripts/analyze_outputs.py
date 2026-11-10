#!/usr/bin/env python3
"""Optional descriptive output report using the same course scoring rules."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from score_outputs import evaluate_outputs


def analyze_outputs(inputs: Sequence[Path], output_dir: Path) -> dict[str, object]:
    """Report measured correctness, length and format; infer no reasoning quality."""
    summary = evaluate_outputs(inputs, output_dir, overwrite=True)
    return {"rows": summary["overall"]["samples"],
            "output_dir": str(output_dir), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_outputs(args.input, args.output_dir)
    print(f"Analyzed {result['rows']} rows: {result['output_dir']}")


if __name__ == "__main__":
    main()
