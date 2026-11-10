#!/usr/bin/env python3
"""Evaluate game-reasoning outputs into a run-local, game-oriented artifact layout.

This is a new output-layout entry point.  The existing
``score_outputs.py`` remains unchanged and is still available for
the legacy flat evaluation layout.  Both entry points use the same answer
rules and row-level evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_outputs import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    _run_manifests,
    _validate_expectations,
    _validate_run_manifests,
    grouped_summaries,
    load_and_evaluate,
    make_diagnostics,
    make_report,
    summarize,
)
from answers import EVALUATION_RULES_VERSION  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        _write_text(path, "")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _safe_game_dir(game_name: str) -> str:
    value = re.sub(r"[^\w.-]+", "-", str(game_name).strip(), flags=re.UNICODE)
    value = value.strip(".-")
    return value or "game"


def default_output_dir(inputs: Sequence[Path]) -> Path:
    """Choose a colocated output only when the input has one clear run owner."""
    if len(inputs) != 1:
        raise ValueError(
            "--output-dir is required when evaluating multiple input paths; "
            "the result must not be silently assigned to one input run"
        )
    input_path = inputs[0].expanduser().resolve()
    if input_path.is_dir() and (input_path / "games").is_dir():
        return input_path / "evaluation"
    if input_path.is_dir() and input_path.name == "games":
        return input_path.parent / "evaluation"
    if input_path.is_file() and input_path.parent.name == "games":
        return input_path.parent.parent / "evaluation"
    raise ValueError(
        "Cannot infer a colocated evaluation directory from the input; "
        "use --output-dir explicitly"
    )


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _group_records_by_game(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["game_name"])].append(row)
    return dict(sorted(groups.items()))


def _add_category_macro_metrics(
    by_game: list[dict[str, Any]], by_category: list[dict[str, Any]]
) -> None:
    for category in by_category:
        category_games = [
            row for row in by_game
            if row["cognitive_category"] == category["cognitive_category"]
        ]
        category["games"] = len(category_games)
        for metric in (
            "accuracy",
            "relaxed_accuracy",
            "conservative_relaxed_accuracy",
        ):
            category[f"macro_game_{metric}"] = round(
                mean(float(row[metric]) for row in category_games), 6
            ) if category_games else 0.0


def _per_game_summary(
    records: Sequence[dict[str, Any]],
    game_name: str,
    cognitive_category: str,
) -> dict[str, Any]:
    game_records = [row for row in records if row["game_name"] == game_name]
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_rules_version": EVALUATION_RULES_VERSION,
        "generated_at": utc_now(),
        "game_name": game_name,
        "cognitive_category": cognitive_category,
        "overall": summarize(game_records),
        "by_question": grouped_summaries(
            game_records,
            (
                "game_name",
                "cognitive_category",
                "question_id",
                "qa_type",
                "question_description",
            ),
        ),
        "by_qa_type": grouped_summaries(game_records, ("qa_type",)),
        "by_answer_type": grouped_summaries(game_records, ("answer_type",)),
        "by_answer_mode": grouped_summaries(game_records, ("answer_mode",)),
    }


def _per_game_report(summary: dict[str, Any]) -> str:
    overall = summary["overall"]

    def percent(value: float) -> str:
        return f"{value * 100:.2f}%"

    lines = [
        f"# {summary['game_name']} evaluation report",
        "",
        f"- Cognitive category: `{summary['cognitive_category']}`",
        f"- Samples: {overall['samples']:,}",
        f"- Strict accuracy: **{percent(overall['accuracy'])}** "
        f"({overall['correct']:,}/{overall['samples']:,})",
        f"- Content-relaxed accuracy: {percent(overall['relaxed_accuracy'])} "
        f"({overall['relaxed_correct']:,}/{overall['samples']:,})",
        f"- Conservative content-relaxed accuracy: "
        f"{percent(overall['conservative_relaxed_accuracy'])} "
        f"({overall['conservative_relaxed_correct']:,}/{overall['samples']:,})",
        f"- Answer extracted: {percent(overall['answer_extraction_rate'])}",
        f"- Final-line JSON: {percent(overall['final_line_json_rate'])}",
        f"- Truncated: {percent(overall['truncation_rate'])}",
        f"- Median completion tokens: {overall['median_completion_tokens']:,.0f}",
        "",
        "## By question",
        "",
        "| Question ID | QA type | Samples | Strict | Relaxed | Extracted | Truncated | Median tokens |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_question"]:
        lines.append(
            f"| {row['question_id']} | {row['qa_type']} | {row['samples']} | "
            f"{percent(row['accuracy'])} | {percent(row['relaxed_accuracy'])} | "
            f"{percent(row['answer_extraction_rate'])} | "
            f"{row['finish_length']} | {row['median_completion_tokens']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Metric notes",
            "",
            "- Strict accuracy is the formal answer-contract score.",
            "- Content-relaxed metrics are diagnostic comparisons for labeled choices.",
            "- Output-format compliance and answer correctness are reported separately.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_game_artifacts(
    output_dir: Path,
    game_name: str,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    category = str(records[0]["cognitive_category"])
    summary = _per_game_summary(records, game_name, category)
    game_dir = output_dir / _safe_game_dir(game_name)
    game_dir.mkdir(parents=True, exist_ok=True)

    write_json(game_dir / "summary.json", summary)
    write_json(game_dir / "overall.json", summary["overall"])
    write_json(game_dir / "by_question.json", summary["by_question"])
    write_json(game_dir / "by_qa_type.json", summary["by_qa_type"])
    write_json(game_dir / "by_answer_type.json", summary["by_answer_type"])
    write_json(game_dir / "by_answer_mode.json", summary["by_answer_mode"])
    write_json(game_dir / "diagnostics.json", make_diagnostics(records))
    write_jsonl(game_dir / "samples.jsonl", records)
    write_jsonl(game_dir / "errors.jsonl", (row for row in records if not row["correct"]))
    write_jsonl(
        game_dir / "relaxed_uplifts.jsonl",
        (row for row in records if row["option_content_match"]),
    )
    write_jsonl(
        game_dir / "ambiguous_relaxed_uplifts.jsonl",
        (row for row in records if row["option_content_ambiguous"]),
    )
    write_jsonl(
        game_dir / "format_violations.jsonl",
        (row for row in records if not row["final_line_json"]),
    )
    write_csv(game_dir / "by_question.csv", summary["by_question"])
    write_csv(game_dir / "by_qa_type.csv", summary["by_qa_type"])
    write_csv(game_dir / "by_answer_type.csv", summary["by_answer_type"])
    write_csv(game_dir / "by_answer_mode.csv", summary["by_answer_mode"])
    _write_text(game_dir / "report.md", _per_game_report(summary))
    return summary


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty (use --overwrite): {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def evaluate_run(
    inputs: Sequence[Path],
    output_dir: Path | None = None,
    *,
    expected_samples: int | None = None,
    expected_games: int | None = None,
    expected_samples_per_game: int | None = None,
    overwrite: bool = False,
    allow_mixed_configs: bool = False,
) -> dict[str, Any]:
    """Evaluate inputs and write a colocated, per-game artifact tree."""
    if not inputs:
        raise ValueError("At least one input path is required")
    input_paths = [Path(path).expanduser().resolve() for path in inputs]
    if output_dir is None:
        output_dir = default_output_dir(input_paths)

    input_files, input_metadata = _load_inputs(input_paths)
    output_dir = _prepare_output_dir(output_dir, overwrite)
    records, loaded_metadata = load_and_evaluate(input_files)
    # Use the loader's metadata because it is computed after JSONL validation.
    if loaded_metadata != input_metadata:
        input_metadata = loaded_metadata
    run_manifests = _run_manifests(input_files)
    if not allow_mixed_configs:
        _validate_run_manifests(run_manifests)
    _validate_expectations(
        records,
        expected_samples,
        expected_games,
        expected_samples_per_game,
    )

    by_game = grouped_summaries(records, ("game_name", "cognitive_category"))
    by_category = grouped_summaries(records, ("cognitive_category",))
    _add_category_macro_metrics(by_game, by_category)
    game_accuracy = {str(row["game_name"]): float(row["accuracy"]) for row in by_game}
    game_relaxed = {
        str(row["game_name"]): float(row["relaxed_accuracy"]) for row in by_game
    }
    game_conservative = {
        str(row["game_name"]): float(row["conservative_relaxed_accuracy"])
        for row in by_game
    }
    overall = summarize(records)
    overall["macro_game_accuracy"] = round(mean(game_accuracy.values()), 6)
    overall["macro_game_relaxed_accuracy"] = round(mean(game_relaxed.values()), 6)
    overall["macro_game_conservative_relaxed_accuracy"] = round(
        mean(game_conservative.values()), 6
    )
    summary = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_rules_version": EVALUATION_RULES_VERSION,
        "generated_at": utc_now(),
        "game_count": len(by_game),
        "overall": overall,
        "by_game": by_game,
        "by_cognitive_category": by_category,
        "by_qa_type": grouped_summaries(records, ("qa_type",)),
        "by_answer_type": grouped_summaries(records, ("answer_type",)),
        "by_answer_mode": grouped_summaries(records, ("answer_mode",)),
        "record_layout": "per_game",
    }

    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_rules_version": EVALUATION_RULES_VERSION,
        "generated_at": summary["generated_at"],
        "git_revision": _git_revision(),
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "answer_rules_sha256": sha256_file(ROOT / "answers.py"),
        "answer_metadata_sha256": sha256_file(ROOT / "metadata.py"),
        "inputs": input_metadata,
        "run_manifests": run_manifests,
        "expectations": {
            "samples": expected_samples,
            "games": expected_games,
            "samples_per_game": expected_samples_per_game,
        },
        "record_layout": "per_game",
        "allow_mixed_configs": allow_mixed_configs,
    }

    write_json(output_dir / "evaluation_manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "overall.json", overall)
    write_json(output_dir / "by_game.json", by_game)
    write_json(output_dir / "by_cognitive_category.json", by_category)
    write_json(output_dir / "by_qa_type.json", summary["by_qa_type"])
    write_json(output_dir / "by_answer_type.json", summary["by_answer_type"])
    write_json(output_dir / "by_answer_mode.json", summary["by_answer_mode"])
    write_json(output_dir / "diagnostics.json", make_diagnostics(records))
    write_csv(output_dir / "by_game.csv", by_game)
    write_csv(output_dir / "by_cognitive_category.csv", by_category)
    write_csv(output_dir / "by_qa_type.csv", summary["by_qa_type"])
    write_csv(output_dir / "by_answer_type.csv", summary["by_answer_type"])
    write_csv(output_dir / "by_answer_mode.csv", summary["by_answer_mode"])
    _write_text(output_dir / "report.md", make_report(summary))

    game_groups = _group_records_by_game(records)
    game_dirs = [_safe_game_dir(game_name) for game_name in game_groups]
    if len(game_dirs) != len(set(game_dirs)):
        raise ValueError(
            "Game names map to the same evaluation directory; use explicit "
            "input grouping or adjust the game directory naming rule"
        )
    for game_name, game_records in game_groups.items():
        _write_game_artifacts(output_dir, game_name, game_records)
    return summary


def _load_inputs(
    inputs: Sequence[Path],
) -> tuple[list[Path], list[dict[str, Any]]]:
    # Keep input expansion in one place so the output manifest can expose the
    # exact files that were evaluated, including their size and hash.
    from scripts.score_outputs import resolve_input_files

    input_files = resolve_input_files(inputs)
    metadata = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in input_files
    ]
    return input_files, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="JSONL files, a games directory, or an inference run directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Evaluation directory; defaults to <single-run>/evaluation",
    )
    parser.add_argument("--expect-samples", type=int)
    parser.add_argument("--expect-games", type=int)
    parser.add_argument("--expect-samples-per-game", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-mixed-configs",
        action="store_true",
        help="Allow input run manifests with different configuration fingerprints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_run(
        args.input,
        args.output_dir,
        expected_samples=args.expect_samples,
        expected_games=args.expect_games,
        expected_samples_per_game=args.expect_samples_per_game,
        overwrite=args.overwrite,
        allow_mixed_configs=args.allow_mixed_configs,
    )
    overall = summary["overall"]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_output_dir(args.input).resolve()
    )
    print(
        f"Evaluated {overall['samples']} samples across {summary['game_count']} games"
    )
    print(
        f"Accuracy: {overall['accuracy'] * 100:.2f}% "
        f"({overall['correct']}/{overall['samples']}); "
        f"macro game accuracy: {overall['macro_game_accuracy'] * 100:.2f}%"
    )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
