#!/usr/bin/env python3
"""Evaluate game-reasoning inference JSONL files and write reproducible benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from answers import (  # noqa: E402
    EVALUATION_RULES_VERSION,
    answer_details,
)
from metadata import cognitive_category  # noqa: E402


EVALUATION_SCHEMA_VERSION = 3
MATCH_RULES = (
    "exact",
    "option_value",
    "numeric",
    "integer_structure",
    "singleton_coordinate",
    "pyramid_position",
    "spider_card",
)
REQUIRED_FIELDS = {"uid", "game_name", "question_id", "reference_answer", "status"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[int], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def resolve_input_files(inputs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in inputs:
        path = raw_path.resolve()
        if path.is_file():
            if path.suffix != ".jsonl":
                raise ValueError(f"Input file is not JSONL: {path}")
            files.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if (path / "games").is_dir():
            candidates = sorted((path / "games").glob("*.jsonl"))
        elif path.name == "games":
            candidates = sorted(path.glob("*.jsonl"))
        else:
            candidates = sorted(path.rglob("games/*.jsonl"))
        if not candidates:
            raise ValueError(f"No games/*.jsonl files found under: {path}")
        files.extend(candidate.resolve() for candidate in candidates)

    unique = sorted(set(files), key=str)
    if not unique:
        raise ValueError("No input JSONL files were selected")
    return unique


def _token_count(row: dict[str, Any], name: str, source: str) -> int:
    value = (row.get("usage") or {}).get(name, 0)
    try:
        result = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid usage.{name} at {source}: {value!r}") from exc
    if result < 0:
        raise ValueError(f"Negative usage.{name} at {source}: {result}")
    return result


def _format_issue(details: dict[str, Any]) -> str | None:
    if details["final_line_json"]:
        return None
    forms = {
        "standalone_json": "json_not_final_line",
        "embedded_json": "embedded_json_only",
        "relaxed_answer_field": "non_json_answer_field",
        "missing": "missing_answer",
    }
    return forms.get(str(details["answer_form"]), "unknown_format")


def evaluate_row(
    row: dict[str, Any], source_file: Path, line_number: int
) -> dict[str, Any]:
    source = f"{source_file}:{line_number}"
    missing = REQUIRED_FIELDS.difference(row)
    if missing:
        raise ValueError(f"Missing fields {sorted(missing)} at {source}")
    uid = str(row["uid"]).strip()
    game_name = str(row["game_name"]).strip()
    if not uid or not game_name:
        raise ValueError(f"Empty uid or game_name at {source}")

    details = answer_details(row)
    status_ok = row.get("status") == "ok"
    correct = bool(status_ok and details["correct"])
    relaxed_correct = bool(status_ok and details["relaxed_correct"])
    conservative_relaxed_correct = bool(
        status_ok and details["conservative_relaxed_correct"]
    )
    if not status_ok:
        error_type = "inference_error"
    elif details["answer_form"] == "missing":
        error_type = "missing_answer"
    elif not correct:
        error_type = "wrong_answer"
    else:
        error_type = None

    answer_mode = str(details["answer_mode"])
    multiple_choice = answer_mode != "free_form"
    return {
        "uid": uid,
        "data_id": row.get("data_id"),
        "game_name": game_name,
        "cognitive_category": cognitive_category(game_name, row.get("game_type")),
        "question_id": row.get("question_id"),
        "question_description": row.get("question_description"),
        "qa_type": row.get("qa_type"),
        "qa_level": row.get("qa_level"),
        "plot_level": row.get("plot_level"),
        "answer_type": "multiple_choice" if multiple_choice else "free_form",
        "answer_mode": answer_mode,
        "answer_mode_source": details["answer_mode_source"],
        "label_scheme": details["label_scheme"],
        "model_answer": details["model_answer"],
        "reference_answer": details["reference_answer"],
        "correct": correct,
        "match_rule": details["match_rule"] if correct else None,
        "relaxed_correct": relaxed_correct,
        "conservative_relaxed_correct": conservative_relaxed_correct,
        "option_content_match": bool(status_ok and details["option_content_match"]),
        "option_content_ambiguous": bool(
            status_ok and details["option_content_ambiguous"]
        ),
        "answer_form": details["answer_form"],
        "answer_extracted": details["answer_form"] != "missing",
        "json_compliant": details["json_compliant"],
        "final_line_json": details["final_line_json"],
        "format_issue": _format_issue(details),
        "error_type": error_type,
        "status": row.get("status"),
        "finish_reason": row.get("finish_reason"),
        "prompt_tokens": _token_count(row, "prompt_tokens", source),
        "completion_tokens": _token_count(row, "completion_tokens", source),
        "total_tokens": _token_count(row, "total_tokens", source),
        "source_file": str(source_file),
        "source_line": line_number,
    }


def load_and_evaluate(
    input_files: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    seen_uids: dict[str, str] = {}
    for path in input_files:
        row_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                record = evaluate_row(row, path, line_number)
                previous = seen_uids.get(record["uid"])
                if previous is not None:
                    raise ValueError(
                        f"Duplicate uid {record['uid']!r} at {path}:{line_number}; first seen at {previous}"
                    )
                seen_uids[record["uid"]] = f"{path}:{line_number}"
                records.append(record)
                row_count += 1
        inputs.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "rows": row_count,
            }
        )
    return records, inputs


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    correct = sum(bool(row["correct"]) for row in records)
    relaxed_correct = sum(bool(row["relaxed_correct"]) for row in records)
    conservative_relaxed_correct = sum(
        bool(row["conservative_relaxed_correct"]) for row in records
    )
    extracted = sum(bool(row["answer_extracted"]) for row in records)
    json_present = sum(bool(row["json_compliant"]) for row in records)
    final_line = sum(bool(row["final_line_json"]) for row in records)
    status_ok = sum(row["status"] == "ok" for row in records)
    truncated = sum(row["finish_reason"] == "length" for row in records)
    completion = [int(row["completion_tokens"]) for row in records]
    answer_forms = Counter(str(row["answer_form"]) for row in records)
    error_types = Counter(
        str(row["error_type"]) for row in records if row["error_type"]
    )
    match_rules = Counter(
        str(row["match_rule"]) for row in records if row["match_rule"]
    )

    result = {
        "samples": count,
        "correct": correct,
        "incorrect": count - correct,
        "accuracy": ratio(correct, count),
        "relaxed_correct": relaxed_correct,
        "relaxed_accuracy": ratio(relaxed_correct, count),
        "relaxed_uplift": relaxed_correct - correct,
        "conservative_relaxed_correct": conservative_relaxed_correct,
        "conservative_relaxed_accuracy": ratio(conservative_relaxed_correct, count),
        "conservative_relaxed_uplift": conservative_relaxed_correct - correct,
        "ambiguous_option_content_uplift": sum(
            bool(row["option_content_match"] and row["option_content_ambiguous"])
            for row in records
        ),
        "status_ok": status_ok,
        "inference_errors": count - status_ok,
        "answer_extracted": extracted,
        "answer_extraction_rate": ratio(extracted, count),
        "json_answer_present": json_present,
        "json_answer_rate": ratio(json_present, count),
        "final_line_json": final_line,
        "final_line_json_rate": ratio(final_line, count),
        "correct_with_final_line_json": sum(
            bool(row["correct"] and row["final_line_json"]) for row in records
        ),
        "correct_without_final_line_json": sum(
            bool(row["correct"] and not row["final_line_json"]) for row in records
        ),
        "finish_length": truncated,
        "truncation_rate": ratio(truncated, count),
        "multiple_choice": sum(
            row["answer_type"] == "multiple_choice" for row in records
        ),
        "free_form": sum(row["answer_type"] == "free_form" for row in records),
        "standalone_json": answer_forms["standalone_json"],
        "embedded_json": answer_forms["embedded_json"],
        "relaxed_answer_field": answer_forms["relaxed_answer_field"],
        "missing_answer": answer_forms["missing"],
        "wrong_answer": error_types["wrong_answer"],
        "mean_completion_tokens": round(mean(completion), 3) if completion else 0.0,
        "median_completion_tokens": round(median(completion), 3) if completion else 0.0,
        "p90_completion_tokens": round(percentile(completion, 0.9), 3),
        "max_completion_tokens": max(completion, default=0),
    }
    result.update(
        {
            f"answer_mode_{mode}": sum(row["answer_mode"] == mode for row in records)
            for mode in ("number_label", "letter_label", "option_value", "free_form")
        }
    )
    result.update({f"match_{rule}": match_rules[rule] for rule in MATCH_RULES})
    return result


def grouped_summaries(
    records: Sequence[dict[str, Any]], dimensions: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[tuple(row.get(dimension) for dimension in dimensions)].append(row)
    summaries = []
    for key, rows in groups.items():
        entry = {dimension: value for dimension, value in zip(dimensions, key)}
        entry.update(summarize(rows))
        summaries.append(entry)
    return sorted(
        summaries, key=lambda row: tuple(str(row.get(key) or "") for key in dimensions)
    )


def _run_manifests(input_files: Sequence[Path]) -> list[dict[str, Any]]:
    manifests = sorted({path.parent.parent / "manifest.json" for path in input_files})
    result = []
    for path in manifests:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid run manifest {path}: {exc}") from exc
        result.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "run_name": value.get("run_name"),
                "config_fingerprint": value.get("config_fingerprint"),
                "config": value.get("config"),
            }
        )
    return result


def _validate_run_manifests(manifests: Sequence[dict[str, Any]]) -> None:
    fingerprints = {
        str(manifest["config_fingerprint"])
        for manifest in manifests
        if manifest.get("config_fingerprint")
    }
    if len(fingerprints) > 1:
        raise ValueError(
            "Input files come from incompatible inference configurations: "
            f"{sorted(fingerprints)}"
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


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    _write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        _write_text(path, "")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def make_report(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# game-reasoning evaluation report",
        "",
        f"- Evaluation rules: `{summary['evaluation_rules_version']}`",
        f"- Samples / games: {overall['samples']:,} / {summary['game_count']}",
        f"- Micro accuracy: **{_percent(overall['accuracy'])}** ({overall['correct']:,}/{overall['samples']:,})",
        f"- Content-relaxed accuracy: **{_percent(overall['relaxed_accuracy'])}** "
        f"({overall['relaxed_correct']:,}/{overall['samples']:,}; "
        f"+{overall['relaxed_uplift']:,})",
        f"- Conservative content-relaxed accuracy: "
        f"**{_percent(overall['conservative_relaxed_accuracy'])}** "
        f"({overall['conservative_relaxed_correct']:,}/{overall['samples']:,}; "
        f"+{overall['conservative_relaxed_uplift']:,})",
        f"- Macro game accuracy: **{_percent(overall['macro_game_accuracy'])}**",
        f"- Macro game content-relaxed accuracy: "
        f"**{_percent(overall['macro_game_relaxed_accuracy'])}**",
        f"- Macro game conservative relaxed accuracy: "
        f"**{_percent(overall['macro_game_conservative_relaxed_accuracy'])}**",
        f"- Final-line JSON: {_percent(overall['final_line_json_rate'])}",
        f"- Any valid answer JSON: {_percent(overall['json_answer_rate'])}",
        f"- Answer extracted: {_percent(overall['answer_extraction_rate'])}",
        f"- Truncated: {_percent(overall['truncation_rate'])}",
        "",
        "## Answer modes",
        "",
        "| Mode | Samples | Strict | Relaxed | Conservative relaxed |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["by_answer_mode"]:
        lines.append(
            f"| {row['answer_mode']} | {row['samples']} | {_percent(row['accuracy'])} | "
            f"{_percent(row['relaxed_accuracy'])} | "
            f"{_percent(row['conservative_relaxed_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Cognitive categories",
            "",
            "| Category | Games | Samples | Strict | Relaxed | Conservative relaxed | Ambiguous uplift | Final-line JSON | Truncated |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["by_cognitive_category"]:
        lines.append(
            f"| {row['cognitive_category']} | {row['games']} | {row['samples']} | "
            f"{_percent(row['accuracy'])} | {_percent(row['relaxed_accuracy'])} | "
            f"{_percent(row['conservative_relaxed_accuracy'])} | "
            f"{row['ambiguous_option_content_uplift']} | "
            f"{_percent(row['final_line_json_rate'])} | {_percent(row['truncation_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Games",
            "",
            "| Game | Category | N | Strict | Relaxed | Conservative relaxed | Uplift | Ambiguous | Final-line JSON | Missing | Truncated | Median tokens |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["by_game"]:
        lines.append(
            f"| {row['game_name']} | {row['cognitive_category']} | {row['samples']} | "
            f"{_percent(row['accuracy'])} | {_percent(row['relaxed_accuracy'])} | "
            f"{_percent(row['conservative_relaxed_accuracy'])} | "
            f"{row['relaxed_uplift']} | {row['ambiguous_option_content_uplift']} | "
            f"{_percent(row['final_line_json_rate'])} | {row['missing_answer']} | "
            f"{row['finish_length']} | {row['median_completion_tokens']:,.0f} |"
        )

    lines.extend(
        [
            "",
            "## Answer diagnostics",
            "",
            f"- Standalone / embedded JSON: {overall['standalone_json']:,} / {overall['embedded_json']:,}",
            f"- Relaxed answer fields / missing answers: {overall['relaxed_answer_field']:,} / {overall['missing_answer']:,}",
            f"- Correct answers violating final-line JSON: {overall['correct_without_final_line_json']:,}",
            f"- Exact / semantic-normalized matches: {overall['match_exact']:,} / "
            f"{overall['correct'] - overall['match_exact']:,}",
            "",
            "## Metric definitions",
            "",
            "- Accuracy uses every input row as the denominator. Inference errors and missing answers are incorrect.",
            "- Micro accuracy is sample-weighted. Macro game accuracy is the unweighted mean of game accuracies.",
            "- Final-line JSON requires the last non-empty response line to be a complete JSON object containing `answer`.",
            "- Any valid answer JSON includes standalone, embedded, multiline, and fenced JSON objects.",
            "- Answer correctness and output-format compliance are independent metrics.",
            "- Strict label accuracy compares the extracted label with the official answer label.",
            "- Unlabeled option-value answers are compared directly with the official listed value.",
            "- Content-relaxed accuracy additionally accepts an exact, unique option-content match only for labeled choices.",
            "- Conservative content-relaxed accuracy excludes values that are also valid labels for a different option.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_diagnostics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build compact behavior counts without duplicating all sample records."""
    return {
        "answer_modes": dict(
            sorted(Counter(row["answer_mode"] for row in records).items())
        ),
        "answer_mode_sources": dict(
            sorted(Counter(row["answer_mode_source"] for row in records).items())
        ),
        "label_schemes": dict(
            sorted(Counter(str(row["label_scheme"]) for row in records).items())
        ),
        "answer_forms": dict(
            sorted(Counter(row["answer_form"] for row in records).items())
        ),
        "match_rules": dict(
            sorted(
                Counter(
                    row["match_rule"] for row in records if row["match_rule"]
                ).items()
            )
        ),
        "error_types": dict(
            sorted(
                Counter(
                    row["error_type"] for row in records if row["error_type"]
                ).items()
            )
        ),
        "format_issues": dict(
            sorted(
                Counter(
                    row["format_issue"] for row in records if row["format_issue"]
                ).items()
            )
        ),
        "finish_reasons": dict(
            sorted(Counter(str(row["finish_reason"]) for row in records).items())
        ),
        "option_content": {
            "uplifts": sum(bool(row["option_content_match"]) for row in records),
            "ambiguous": sum(bool(row["option_content_ambiguous"]) for row in records),
        },
    }


def _validate_expectations(
    records: Sequence[dict[str, Any]],
    expected_samples: int | None,
    expected_games: int | None,
    expected_samples_per_game: int | None,
) -> None:
    if expected_samples is not None and len(records) != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples, found {len(records)}")
    game_counts = Counter(str(row["game_name"]) for row in records)
    if expected_games is not None and len(game_counts) != expected_games:
        raise ValueError(f"Expected {expected_games} games, found {len(game_counts)}")
    if expected_samples_per_game is not None:
        mismatches = {
            game: count
            for game, count in sorted(game_counts.items())
            if count != expected_samples_per_game
        }
        if mismatches:
            raise ValueError(
                f"Expected {expected_samples_per_game} samples per game; mismatches: {mismatches}"
            )


def evaluate_outputs(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    expected_samples: int | None = None,
    expected_games: int | None = None,
    expected_samples_per_game: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_files = resolve_input_files(inputs)
    output_dir = output_dir.resolve()
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty (use --overwrite): {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    records, input_metadata = load_and_evaluate(input_files)
    run_manifests = _run_manifests(input_files)
    _validate_run_manifests(run_manifests)
    _validate_expectations(
        records,
        expected_samples,
        expected_games,
        expected_samples_per_game,
    )

    by_game = grouped_summaries(records, ("game_name", "cognitive_category"))
    by_category = grouped_summaries(records, ("cognitive_category",))
    game_accuracy = {str(row["game_name"]): float(row["accuracy"]) for row in by_game}
    game_relaxed_accuracy = {
        str(row["game_name"]): float(row["relaxed_accuracy"]) for row in by_game
    }
    game_conservative_accuracy = {
        str(row["game_name"]): float(row["conservative_relaxed_accuracy"])
        for row in by_game
    }
    for row in by_category:
        category_games = [
            str(game["game_name"])
            for game in by_game
            if game["cognitive_category"] == row["cognitive_category"]
        ]
        row["games"] = len(category_games)
        row["macro_game_accuracy"] = (
            round(mean(game_accuracy[game] for game in category_games), 6)
            if category_games
            else 0.0
        )
        row["macro_game_relaxed_accuracy"] = (
            round(mean(game_relaxed_accuracy[game] for game in category_games), 6)
            if category_games
            else 0.0
        )
        row["macro_game_conservative_relaxed_accuracy"] = (
            round(mean(game_conservative_accuracy[game] for game in category_games), 6)
            if category_games
            else 0.0
        )

    overall = summarize(records)
    overall["macro_game_accuracy"] = (
        round(mean(game_accuracy.values()), 6) if game_accuracy else 0.0
    )
    overall["macro_game_relaxed_accuracy"] = (
        round(mean(game_relaxed_accuracy.values()), 6) if game_relaxed_accuracy else 0.0
    )
    overall["macro_game_conservative_relaxed_accuracy"] = (
        round(mean(game_conservative_accuracy.values()), 6)
        if game_conservative_accuracy
        else 0.0
    )
    summary = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_rules_version": EVALUATION_RULES_VERSION,
        "generated_at": utc_now(),
        "game_count": len(by_game),
        "overall": overall,
        "by_cognitive_category": by_category,
        "by_game": by_game,
        "by_qa_type": grouped_summaries(records, ("qa_type",)),
        "by_answer_type": grouped_summaries(records, ("answer_type",)),
        "by_answer_mode": grouped_summaries(records, ("answer_mode",)),
        "by_question": grouped_summaries(
            records,
            (
                "game_name",
                "cognitive_category",
                "question_id",
                "qa_type",
                "question_description",
            ),
        ),
    }
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_rules_version": EVALUATION_RULES_VERSION,
        "generated_at": summary["generated_at"],
        "git_revision": _git_revision(),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "answer_rules_sha256": sha256_file(ROOT / "answers.py"),
        "answer_metadata_sha256": sha256_file(ROOT / "metadata.py"),
        "inputs": input_metadata,
        "run_manifests": run_manifests,
        "expectations": {
            "samples": expected_samples,
            "games": expected_games,
            "samples_per_game": expected_samples_per_game,
        },
    }

    write_json(output_dir / "evaluation_manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "overall.json", overall)
    write_json(output_dir / "by_cognitive_category.json", by_category)
    write_json(output_dir / "by_game.json", by_game)
    write_json(output_dir / "by_qa_type.json", summary["by_qa_type"])
    write_json(output_dir / "by_answer_type.json", summary["by_answer_type"])
    write_json(output_dir / "by_answer_mode.json", summary["by_answer_mode"])
    write_json(output_dir / "by_question.json", summary["by_question"])
    write_json(output_dir / "diagnostics.json", make_diagnostics(records))
    write_jsonl(output_dir / "samples.jsonl", records)
    write_jsonl(
        output_dir / "errors.jsonl", (row for row in records if not row["correct"])
    )
    write_jsonl(
        output_dir / "relaxed_uplifts.jsonl",
        (row for row in records if row["option_content_match"]),
    )
    write_jsonl(
        output_dir / "ambiguous_relaxed_uplifts.jsonl",
        (row for row in records if row["option_content_ambiguous"]),
    )
    write_jsonl(
        output_dir / "format_violations.jsonl",
        (row for row in records if not row["final_line_json"]),
    )
    write_csv(output_dir / "by_cognitive_category.csv", by_category)
    write_csv(output_dir / "by_game.csv", by_game)
    write_csv(output_dir / "by_qa_type.csv", summary["by_qa_type"])
    write_csv(output_dir / "by_answer_type.csv", summary["by_answer_type"])
    write_csv(output_dir / "by_answer_mode.csv", summary["by_answer_mode"])
    write_csv(output_dir / "by_question.csv", summary["by_question"])
    _write_text(output_dir / "report.md", make_report(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="JSONL files, games directories, run directories, or a parent containing runs",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect-samples", type=int)
    parser.add_argument("--expect-games", type=int)
    parser.add_argument("--expect-samples-per-game", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_outputs(
        args.input,
        args.output_dir,
        expected_samples=args.expect_samples,
        expected_games=args.expect_games,
        expected_samples_per_game=args.expect_samples_per_game,
        overwrite=args.overwrite,
    )
    overall = summary["overall"]
    print(
        f"Evaluated {overall['samples']} samples across {summary['game_count']} games"
    )
    print(
        f"Accuracy: {overall['accuracy'] * 100:.2f}% "
        f"({overall['correct']}/{overall['samples']}); "
        f"macro game accuracy: {overall['macro_game_accuracy'] * 100:.2f}%"
    )
    print(
        f"Content-relaxed: {overall['relaxed_accuracy'] * 100:.2f}% "
        f"({overall['relaxed_correct']}/{overall['samples']}); "
        f"conservative: {overall['conservative_relaxed_accuracy'] * 100:.2f}% "
        f"({overall['conservative_relaxed_correct']}/{overall['samples']})"
    )
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
