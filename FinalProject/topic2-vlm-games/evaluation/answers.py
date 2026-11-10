"""Shared answer parsing and evaluation for game-reasoning outputs."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Sequence

from metadata import (
    ANSWER_MODE_OVERRIDES,
    LABEL_SCHEME_OVERRIDES,
    TASK_OPTION_OVERRIDES,
)


EVALUATION_RULES_VERSION = "2.1.0"
ANSWER_MODES = frozenset({"number_label", "letter_label", "option_value", "free_form"})
RELAXED_ANSWER_LINE_RE = re.compile(
    r"""^(?:[-*>]\s*)?(?:[*_]{0,2})["']?answer["']?(?:[*_]{0,2})"""
    r"""\s*[:=]\s*(.+?)\s*,?\s*$""",
    re.IGNORECASE,
)
LEADING_LABEL_RE = re.compile(
    r"^\s*(?:\[\s*([A-Za-z]|\d+)\s*\](?:\s+|$)|"
    r"([A-Za-z]|\d+)\s*[.:)](?:\s+|$))"
)
# Require digits after a decimal point. This accepts ``.767`` and ``0.7670``
# but does not reinterpret sentence punctuation such as ``Answer: -1.``.
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$")
# Only the tail of a response is eligible for answer extraction.  This keeps
# examples and speculative JSON from the reasoning section out of evaluation,
# while leaving enough room for a normal final explanation and code fence.
ANSWER_TAIL_CHARS = 4096
PYRAMID_POSITION_RE = re.compile(
    r"^\s*[\[(]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\])]\s+at\s+level\s+(-?\d+)\s*$",
    re.IGNORECASE,
)
CARD_RANKS = {
    "ace",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "jack",
    "queen",
    "king",
}
CARD_SUITS = {
    "club": "club",
    "clubs": "club",
    "diamond": "diamond",
    "diamonds": "diamond",
    "heart": "heart",
    "hearts": "heart",
    "spade": "spade",
    "spades": "spade",
}


@dataclass(frozen=True)
class AnswerSpec:
    mode: str
    source: str
    label_scheme: str | None
    options: tuple[str, ...]


def task_key(sample: dict[str, Any]) -> tuple[str, str]:
    return str(sample.get("game_name") or "").casefold(), str(sample.get("question_id"))


def _row_reference(sample: dict[str, Any]) -> Any:
    if "reference_answer" in sample:
        return sample.get("reference_answer")
    return sample.get("answer")


def effective_options(sample: dict[str, Any]) -> tuple[str, ...]:
    value = sample.get("options")
    if isinstance(value, list) and value:
        return tuple(str(option) for option in value)
    return TASK_OPTION_OVERRIDES.get(task_key(sample), ())


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    bracketed = re.fullmatch(r"\[\s*([A-Za-z]|\d+)\s*\]", text)
    if bracketed:
        text = bracketed.group(1)
    prefixed = re.fullmatch(r"(?:option|choice)\s*([A-Za-z]|\d+)", text, re.IGNORECASE)
    if prefixed:
        text = prefixed.group(1)
    return text.upper() if len(text) == 1 and text.isalpha() else text


def option_index(
    label: Any,
    options: Sequence[str],
    *,
    label_scheme: str = "one_based_numeric",
) -> int | None:
    normalized = normalize_label(label)
    if normalized is None:
        return None
    if normalized.isdigit():
        offset = 0 if label_scheme == "zero_based_numeric" else 1
        index = int(normalized) - offset
        if 0 <= index < len(options):
            return index
    if len(normalized) == 1 and "A" <= normalized <= "Z":
        index = ord(normalized) - ord("A")
        if 0 <= index < len(options):
            return index
    folded = normalized.casefold()
    for index, option in enumerate(options):
        if option.strip().casefold() == folded:
            return index
    return None


def _label_index(
    value: Any,
    option_count: int,
    label_scheme: str = "one_based_numeric",
) -> int | None:
    """Interpret a bare numeric or alphabetic value only as a choice label."""
    normalized = normalize_label(value)
    if normalized is None:
        return None
    if normalized.isdigit():
        offset = 0 if label_scheme == "zero_based_numeric" else 1
        index = int(normalized) - offset
        return index if 0 <= index < option_count else None
    if len(normalized) == 1 and "A" <= normalized <= "Z":
        index = ord(normalized) - ord("A")
        return index if 0 <= index < option_count else None
    return None


def _normalized_text(value: Any) -> str:
    return " ".join(str(value).strip().split()).casefold()


def _option_content(value: Any) -> str:
    text = str(value).strip()
    leading = LEADING_LABEL_RE.match(text)
    if leading:
        text = text[leading.end() :].strip()
    return _normalized_text(text)


def _unique_option_value_index(value: Any, options: Sequence[str]) -> int | None:
    if value is None:
        return None
    normalized = _normalized_text(value)
    matches = [
        index
        for index, option in enumerate(options)
        if _normalized_text(option) == normalized
    ]
    return matches[0] if len(matches) == 1 else None


def answer_spec(sample: dict[str, Any]) -> AnswerSpec:
    """Return the benchmark answer contract without parsing question layout."""
    key = task_key(sample)
    options = effective_options(sample)
    override = ANSWER_MODE_OVERRIDES.get(key)
    if override is not None:
        mode = str(override["mode"])
        source = str(override["source"])
    elif options:
        reference = answer_key(_row_reference(sample))
        if reference is not None and re.fullmatch(r"\d+", reference):
            mode = "number_label"
        elif reference is not None and re.fullmatch(r"[A-Z]", reference):
            mode = "letter_label"
        else:
            raise ValueError(
                f"Cannot infer labeled-choice answer mode for {key}: "
                f"reference={_row_reference(sample)!r}"
            )
        source = "official_answer_with_options"
    else:
        mode = "free_form"
        source = "no_options"

    if mode not in ANSWER_MODES:
        raise ValueError(f"Unsupported answer mode {mode!r} for {key}")
    if mode == "number_label":
        label_scheme = LABEL_SCHEME_OVERRIDES.get(key, "one_based_numeric")
    elif mode == "letter_label":
        label_scheme = "alphabetic"
    else:
        label_scheme = None

    reference = answer_key(_row_reference(sample))
    if mode == "number_label" and not (reference and re.fullmatch(r"\d+", reference)):
        raise ValueError(
            f"Expected a numeric reference label for {key}, got {reference!r}"
        )
    if mode == "letter_label" and not (reference and re.fullmatch(r"[A-Z]", reference)):
        raise ValueError(
            f"Expected a letter reference label for {key}, got {reference!r}"
        )
    if mode in {"number_label", "letter_label"}:
        reference_index = option_index(
            reference,
            options,
            label_scheme=label_scheme or "one_based_numeric",
        )
        if reference_index is None:
            raise ValueError(
                f"Reference label does not select an option for {key}: "
                f"reference={reference!r}, scheme={label_scheme!r}, options={len(options)}"
            )
    if mode == "option_value":
        if not options:
            raise ValueError(f"Option-value task {key} has no options")
        reference_index = _unique_option_value_index(_row_reference(sample), options)
        if reference_index is None:
            raise ValueError(
                f"Option-value reference must uniquely match one option for {key}: "
                f"reference={_row_reference(sample)!r}"
            )
    return AnswerSpec(mode, source, label_scheme, options)


def option_content_match(
    model_answer: Any,
    reference_answer: Any,
    options: Sequence[str],
    *,
    label_scheme: str = "one_based_numeric",
) -> tuple[bool, bool]:
    """Match a bare answer to the uniquely corresponding option content.

    The second result identifies answers that are also a valid label for a
    different option. Those cases are useful for a permissive counterfactual,
    but are too ambiguous for the conservative relaxed score.
    """
    if model_answer is None or not options:
        return False, False
    reference_index = option_index(
        answer_key(reference_answer), options, label_scheme=label_scheme
    )
    if reference_index is None:
        return False, False
    model_content = _normalized_text(model_answer)
    if not model_content:
        return False, False
    matching_indexes = [
        index
        for index, option in enumerate(options)
        if _option_content(option) == model_content
    ]
    if matching_indexes != [reference_index]:
        return False, False
    label_index = _label_index(answer_key(model_answer), len(options), label_scheme)
    return True, label_index is not None and label_index != reference_index


def answer_key(value: Any) -> str | None:
    """Normalize answer notation without consulting the option list."""
    normalized = normalize_label(value)
    if normalized is None:
        return None
    leading = LEADING_LABEL_RE.match(normalized)
    if leading:
        return normalize_label(leading.group(1) or leading.group(2))
    return normalized


def is_multiple_choice(sample: dict[str, Any]) -> bool:
    """Return whether a sample has a finite label or value choice contract."""
    return answer_spec(sample).mode != "free_form"


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or isinstance(value, (list, tuple, dict)):
        return None
    text = str(value).strip()
    if not NUMBER_RE.fullmatch(text):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _integer_structure(value: Any) -> tuple[Any, ...] | None:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError):
            return None

    def canonical(item: Any) -> int | tuple[Any, ...] | None:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            return item
        if isinstance(item, (list, tuple)):
            converted = tuple(canonical(child) for child in item)
            if any(child is None for child in converted):
                return None
            return converted
        return None

    result = canonical(parsed)
    if not isinstance(result, tuple) or not result:
        return None
    return result


def _pyramid_position(value: Any) -> tuple[int, int, int] | None:
    match = PYRAMID_POSITION_RE.fullmatch(str(value))
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _spider_card(value: Any) -> tuple[str, str] | None:
    text = " ".join(str(value).strip().casefold().split())
    words = text.split()
    if len(words) == 2:
        suit, rank = words
    elif len(words) == 3 and words[1] == "of":
        rank, _, suit = words
    else:
        return None
    normalized_suit = CARD_SUITS.get(suit)
    if normalized_suit is None or rank not in CARD_RANKS:
        return None
    return normalized_suit, rank


def compare_answers(
    model_answer: Any,
    reference_answer: Any,
    *,
    game_name: str | None = None,
    question_id: Any = None,
    answer_mode: str | None = None,
    multiple_choice: bool | None = None,
) -> tuple[bool, str | None]:
    """Compare answers and report the exact normalization rule that matched."""
    if answer_mode is None:
        answer_mode = "number_label" if multiple_choice is not False else "free_form"
    if answer_mode not in ANSWER_MODES:
        raise ValueError(f"Unsupported answer mode: {answer_mode!r}")

    # Choice labels are identifiers, not numeric or structured values. Keep their
    # comparison deliberately stricter than free-form answers.
    if answer_mode in {"number_label", "letter_label"}:
        model_key = answer_key(model_answer)
        reference_key = answer_key(reference_answer)
        if (
            model_key is not None
            and reference_key is not None
            and model_key == reference_key
        ):
            return True, "exact"
        return False, None

    if answer_mode == "option_value":
        if (
            model_answer is not None
            and reference_answer is not None
            and _normalized_text(model_answer) == _normalized_text(reference_answer)
        ):
            return True, "option_value"
        return False, None

    model_key = answer_key(model_answer)
    reference_key = answer_key(reference_answer)
    if (
        model_key is not None
        and reference_key is not None
        and model_key == reference_key
    ):
        return True, "exact"

    model_number = _number(model_answer)
    reference_number = _number(reference_answer)
    if (
        model_number is not None
        and reference_number is not None
        and model_number == reference_number
    ):
        return True, "numeric"

    model_structure = _integer_structure(model_answer)
    reference_structure = _integer_structure(reference_answer)
    if model_structure is not None and reference_structure is not None:
        if model_structure == reference_structure:
            return True, "integer_structure"
        reference_is_coordinate = all(
            isinstance(part, int) for part in reference_structure
        )
        if (
            reference_is_coordinate
            and len(model_structure) == 1
            and model_structure[0] == reference_structure
        ):
            return True, "singleton_coordinate"

    normalized_game = str(game_name or "").casefold()
    normalized_qid = str(question_id)
    if normalized_game == "pyramidchess" and normalized_qid == "3":
        model_position = _pyramid_position(model_answer)
        reference_position = _pyramid_position(reference_answer)
        if model_position is not None and model_position == reference_position:
            return True, "pyramid_position"

    if normalized_game == "spider_solitaire" and normalized_qid == "1":
        model_card = _spider_card(model_answer)
        reference_card = _spider_card(reference_answer)
        if model_card is not None and model_card == reference_card:
            return True, "spider_card"

    return False, None


def answers_match(
    model_answer: Any,
    reference_answer: Any,
    *,
    game_name: str | None = None,
    question_id: Any = None,
    answer_mode: str | None = None,
    multiple_choice: bool | None = None,
) -> bool:
    return compare_answers(
        model_answer,
        reference_answer,
        game_name=game_name,
        question_id=question_id,
        answer_mode=answer_mode,
        multiple_choice=multiple_choice,
    )[0]


def _relaxed_answer(response: str) -> str | None:
    tick = "`"
    for original in reversed(response.splitlines()):
        line = original.strip()
        if not line or line.startswith(tick * 3):
            continue
        while len(line) >= 2 and line.startswith(tick) and line.endswith(tick):
            line = line[1:-1].strip()
        line = line.replace(f"{tick}answer{tick}", "answer")
        match = RELAXED_ANSWER_LINE_RE.fullmatch(line)
        if not match:
            continue
        value = match.group(1).strip().strip("*_").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'`":
            value = value[1:-1].strip()
        if value and len(value) <= 200:
            return value
    return None


def parse_answer(response: str) -> dict[str, Any]:
    """Extract an answer from the response tail.

    Models often emit JSON examples or tentative answers while reasoning.  A
    full-response scan can mistake those for the final answer, so extraction is
    intentionally limited to the last few thousand characters.  The final
    non-empty line remains the strongest signal; otherwise we choose the
    complete answer JSON whose end position is latest in the tail.
    """
    tail = response[-ANSWER_TAIL_CHARS:]
    lines = [line.strip() for line in tail.splitlines() if line.strip()]

    if lines:
        try:
            value = json.loads(lines[-1])
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict) and "answer" in value:
            return {
                "value": value.get("answer"),
                "form": "standalone_json",
                "json_compliant": True,
            }

    decoder = json.JSONDecoder()
    matches: list[tuple[int, int, dict[str, Any]]] = []
    for match in re.finditer(r"\{", tail):
        try:
            value, consumed = decoder.raw_decode(tail[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "answer" in value:
            end = match.start() + consumed
            matches.append((end, match.start(), value))
    if matches:
        _, _, value = max(matches)
        return {
            "value": value.get("answer"),
            "form": "embedded_json",
            "json_compliant": True,
        }

    relaxed = _relaxed_answer(tail)
    if relaxed is not None:
        return {
            "value": relaxed,
            "form": "relaxed_answer_field",
            "json_compliant": False,
        }
    return {"value": None, "form": "missing", "json_compliant": False}


def final_line_answer_json(response: str) -> bool:
    """Return whether the final non-empty line is a complete answer JSON object."""
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return False
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and "answer" in value


def answer_details(row: dict[str, Any]) -> dict[str, Any]:
    spec = answer_spec(row)
    options = list(spec.options)
    reference = _row_reference(row)
    parsed = parse_answer(str(row.get("response") or ""))
    model_answer = parsed["value"]
    if spec.mode in {"number_label", "letter_label"}:
        reference_index = option_index(
            answer_key(reference),
            options,
            label_scheme=spec.label_scheme or "one_based_numeric",
        )
        model_index = option_index(
            answer_key(model_answer),
            options,
            label_scheme=spec.label_scheme or "one_based_numeric",
        )
    elif spec.mode == "option_value":
        reference_index = _unique_option_value_index(reference, options)
        model_index = _unique_option_value_index(model_answer, options)
    else:
        reference_index = model_index = None
    # These indexes explain the selections in the UI; they never determine correctness.
    reference_option = options[reference_index] if reference_index is not None else None
    model_option = options[model_index] if model_index is not None else None
    correct, match_rule = compare_answers(
        model_answer,
        reference,
        game_name=row.get("game_name"),
        question_id=row.get("question_id"),
        answer_mode=spec.mode,
    )
    content_match = content_ambiguous = False
    if spec.mode in {"number_label", "letter_label"} and not correct:
        content_match, content_ambiguous = option_content_match(
            model_answer,
            reference,
            options,
            label_scheme=spec.label_scheme or "one_based_numeric",
        )

    return {
        "model_answer": None if model_answer is None else str(model_answer),
        "answer_mode": spec.mode,
        "answer_mode_source": spec.source,
        "label_scheme": spec.label_scheme,
        "answer_form": parsed["form"],
        "json_compliant": parsed["json_compliant"],
        "final_line_json": final_line_answer_json(str(row.get("response") or "")),
        "reference_answer": None if reference is None else str(reference),
        "reference_option": reference_option,
        "reference_option_index": reference_index,
        "model_option": model_option,
        "model_option_index": model_index,
        "correct": correct,
        "match_rule": match_rule,
        "option_content_match": content_match,
        "option_content_ambiguous": content_ambiguous,
        "relaxed_correct": correct or content_match,
        "conservative_relaxed_correct": correct
        or (content_match and not content_ambiguous),
    }
