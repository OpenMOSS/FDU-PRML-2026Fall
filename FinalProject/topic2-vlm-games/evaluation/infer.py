#!/usr/bin/env python3
"""Offline batched multimodal inference for the course game-reasoning tests.

The inference path intentionally goes through ms-swift's VllmEngine instead of
using vLLM directly.  Data inspection, prompt preview, and environment checks do
not initialize the model, so they are safe to run on a CPU-only login node.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
import random
import re
import signal
import struct
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from answers import answer_spec
from identity import sample_uid


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT.parent / "test_data"
DEFAULT_MODEL = Path("Qwen2.5-VL-3B-Instruct")
DATA_FILENAME = "games_data_test.json"
IMAGE_ARCHIVE = "games_images_test.zip"
LOGGER = logging.getLogger("infer")
PROMPT_TEMPLATE_VERSION = 3

QWEN_NUMBER_LABEL_SUFFIX = (
    "Please reason step by step. At the end, provide your final answer as exactly "
    "one valid JSON object on its own line. In the `answer` field, write only the "
    "numeric option label exactly as shown in the question, for example "
    '`{"answer": "3"}`. Do not write the option text or reasoning in the `answer` '
    "field. Do not omit the opening or closing brace."
)
QWEN_LETTER_LABEL_SUFFIX = (
    "Please reason step by step. At the end, provide your final answer as exactly "
    "one valid JSON object on its own line. In the `answer` field, write only the "
    "letter option label exactly as shown in the question, for example "
    '`{"answer": "C"}`. Do not write the option text or reasoning in the `answer` '
    "field. Do not omit the opening or closing brace."
)
QWEN_OPTION_VALUE_SUFFIX = (
    "Please reason step by step. The choices in this question do not have separate "
    "option labels. At the end, provide your final answer as exactly one valid JSON "
    "object on its own line. In the `answer` field, write only the exact selected "
    "value as listed in the question. Do not invent an option label or index, and do "
    "not include reasoning in the `answer` field. Do not omit the opening or closing "
    "brace."
)
QWEN_FREEFORM_SUFFIX = (
    "Please reason step by step. At the end, provide your final answer as a valid "
    'JSON object on its own line, for example: {"answer": "your final '
    'answer"}. Keep the `answer` field concise and do not include reasoning in it. '
    "The final line must contain exactly one JSON object; do not omit its braces."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return value or "model"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class RunInferenceLock:
    """Prevent two independent inference commands from writing one run."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / ".inference.lock"
        self.handle: Any | None = None

    def __enter__(self) -> "RunInferenceLock":
        try:
            import fcntl
        except ImportError as error:  # pragma: no cover - the target is Linux
            raise RuntimeError("Run locking requires the fcntl module") from error
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                f"Another inference command is already using run directory: "
                f"{self.path.parent}"
            ) from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.handle is None:
            return
        import fcntl

        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


def iter_chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_samples(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / DATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"game-reasoning annotation file not found: {path}")
    samples = read_json(path)
    if not isinstance(samples, list):
        raise ValueError(
            f"Expected a JSON list in {path}, got {type(samples).__name__}"
        )
    required = {"data_id", "question_id", "question", "image", "game_name"}
    for index, sample in enumerate(samples):
        missing = required.difference(sample)
        if missing:
            raise ValueError(f"Sample {index} is missing fields: {sorted(missing)}")
    return samples


def game_names(samples: Sequence[dict[str, Any]]) -> list[str]:
    # Stable dataset order is useful for reproducible all-game runs.
    return list(dict.fromkeys(sample["game_name"] for sample in samples))


def parse_games(requested: Sequence[str], available: Sequence[str]) -> list[str]:
    flattened: list[str] = []
    for token in requested:
        flattened.extend(part for part in token.split(",") if part)
    if not flattened or flattened == ["all"]:
        return list(available)
    if "all" in flattened:
        raise ValueError("Use either --games all or explicit game names, not both")
    unknown = sorted(set(flattened).difference(available))
    if unknown:
        raise ValueError(
            f"Unknown game(s): {unknown}. Use `inspect` to list valid names."
        )
    return list(dict.fromkeys(flattened))


def deterministic_sample(
    samples: Sequence[dict[str, Any]], count: int | None, seed: int, namespace: str
) -> list[dict[str, Any]]:
    """Sample a stable subset while preserving dataset order for readable outputs."""
    if count is None or len(samples) <= count:
        return list(samples)
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    selected_indices = set(rng.sample(range(len(samples)), count))
    return [sample for index, sample in enumerate(samples) if index in selected_indices]


def resolve_image_path(data_dir: Path, sample: dict[str, Any]) -> Path:
    image_path = Path(sample["image"])
    if not image_path.is_absolute():
        image_path = data_dir / image_path
    return image_path.resolve()


def extract_images(data_dir: Path) -> dict[str, int]:
    """Extract missing images while rejecting archive path traversal."""
    archive_path = data_dir / IMAGE_ARCHIVE
    if not archive_path.is_file():
        raise FileNotFoundError(f"Image archive not found: {archive_path}")
    root = data_dir.resolve()
    extracted = skipped = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in {archive_path}: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == member.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            extracted += 1
    return {"extracted": extracted, "already_present": skipped}


def validate_images(
    data_dir: Path, samples: Sequence[dict[str, Any]], games: Sequence[str]
) -> None:
    selected = set(games)
    missing = [
        resolve_image_path(data_dir, sample)
        for sample in samples
        if sample["game_name"] in selected
        and not resolve_image_path(data_dir, sample).is_file()
    ]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} selected image(s) are missing. Run `prepare` or use "
            f"--auto-extract. First missing paths:\n{preview}"
        )


def build_prompt(sample: dict[str, Any], style: str, extra_suffix: str = "") -> str:
    question = sample["question"].strip()
    suffixes: list[str] = []
    if style == "qwen-json":
        mode = answer_spec(sample).mode
        suffixes.append(
            {
                "number_label": QWEN_NUMBER_LABEL_SUFFIX,
                "letter_label": QWEN_LETTER_LABEL_SUFFIX,
                "option_value": QWEN_OPTION_VALUE_SUFFIX,
                "free_form": QWEN_FREEFORM_SUFFIX,
            }[mode]
        )
    elif style != "original":
        raise ValueError(f"Unknown prompt style: {style}")
    if extra_suffix.strip():
        suffixes.append(extra_suffix.strip())
    text = question
    if suffixes:
        text += "\n\n" + "\n\n".join(suffixes)
    # The marker and images list are paired by ms-swift's InferRequest protocol.
    return "<image>\n" + text


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def is_course_model_config(config: dict[str, Any]) -> bool:
    """Check the 3B backbone, including nested configs saved by newer tools.

    This checks structure, not the provenance of weights.
    """
    text = config.get("text_config") or config
    expected = {"hidden_size": 2048, "intermediate_size": 11008,
                "num_hidden_layers": 36, "num_attention_heads": 16,
                "num_key_value_heads": 2}
    return ((config.get("architectures") or [None])[0]
            == "Qwen2_5_VLForConditionalGeneration"
            and all(text.get(key) == value for key, value in expected.items()))


def explicit_model_type(model_path: Path) -> str | None:
    """Identify the permitted 3B model even under a local submission name."""
    config_path = model_path / "config.json"
    if config_path.is_file() and is_course_model_config(read_json(config_path)):
        return "qwen2_5_vl"
    return None


def environment_report(model_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {
            name: package_version(name)
            for name in (
                "ms-swift",
                "vllm",
                "transformers",
                "torch",
                "qwen-vl-utils",
                "decord",
            )
        },
        "model_path": str(model_path.resolve()),
        "checks": {},
    }
    config_path = model_path / "config.json"
    if config_path.is_file():
        config = read_json(config_path)
        architecture = (config.get("architectures") or [None])[0]
        report["model_config"] = {
            "model_type": config.get("model_type"),
            "architecture": architecture,
            "language_model_only": config.get("language_model_only"),
        }
    else:
        architecture = None
        report["model_config"] = None

    report["checks"]["permitted_qwen25vl_3b"] = (
        config_path.is_file() and is_course_model_config(read_json(config_path))
    )
    vllm_version = report["packages"]["vllm"]
    swift_version = report["packages"]["ms-swift"]
    transformers_version = report["packages"]["transformers"]
    report["checks"]["qwen25vl_architecture"] = (
        architecture == "Qwen2_5_VLForConditionalGeneration"
    )
    report["checks"]["vllm_recommended_version"] = bool(
        vllm_version and _version_tuple(vllm_version) >= (0, 19, 0)
    )
    report["checks"]["swift_installed"] = swift_version is not None
    report["checks"]["transformers_installed"] = transformers_version is not None
    # These two registry checks are intentionally best-effort: they import only
    # framework registries and never initialize model weights.
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        report["checks"]["vllm_architecture_registered"] = (
            architecture in ModelRegistry.models
        )
    except Exception as error:
        report["checks"]["vllm_architecture_registered"] = False
        report["vllm_registry_error"] = repr(error)
    try:
        from swift.model.model_meta import get_matched_model_types

        configured_type = (
            read_json(config_path).get("model_type") if config_path.is_file() else None
        )
        registered_types = get_matched_model_types([architecture])
        resolved_type = explicit_model_type(model_path)
        report["checks"]["swift_model_registered"] = (
            configured_type in registered_types or resolved_type in registered_types
        )
        report["swift_model_type"] = resolved_type or configured_type
        report["swift_model_type_candidates"] = registered_types
    except Exception as error:
        report["checks"]["swift_model_registered"] = False
        report["swift_registry_error"] = repr(error)
    report["supported_by_installed_stack"] = all(report["checks"].values())

    try:
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
    except Exception as error:  # doctor should report import failures rather than crash
        report["cuda"] = {"available": False, "device_count": 0, "error": repr(error)}
    return report


def _version_tuple(version: str) -> tuple[int, ...]:
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    return tuple(map(int, match.groups())) if match else ()


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def completed_uids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if record.get("status") == "ok":
                completed.add(record["uid"])
    return completed


def result_record(
    sample: dict[str, Any], prompt: str, image_path: Path, response: Any
) -> dict[str, Any]:
    choice = response.choices[0]
    usage = jsonable(response.usage)
    spec = answer_spec(sample)
    return {
        "uid": sample_uid(sample),
        "status": "ok",
        "game_name": sample["game_name"],
        "data_id": sample["data_id"],
        "question_id": sample["question_id"],
        "qa_type": sample.get("qa_type"),
        "qa_level": sample.get("qa_level"),
        "plot_level": sample.get("plot_level"),
        "question_description": sample.get("question_description"),
        "game_type": sample.get("game_type"),
        "question": sample["question"],
        "options": sample.get("options"),
        "reference_answer": sample.get("answer"),
        "answer_mode": spec.mode,
        "answer_mode_source": spec.source,
        "label_scheme": spec.label_scheme,
        "image": sample["image"],
        "state": sample.get("state"),
        "resolved_image": str(image_path),
        "prompt": prompt,
        "response": choice.message.content,
        "reasoning_content": getattr(choice.message, "reasoning_content", None),
        "finish_reason": choice.finish_reason,
        "usage": usage,
        "created_at": utc_now(),
    }


def append_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def adapter_metadata(adapter: Path, model_path: Path) -> dict[str, Any]:
    """Validate a local PEFT LoRA and fingerprint its actual inference weights.

    Reading the safetensors header also rejects checkpoints still being written.
    No torch import or model loading is needed here.
    """
    adapter = adapter.expanduser().resolve()
    config_path = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    config = read_json(config_path)
    if config.get("peft_type") != "LORA":
        raise ValueError(f"Only PEFT LoRA adapters are supported: {adapter}")
    for key in ("modules_to_save", "use_dora", "lora_bias", "target_parameters"):
        if config.get(key):
            raise ValueError(f"vLLM adapter inference does not support {key}: {adapter}")
    if config.get("bias", "none") != "none":
        raise ValueError("vLLM adapter inference requires bias=none")
    base = config.get("base_model_name_or_path")
    if base and Path(base).is_absolute() and Path(base).resolve() != model_path.resolve():
        raise ValueError(f"Adapter base model is {base}, but --model is {model_path}")
    rank = max([config.get("r", 0), *(config.get("rank_pattern") or {}).values()])
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"Invalid LoRA rank in {config_path}")
    supported_ranks = (8, 16, 32, 64, 128, 256, 320, 512)
    capacity = next((r for r in supported_ranks if r >= rank), None)
    if capacity is None:
        raise ValueError(f"LoRA rank {rank} exceeds supported vLLM capacity")
    with weights.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Incomplete adapter weights: {weights}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > 100_000_000:
            raise ValueError(f"Invalid safetensors header: {weights}")
        header = json.loads(handle.read(header_size))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    if not tensors or weights.stat().st_size != 8 + header_size + max(
        t["data_offsets"][1] for t in tensors
    ):
        raise ValueError(f"Incomplete adapter weights: {weights}")
    hashes = {}
    for path in (config_path, weights):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.name] = digest.hexdigest()
    return {"path": str(adapter), "rank": rank, "max_lora_rank": capacity,
            "base_model_name_or_path": base, "sha256": hashes}


def full_checkpoint_metadata(model_path: Path) -> dict[str, Any]:
    """Reject incomplete HF saves and identify immutable full-training weights.

    Check every shard header/length and index entry without reading tens of GB
    on each scheduler poll. File sizes and nanosecond mtimes detect replacement;
    configuration and headers are hashed. This is not a full content checksum.
    """
    model_path = model_path.resolve()
    config = read_json(model_path / "config.json")
    state = read_json(model_path / "trainer_state.json")
    if not config.get("model_type") or int(state["global_step"]) <= 0:
        raise ValueError(f"Invalid full checkpoint: {model_path}")
    index_path = model_path / "model.safetensors.index.json"
    weight_map = read_json(index_path)["weight_map"] if index_path.exists() else None
    names = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
    shards, found = {}, {}
    for name in names:
        if Path(name).name != name:
            raise ValueError(f"Invalid checkpoint shard name: {name}")
        path = model_path / name
        before = path.stat()
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Incomplete model shard: {path}")
            size = struct.unpack("<Q", prefix)[0]
            if size > 100_000_000:
                raise ValueError(f"Invalid model shard header: {path}")
            raw = handle.read(size)
            header = json.loads(raw)
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if not tensors or before.st_size != 8 + size + max(
            t["data_offsets"][1] for t in tensors.values()
        ):
            raise ValueError(f"Incomplete model shard: {path}")
        for key in tensors:
            if key in found:
                raise ValueError(f"Duplicate model tensor: {key}")
            found[key] = name
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Model shard is changing: {path}")
        shards[name] = {"size": after.st_size, "mtime_ns": after.st_mtime_ns,
                        "header_sha256": hashlib.sha256(raw).hexdigest()}
    if weight_map is not None and found != weight_map:
        raise ValueError(f"Model shards differ from index: {model_path}")
    return {"path": str(model_path), "global_step": state["global_step"],
            "config_sha256": fingerprint(config), "shards": shards}


def _inference_config_fingerprint(config: dict[str, Any]) -> str:
    """Fingerprint model/output semantics, excluding worker topology."""
    comparable = {
        key: value for key, value in config.items() if key != "data_parallel_size"
    }
    return fingerprint(comparable)


def analyze_result_files(input_paths: Sequence[Path], output_dir: Path) -> None:
    try:
        scripts_dir = str(ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from analyze_outputs import analyze_outputs

        result = analyze_outputs(list(input_paths), output_dir)
        LOGGER.info("Analysis saved for %d row(s): %s", result["rows"], output_dir)
    except Exception:
        LOGGER.exception(
            "Analysis failed for %s", ", ".join(str(path) for path in input_paths)
        )


def _inference_config(
    args: argparse.Namespace,
    data_dir: Path,
    model_path: Path,
    suffix: str,
) -> dict[str, Any]:
    config = {
        "model": str(model_path),
        "data_file": str((data_dir / DATA_FILENAME).resolve()),
        "prompt_style": args.prompt_style,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_suffix": suffix,
        "enable_thinking": args.enable_thinking,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        # Keep the historical field for readers; partitioning is explicit in
        # manifest.parallelism. Worker count remains outside the fingerprint.
        "data_parallel_size": getattr(args, "sample_workers", None) or args.data_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "batch_size": args.batch_size,
        "sample_per_game": args.sample_per_game,
        "sample_seed": args.sample_seed,
        "analyze_after_game": args.analyze_after_game,
    }
    # Omit absent options to preserve compatibility with existing base-model runs.
    if getattr(args, "adapter", None):
        config["adapter"] = adapter_metadata(args.adapter, model_path)
    if getattr(args, "checkpoint_model", False):
        config["full_checkpoint"] = full_checkpoint_metadata(model_path)
    if getattr(args, "max_num_seqs", None) is not None:
        config["max_num_seqs"] = args.max_num_seqs
    return config


def _compatible_manifest_config(
    old_manifest: dict[str, Any], config: dict[str, Any]
) -> bool:
    old_config = old_manifest.get("config")
    if isinstance(old_config, dict):
        # Data parallelism is execution topology rather than model/output
        # semantics, so a run can resume with a different worker count.
        return _inference_config_fingerprint(old_config) == _inference_config_fingerprint(
            config
        )
    return old_manifest.get("config_fingerprint") in {
        _inference_config_fingerprint(config),
        fingerprint(config),
    }


def _ensure_manifest(
    manifest_path: Path,
    run_name: str,
    config: dict[str, Any],
    model_path: Path,
    selected_games: Sequence[str],
    selected_specs: Sequence[Any],
) -> None:
    if manifest_path.exists():
        old_manifest = read_json(manifest_path)
        if not _compatible_manifest_config(old_manifest, config):
            raise RuntimeError(
                f"Run directory already contains incompatible settings: {manifest_path.parent}. "
                "Choose another --run-name (recommended), or use the exact original settings."
            )
        return

    report = environment_report(model_path)
    if not report["supported_by_installed_stack"]:
        raise RuntimeError(
            "Installed stack did not pass compatibility checks. Run `doctor` for details."
        )
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "run_name": run_name,
            "config_fingerprint": _inference_config_fingerprint(config),
            "config": config,
            "environment": report,
            "games_requested": list(selected_games),
            "answer_mode_counts": dict(
                sorted(Counter(spec.mode for spec in selected_specs).items())
            ),
        },
    )


def _visible_gpu_ids() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        devices = [item.strip() for item in raw.split(",") if item.strip()]
        if not devices or devices == ["-1"]:
            raise RuntimeError("CUDA_VISIBLE_DEVICES does not expose any GPU")
        return devices
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception as error:
        raise RuntimeError(
            "Cannot determine visible GPUs; set CUDA_VISIBLE_DEVICES explicitly."
        ) from error
    if count <= 0:
        raise RuntimeError("No CUDA GPU is visible for data-parallel inference")
    return [str(index) for index in range(count)]


def _device_groups(args: argparse.Namespace) -> list[list[str]]:
    devices = _visible_gpu_ids()
    workers = getattr(args, "sample_workers", None) or args.data_parallel_size
    required = workers * args.tensor_parallel_size
    if len(devices) < required:
        raise RuntimeError(
            "Data parallel inference needs "
            f"{required} visible GPU(s) ({workers} independent "
            f"worker(s) x {args.tensor_parallel_size} tensor-parallel GPU(s)), "
            f"but only {len(devices)} are visible: {devices}"
        )
    return [
        devices[start : start + args.tensor_parallel_size]
        for start in range(0, required, args.tensor_parallel_size)
    ]


def _game_pending_count(
    samples: Sequence[dict[str, Any]], game: str, args: argparse.Namespace, run_dir: Path
) -> tuple[int, int, int]:
    game_samples = [sample for sample in samples if sample["game_name"] == game]
    original_count = len(game_samples)
    game_samples = deterministic_sample(
        game_samples, args.sample_per_game, args.sample_seed, game
    )
    if args.limit is not None:
        game_samples = game_samples[: args.limit]
    done = completed_uids(run_dir / "games" / f"{game}.jsonl")
    pending = sum(sample_uid(sample) not in done for sample in game_samples)
    return original_count, len(game_samples), pending


def partition_games(
    selected_games: Sequence[str],
    samples: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[list[list[str]], dict[str, dict[str, int]]]:
    """Assign whole games to workers using pending sample counts as weights."""
    if args.data_parallel_size > len(selected_games):
        raise ValueError(
            "--game-workers cannot exceed the number of selected games; "
            "use --sample-workers to split samples within a game"
        )
    details = {
        game: dict(
            zip(
                ("total", "selected", "pending"),
                _game_pending_count(samples, game, args, run_dir),
            )
        )
        for game in selected_games
    }
    assignments: list[list[str]] = [[] for _ in range(args.data_parallel_size)]
    loads = [0] * args.data_parallel_size
    order = {game: index for index, game in enumerate(selected_games)}
    for game in sorted(
        selected_games, key=lambda item: (-details[item]["pending"], order[item])
    ):
        worker = min(range(args.data_parallel_size), key=lambda index: (loads[index], index))
        assignments[worker].append(game)
        loads[worker] += details[game]["pending"]
    for games in assignments:
        games.sort(key=order.__getitem__)
    return assignments, details


def _update_parallel_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    assignments: Sequence[Sequence[str]],
    details: dict[str, dict[str, int]],
    device_groups: Sequence[Sequence[str]],
) -> None:
    manifest = read_json(manifest_path)
    manifest["parallelism"] = {
        "partition_by": "game",
        "worker_count": args.data_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "workers": [
            {
                "worker_id": worker_id,
                "cuda_visible_devices": list(devices),
                "games": list(games),
                "pending_samples": sum(details[game]["pending"] for game in games),
            }
            for worker_id, (games, devices) in enumerate(
                zip(assignments, device_groups)
            )
        ],
    }
    write_json_atomic(manifest_path, manifest)


def _analyze_run_outputs(
    run_dir: Path, selected_games: Sequence[str], analyze_after_game: bool
) -> None:
    if not analyze_after_game:
        return
    output_paths = [
        run_dir / "games" / f"{game}.jsonl"
        for game in selected_games
        if (run_dir / "games" / f"{game}.jsonl").is_file()
    ]
    for output_path in output_paths:
        analyze_result_files(
            [output_path], run_dir / "analysis" / "games" / output_path.stem
        )
    all_output_paths = sorted((run_dir / "games").glob("*.jsonl"))
    if all_output_paths:
        analyze_result_files(all_output_paths, run_dir / "analysis" / "all")


def _run_inference_worker(
    args: argparse.Namespace,
    samples: Sequence[dict[str, Any]],
    selected_games: Sequence[str],
    suffix: str,
    model_path: Path,
    run_dir: Path,
    worker_id: int,
    analyze_after_game: bool,
    result_dir: Path | None = None,
) -> None:
    # Importing swift/vLLM can initialize CUDA-related libraries.  This function
    # is called only after the parent has validated inputs and written the manifest.
    from swift.infer_engine import InferRequest, RequestConfig, VllmEngine

    prefix = f"[worker {worker_id}] "
    LOGGER.info("%sInitializing ms-swift VllmEngine for %s", prefix, model_path)
    adapter_kwargs = {}
    if getattr(args, "checkpoint_model", False):
        expected = read_json(run_dir / "manifest.json")["config"]["full_checkpoint"]
        if full_checkpoint_metadata(model_path) != expected:
            raise RuntimeError("Full checkpoint changed after the run manifest was created")
    if getattr(args, "adapter", None):
        metadata = adapter_metadata(args.adapter, model_path)
        expected = read_json(run_dir / "manifest.json")["config"]["adapter"]
        if metadata != expected:
            raise RuntimeError("Adapter changed after the run manifest was created")
        adapter_kwargs = {"adapters": [metadata["path"]],
                          "max_lora_rank": metadata["max_lora_rank"]}
        LOGGER.info("%sLoading unmerged LoRA: %s", prefix, metadata)
    if getattr(args, "max_num_seqs", None) is not None:
        adapter_kwargs["max_num_seqs"] = args.max_num_seqs
    engine_start = time.perf_counter()
    engine = VllmEngine(
        str(model_path),
        model_type=explicit_model_type(model_path),
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        seed=args.seed,
        **adapter_kwargs,
    )
    append_jsonl(run_dir / "timing.jsonl", [{"kind": "engine_load", "seconds": time.perf_counter()-engine_start, "worker_id": worker_id, "created_at": utc_now()}])
    request_config = RequestConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        stream=False,
    )

    effective = engine._prepare_generation_config(request_config)
    engine._add_stop_words(effective, request_config)
    if effective.temperature != 0 or effective.max_tokens != 4096 or effective.sampling_type.name != "GREEDY":
        raise RuntimeError("Effective vLLM decoding is not greedy-4096")
    write_json_atomic(run_dir / "effective-sampling.json", {
        "temperature": effective.temperature, "max_tokens": effective.max_tokens,
        "sampling_type": effective.sampling_type.name, "top_p": effective.top_p,
        "top_k": effective.top_k, "repetition_penalty": effective.repetition_penalty,
        "presence_penalty": effective.presence_penalty, "seed": effective.seed,
        "stop": effective.stop, "stop_token_ids": effective.stop_token_ids,
    })
    output_paths: list[Path] = []
    for game_index, game in enumerate(selected_games, 1):
        output_path = (result_dir if result_dir is not None else run_dir / "games") / f"{game}.jsonl"
        output_paths.append(output_path)
        game_samples = [sample for sample in samples if sample["game_name"] == game]
        original_game_count = len(game_samples)
        game_samples = deterministic_sample(
            game_samples, args.sample_per_game, args.sample_seed, game
        )
        if args.limit is not None:
            game_samples = game_samples[: args.limit]
        done = completed_uids(output_path)
        pending = [sample for sample in game_samples if sample_uid(sample) not in done]
        LOGGER.info(
            "%s[%d/%d] %s: total=%d selected=%d completed=%d pending=%d output=%s",
            prefix,
            game_index,
            len(selected_games),
            game,
            original_game_count,
            len(game_samples),
            len(done.intersection(sample_uid(sample) for sample in game_samples)),
            len(pending),
            output_path,
        )
        for batch_index, batch in enumerate(iter_chunks(pending, args.batch_size), 1):
            prompts = [
                build_prompt(sample, args.prompt_style, suffix) for sample in batch
            ]
            image_paths = [resolve_image_path(args.data_dir, sample) for sample in batch]
            requests = [
                InferRequest(
                    messages=[{"role": "user", "content": prompt}],
                    images=[str(image_path)],
                    chat_template_kwargs={"enable_thinking": args.enable_thinking},
                )
                for prompt, image_path in zip(prompts, image_paths)
            ]
            for attempt in range(args.max_retries + 1):
                try:
                    batch_start = time.perf_counter()
                    responses = engine.infer(requests, request_config, use_tqdm=True)
                    batch_seconds = time.perf_counter()-batch_start
                    if len(responses) != len(batch):
                        raise RuntimeError(
                            f"Engine returned {len(responses)} responses for {len(batch)} requests"
                        )
                    records = [
                        result_record(sample, prompt, image_path, response)
                        for sample, prompt, image_path, response in zip(
                            batch, prompts, image_paths, responses
                        )
                    ]
                    append_jsonl(output_path, records)
                    append_jsonl(run_dir / "timing.jsonl", [{"kind": "generation", "game": game, "batch": batch_index, "rows": len(records), "seconds": batch_seconds, "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in records), "worker_id": worker_id, "created_at": utc_now()}])
                    LOGGER.info(
                        "%s%s batch %d: saved %d result(s)",
                        prefix,
                        game,
                        batch_index,
                        len(records),
                    )
                    break
                except Exception:
                    if attempt >= args.max_retries:
                        raise
                    delay = args.retry_backoff * (2**attempt)
                    LOGGER.exception(
                        "%s%s batch %d failed (attempt %d/%d); retrying in %.1fs",
                        prefix,
                        game,
                        batch_index,
                        attempt + 1,
                        args.max_retries + 1,
                        delay,
                    )
                    time.sleep(delay)
        if analyze_after_game and output_path.is_file():
            analyze_result_files([output_path], run_dir / "analysis" / "games" / game)

    LOGGER.info("%sInference worker complete", prefix)
    if analyze_after_game:
        existing_outputs = [path for path in output_paths if path.is_file()]
        if existing_outputs:
            analyze_result_files(existing_outputs, run_dir / "analysis" / "all")


def _data_parallel_worker_entry(
    args: argparse.Namespace,
    selected_games: Sequence[str],
    suffix: str,
    model_path: Path,
    run_dir: Path,
    worker_id: int,
    devices: Sequence[str],
) -> None:
    os.setsid()
    # Spawn is required so no CUDA context from the parent is inherited.  Set
    # the worker-local visibility before importing ms-swift/vLLM.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    os.environ["COURSE_DP_WORKER"] = str(worker_id)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        worker_args = argparse.Namespace(**vars(args))
        worker_args.data_parallel_size = 1
        samples = load_samples(worker_args.data_dir)
        _run_inference_worker(
            worker_args,
            samples,
            selected_games,
            suffix,
            model_path,
            run_dir,
            worker_id,
            analyze_after_game=False,
        )
    except BaseException:
        LOGGER.exception("[worker %d] Inference failed", worker_id)
        raise


def _run_data_parallel(
    args: argparse.Namespace,
    selected_games: Sequence[str],
    suffix: str,
    model_path: Path,
    run_dir: Path,
    samples: Sequence[dict[str, Any]],
) -> None:
    assignments, details = partition_games(selected_games, samples, args, run_dir)
    device_groups = _device_groups(args)
    _update_parallel_manifest(
        run_dir / "manifest.json", args, assignments, details, device_groups
    )
    LOGGER.info(
        "Starting data-parallel inference with %d worker(s), tensor_parallel_size=%d",
        args.data_parallel_size,
        args.tensor_parallel_size,
    )
    for worker_id, (games, devices) in enumerate(zip(assignments, device_groups)):
        LOGGER.info(
            "Worker %d: GPUs=%s games=%s pending=%d",
            worker_id,
            ",".join(devices),
            ",".join(games),
            sum(details[game]["pending"] for game in games),
        )

    context = multiprocessing.get_context("spawn")
    worker_args = argparse.Namespace(
        **{key: value for key, value in vars(args).items() if key != "func"}
    )
    processes = [
        context.Process(
            target=_data_parallel_worker_entry,
            args=(worker_args, games, suffix, model_path, run_dir, worker_id, devices),
            name=f"course-infer-worker-{worker_id}",
        )
        for worker_id, (games, devices) in enumerate(zip(assignments, device_groups))
    ]
    _run_processes(processes)
    _analyze_run_outputs(run_dir, selected_games, args.analyze_after_game)


def _signal_worker(process: Any, sig: int) -> None:
    # Each worker is a session leader, so its vLLM/TP children are stopped too.
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        if process.is_alive():
            if sig == signal.SIGKILL:
                process.kill()
            else:
                process.terminate()


def _run_processes(processes: Sequence[Any]) -> None:
    def interrupted(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        while any(process.is_alive() for process in processes):
            failed = next(
                (process for process in processes if process.exitcode not in (None, 0)),
                None,
            )
            if failed is not None:
                raise RuntimeError(
                    f"Data-parallel worker {failed.name} exited with code {failed.exitcode}"
                )
            time.sleep(0.5)
        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Data-parallel worker {process.name} exited with code {process.exitcode}"
                )
    except BaseException:
        LOGGER.warning("Stopping inference workers and their TP processes")
        for process in started:
            _signal_worker(process, signal.SIGTERM)
        for process in started:
            process.join(timeout=10)
            _signal_worker(process, signal.SIGKILL)
            process.join()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def _sample_selection(
    args: argparse.Namespace, samples: Sequence[dict[str, Any]], games: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    selected = {}
    for game in games:
        rows = deterministic_sample(
            [s for s in samples if s["game_name"] == game],
            args.sample_per_game, args.sample_seed, game,
        )
        if args.limit is not None:
            rows = rows[:args.limit]
        uids = [sample_uid(s) for s in rows]
        if len(set(uids)) != len(uids):
            raise ValueError(f"Duplicate sample UIDs in {game}")
        selected[game] = rows
    return selected


def _ensure_sample_plan(
    run_dir: Path, selected: dict[str, list[dict[str, Any]]]
) -> None:
    # Freeze the selected inputs, not their worker assignment. Resumes can use
    # fewer/more replicas without repeating already completed samples.
    payload = json.dumps(selected, ensure_ascii=False, sort_keys=True).encode("utf-8")
    plan = {
        "schema_version": 1,
        "selection_sha256": hashlib.sha256(payload).hexdigest(),
        "games": {game: [sample_uid(s) for s in rows] for game, rows in selected.items()},
    }
    path = run_dir / "sample-plan.json"
    if path.exists():
        if read_json(path) != plan:
            raise ValueError("Sample selection or annotations changed; choose a new --run-name")
    else:
        write_json_atomic(path, plan)


def _read_shard_rows(path: Path, tolerate_partial_tail: bool) -> Iterable[dict[str, Any]]:
    # Interrupted writes may leave an incomplete UTF-8/JSON final line. Retain
    # the shard as evidence and recover all complete rows without appending to it.
    with path.open("rb") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError) as error:
                if tolerate_partial_tail and not line.endswith(b"\n"):
                    LOGGER.warning("Ignoring incomplete shard tail: %s:%d", path, number)
                    return
                raise ValueError(f"Invalid JSONL at {path}:{number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Invalid result at {path}:{number}")
            yield row


def _merge_sample_outputs(
    run_dir: Path,
    selected: dict[str, list[dict[str, Any]]],
    *,
    require_complete: bool = False,
) -> dict[str, set[str]]:
    completed = {}
    for game, samples in selected.items():
        expected = {sample_uid(s): s for s in samples}
        output = run_dir / "games" / f"{game}.jsonl"
        paths = ([output] if output.exists() else []) + sorted(
            (run_dir / "sample-shards").glob(f"attempt-*/worker-*/{game}.jsonl")
        )
        records: dict[str, dict[str, Any]] = {}
        shard_uids: set[str] = set()
        for path in paths:
            seen = set()
            for row in _read_shard_rows(path, tolerate_partial_tail=path != output):
                uid = row.get("uid")
                if not isinstance(uid, str) or uid not in expected:
                    raise ValueError(f"Unexpected sample UID {uid!r} in {path}")
                if row.get("status") != "ok":
                    continue
                sample = expected[uid]
                for result_key, sample_key in (
                    ("game_name", "game_name"), ("question", "question"),
                    ("data_id", "data_id"), ("question_id", "question_id"),
                    ("options", "options"), ("reference_answer", "answer"),
                    ("image", "image"),
                ):
                    if row.get(result_key) != sample.get(sample_key):
                        raise ValueError(f"Result input mismatch ({result_key}) for {uid} in {path}")
                if (uid in seen or (path != output and uid in shard_uids)
                        or (uid in records and records[uid] != row)):
                    raise ValueError(f"Duplicate or conflicting result for {uid} in {path}")
                seen.add(uid)
                if path != output:
                    shard_uids.add(uid)
                records[uid] = row
        if require_complete and set(records) != set(expected):
            raise ValueError(f"Incomplete {game}: {len(records)}/{len(expected)} successful samples")
        if records:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for uid in expected:
                    if uid in records:
                        handle.write(json.dumps(records[uid], ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(output)
        completed[game] = set(records)
    return completed


def _sample_parallel_worker_entry(
    args: argparse.Namespace, samples: list[dict[str, Any]], suffix: str,
    model_path: Path, run_dir: Path, worker_id: int, devices: Sequence[str],
    result_dir: Path,
) -> None:
    os.setsid()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    os.environ["COURSE_DP_WORKER"] = str(worker_id)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    worker_args = argparse.Namespace(**vars(args))
    # Selection and debug limits have already been applied before partitioning.
    worker_args.sample_per_game = None
    worker_args.limit = None
    _run_inference_worker(
        worker_args, samples, game_names(samples), suffix, model_path, run_dir,
        worker_id, False, result_dir=result_dir,
    )


def _run_sample_parallel(
    args: argparse.Namespace, selected_games: Sequence[str], suffix: str,
    model_path: Path, run_dir: Path, samples: Sequence[dict[str, Any]],
) -> None:
    selected = _sample_selection(args, samples, selected_games)
    _ensure_sample_plan(run_dir, selected)
    completed = _merge_sample_outputs(run_dir, selected)
    pending = [s for game, rows in selected.items() for s in rows
               if sample_uid(s) not in completed[game]]
    if not pending:
        LOGGER.info("All selected samples are already complete; no model will be loaded")
        _analyze_run_outputs(run_dir, selected_games, args.analyze_after_game)
        return
    device_groups = _device_groups(args)
    # Unique attempt directories preserve previous results even after a crash.
    shard_root = run_dir / "sample-shards"
    shard_root.mkdir(exist_ok=True)
    index = 1
    while True:
        attempt = shard_root / f"attempt-{index:04d}"
        try:
            attempt.mkdir()
            break
        except FileExistsError:
            index += 1
    chunks = [pending[i::args.sample_workers] for i in range(args.sample_workers)]
    parallelism = {
        "partition_by": "sample",
        "worker_count": args.sample_workers,
        "tensor_parallel_size": args.tensor_parallel_size,
        "attempt": str(attempt.relative_to(run_dir)),
        "workers": [
            {"worker_id": i, "cuda_visible_devices": devices,
             "pending_samples": len(chunk), "sample_uids": [sample_uid(s) for s in chunk]}
            for i, (devices, chunk) in enumerate(zip(device_groups, chunks))
        ],
    }
    write_json_atomic(attempt / "plan.json", parallelism)
    manifest = read_json(run_dir / "manifest.json")
    manifest["parallelism"] = parallelism
    write_json_atomic(run_dir / "manifest.json", manifest)
    context = multiprocessing.get_context("spawn")
    worker_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k != "func"})
    processes = []
    for i, (devices, chunk) in enumerate(zip(device_groups, chunks)):
        if not chunk:
            continue
        LOGGER.info("Sample worker %d: GPUs=%s pending=%d", i, ",".join(devices), len(chunk))
        processes.append(context.Process(
            target=_sample_parallel_worker_entry,
            args=(worker_args, chunk, suffix, model_path, run_dir, i, devices, attempt / f"worker-{i}"),
            name=f"course-sample-worker-{i}",
        ))
    try:
        _run_processes(processes)
    except BaseException:
        # Make completed batches visible to the viewer even after interruption.
        _merge_sample_outputs(run_dir, selected)
        raise
    _merge_sample_outputs(run_dir, selected, require_complete=True)
    _analyze_run_outputs(run_dir, selected_games, args.analyze_after_game)


def infer_command(args: argparse.Namespace) -> None:
    samples = load_samples(args.data_dir)
    selected_games = parse_games(args.games, game_names(samples))
    selected_set = set(selected_games)
    selected_specs = [
        answer_spec(sample) for sample in samples if sample["game_name"] in selected_set
    ]
    if args.auto_extract:
        missing_any = any(
            not resolve_image_path(args.data_dir, sample).is_file()
            for sample in samples
            if sample["game_name"] in selected_set
        )
        if missing_any:
            stats = extract_images(args.data_dir)
            LOGGER.info("Image extraction: %s", stats)
    validate_images(args.data_dir, samples, selected_games)

    suffix = args.prompt_suffix
    if args.prompt_suffix_file:
        suffix = args.prompt_suffix_file.read_text(encoding="utf-8").strip()
    model_path = args.model.resolve()
    run_name = args.run_name or slugify(model_path.name)
    run_dir = args.output_dir.resolve() / run_name
    config = _inference_config(args, args.data_dir, model_path, suffix)
    with RunInferenceLock(run_dir):
        _ensure_manifest(
            run_dir / "manifest.json",
            run_name,
            config,
            model_path,
            selected_games,
            selected_specs,
        )

        if getattr(args, "sample_workers", None) is not None:
            _run_sample_parallel(args, selected_games, suffix, model_path, run_dir, samples)
        elif (run_dir / "sample-plan.json").exists():
            raise ValueError("This run uses sample shards; resume with --sample-workers (including 1)")
        elif args.data_parallel_size > 1:
            _run_data_parallel(
                args, selected_games, suffix, model_path, run_dir, samples
            )
        else:
            _run_inference_worker(
                args,
                samples,
                selected_games,
                suffix,
                model_path,
                run_dir,
                worker_id=0,
                analyze_after_game=args.analyze_after_game,
            )
    LOGGER.info("Inference complete. Results: %s", run_dir)


def inspect_command(args: argparse.Namespace) -> None:
    samples = load_samples(args.data_dir)
    counts = Counter(sample["game_name"] for sample in samples)
    option_count = sum(bool(sample.get("options")) for sample in samples)
    answer_modes = Counter(answer_spec(sample).mode for sample in samples)
    report = {
        "data_file": str((args.data_dir / DATA_FILENAME).resolve()),
        "sample_count": len(samples),
        "game_count": len(counts),
        "games": dict(sorted(counts.items())),
        "with_options": option_count,
        "without_options": len(samples) - option_count,
        "answer_modes": dict(sorted(answer_modes.items())),
        "qa_types": dict(
            sorted(Counter(sample.get("qa_type") for sample in samples).items())
        ),
        "image_archive_exists": (args.data_dir / IMAGE_ARCHIVE).is_file(),
        "resolved_images_present": sum(
            resolve_image_path(args.data_dir, sample).is_file() for sample in samples
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def preview_command(args: argparse.Namespace) -> None:
    samples = load_samples(args.data_dir)
    selected_games = parse_games(args.games, game_names(samples))
    suffix = args.prompt_suffix
    if args.prompt_suffix_file:
        suffix = args.prompt_suffix_file.read_text(encoding="utf-8").strip()
    shown = 0
    for sample in samples:
        if sample["game_name"] not in selected_games:
            continue
        spec = answer_spec(sample)
        print(
            json.dumps(
                {
                    "uid": sample_uid(sample),
                    "game_name": sample["game_name"],
                    "image": str(resolve_image_path(args.data_dir, sample)),
                    "answer_mode": spec.mode,
                    "answer_mode_source": spec.source,
                    "label_scheme": spec.label_scheme,
                    "prompt": build_prompt(sample, args.prompt_style, suffix),
                    "reference_answer": sample.get("answer"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        shown += 1
        if shown >= args.count:
            break


def add_data_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)


def add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prompt-style",
        choices=("qwen-json", "original"),
        default="qwen-json",
        help="qwen-json appends a benchmark output-format instruction; original only adds <image>.",
    )
    suffix_group = parser.add_mutually_exclusive_group()
    suffix_group.add_argument("--prompt-suffix", default="")
    suffix_group.add_argument("--prompt-suffix-file", type=Path)


class DeprecatedGameWorkers(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        LOGGER.warning("--data-parallel-size is deprecated; use --game-workers (whole-game partitioning)")
        setattr(namespace, self.dest, values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Summarize local annotations and images"
    )
    add_data_arg(inspect_parser)
    inspect_parser.set_defaults(func=inspect_command)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Safely extract missing test images"
    )
    add_data_arg(prepare_parser)
    prepare_parser.set_defaults(
        func=lambda args: print(json.dumps(extract_images(args.data_dir)))
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check Qwen2.5-VL model/runtime compatibility without loading weights",
    )
    doctor_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    doctor_parser.set_defaults(
        func=lambda args: print(json.dumps(environment_report(args.model), indent=2))
    )

    preview_parser = subparsers.add_parser(
        "preview", help="Preview final prompts without a model"
    )
    add_data_arg(preview_parser)
    add_prompt_args(preview_parser)
    preview_parser.add_argument("--games", nargs="+", default=["all"])
    preview_parser.add_argument("--count", type=int, default=1)
    preview_parser.set_defaults(func=preview_command)

    infer_parser = subparsers.add_parser(
        "infer", help="Run per-game batched ms-swift + vLLM inference"
    )
    add_data_arg(infer_parser)
    add_prompt_args(infer_parser)
    infer_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    infer_parser.add_argument(
        "--adapter", type=Path,
        help="Local PEFT LoRA checkpoint; load alongside --model without merging.",
    )
    infer_parser.add_argument(
        "--checkpoint-model", action="store_true",
        help="Validate and fingerprint --model as a saved full-training HF checkpoint.",
    )
    infer_parser.add_argument(
        "--max-num-seqs", type=int,
        help="Maximum concurrent vLLM sequences (defaults to the engine default).",
    )
    infer_parser.add_argument("--games", nargs="+", default=["all"])
    infer_parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    infer_parser.add_argument("--run-name")
    infer_parser.add_argument("--batch-size", type=int, default=64)
    infer_parser.add_argument("--limit", type=int, help="Per-game debug limit")
    infer_parser.add_argument(
        "--sample-per-game",
        type=int,
        help="Randomly sample this many rows per game before inference.",
    )
    infer_parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed used with the game name for reproducible per-game sampling.",
    )
    infer_parser.add_argument(
        "--analyze-after-game",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save length, format, and answer-behavior analysis after each game.",
    )
    infer_parser.add_argument(
        "--auto-extract", action=argparse.BooleanOptionalAction, default=True
    )
    infer_parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=False
    )
    infer_parser.add_argument("--max-tokens", type=int, default=4096)
    infer_parser.add_argument("--temperature", type=float, default=0.0)
    infer_parser.add_argument("--top-p", type=float, default=1.0)
    infer_parser.add_argument("--top-k", type=int, default=-1)
    infer_parser.add_argument("--presence-penalty", type=float, default=0.0)
    infer_parser.add_argument("--repetition-penalty", type=float, default=1.0)
    infer_parser.add_argument("--seed", type=int, default=0)
    infer_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    workers = infer_parser.add_mutually_exclusive_group()
    workers.add_argument(
        "--game-workers", dest="data_parallel_size", type=int, default=1, metavar="N",
        help="Independent model replicas assigned whole games (default: 1). Each uses tensor-parallel-size GPUs.",
    )
    workers.add_argument(
        "--sample-workers", type=int, metavar="N",
        help="Independent model replicas assigned disjoint samples, including within one game. Each uses tensor-parallel-size GPUs.",
    )
    workers.add_argument(
        "--data-parallel-size", dest="data_parallel_size", type=int, metavar="N",
        default=argparse.SUPPRESS, action=DeprecatedGameWorkers,
        help="Deprecated alias for --game-workers; does NOT partition samples.",
    )
    infer_parser.add_argument("--max-model-len", type=int, default=8192)
    infer_parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    infer_parser.add_argument("--max-retries", type=int, default=2)
    infer_parser.add_argument("--retry-backoff", type=float, default=2.0)
    infer_parser.set_defaults(func=infer_command)
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if getattr(args, "func", None) is infer_command:
        if (args.temperature != 0 or args.max_tokens != 4096 or args.top_p != 1
                or args.top_k != -1 or args.presence_penalty != 0 or args.repetition_penalty != 1
                or args.enable_thinking or args.prompt_style != "qwen-json"
                or args.prompt_suffix or args.prompt_suffix_file):
            raise ValueError("Course evaluation fixes greedy decoding, 4096 output tokens and the fixed course prompt")
        if args.tensor_parallel_size != 1 or args.data_parallel_size != 1 or getattr(args, "sample_workers", None) not in (None,1):
            raise ValueError("Each course experiment must use one GPU and one model replica")
    for name in (
        "batch_size",
        "count",
        "max_tokens",
        "max_model_len",
        "tensor_parallel_size",
        "data_parallel_size",
        "sample_workers",
        "max_num_seqs",
    ):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            flag = "game-workers (--data-parallel-size)" if name == "data_parallel_size" else name.replace("_", "-")
            raise ValueError(f"--{flag} must be positive")
    if getattr(args, "limit", None) is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if getattr(args, "sample_per_game", None) is not None and args.sample_per_game <= 0:
        raise ValueError("--sample-per-game must be positive")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        validate_cli_args(args)
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
