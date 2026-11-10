"""Stable sample identity shared by inference and read-only result tools."""

from typing import Any


def sample_uid(sample: dict[str, Any]) -> str:
    # The released test set has a (data_id, question_id) collision in Pac-Man.
    # Preserve the original image reference, not a machine-specific resolved path.
    return f"{sample['data_id']}::{sample['question_id']}::{sample['image']}"
