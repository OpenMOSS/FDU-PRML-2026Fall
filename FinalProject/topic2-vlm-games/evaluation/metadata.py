"""Shared game-reasoning benchmark metadata."""

from __future__ import annotations

from typing import Any


# Answer-mode exceptions audited against reference benchmark's released test JSON.
# Most rows with a non-empty ``options`` field use explicit labels and can infer
# the label kind from the official answer. These tasks are the deliberate
# exceptions to that rule.
ANSWER_MODE_OVERRIDES = {
    ("hue", "0"): {
        "mode": "option_value",
        "source": "task_override",
        "reason": "Palette numbers are answer values, not separate option labels.",
    },
    ("zuma", "4"): {
        "mode": "option_value",
        "source": "task_override",
        "reason": "The question lists allowed color values without labels.",
    },
    ("tetris", "1"): {
        "mode": "number_label",
        "source": "task_override",
        "reason": "The question has numbered choices although options is null.",
    },
}


# Tetris question 1 omits the structured options field. Keeping its fixed
# choices here makes UI selection mapping and relaxed diagnostics complete.
TASK_OPTION_OVERRIDES = {
    ("tetris", "1"): (
        "I",
        "O",
        "T",
        "L",
        "J",
        "S",
        "Z",
        "no falling Tetrimino",
    ),
}


# Numeric labels are one-based unless an audited task declares otherwise.
LABEL_SCHEME_OVERRIDES = {
    ("snake", "3"): "zero_based_numeric",
}


# The reference task cognitive categories. The released JSON metadata labels some
# games differently, so evaluation and visualization use this fixed mapping.
COGNITIVE_CATEGORIES = {
    "3d_maze": "3D Spatial Perception and Understanding",
    "rubiks_cube": "3D Spatial Perception and Understanding",
    "3d_reconstruction": "3D Spatial Perception and Understanding",
    "pyramidchess": "3D Spatial Perception and Understanding",
    "minecraft": "3D Spatial Perception and Understanding",
    "tengram": "Pattern Recognition and Matching",
    "freecell": "Pattern Recognition and Matching",
    "tetris": "Pattern Recognition and Matching",
    "zuma": "Pattern Recognition and Matching",
    "spider_solitaire": "Pattern Recognition and Matching",
    "hue": "Pattern Recognition and Matching",
    "jewel2": "Pattern Recognition and Matching",
    "klondike": "Pattern Recognition and Matching",
    "langton_ant": "Multi-step Reasoning",
    "2d_turing_machine": "Multi-step Reasoning",
    "word_search": "Multi-step Reasoning",
    "tents": "Multi-step Reasoning",
    "rhythm_game": "Multi-step Reasoning",
    "star_battle": "Multi-step Reasoning",
    "sudoku": "Multi-step Reasoning",
    "lifegame": "Multi-step Reasoning",
    "minesweeper": "Multi-step Reasoning",
    "sokoban": "Strategic Planning",
    "maze": "Strategic Planning",
    "tictactoe": "Strategic Planning",
    "ultra_tictactoe": "Strategic Planning",
    "space_invaders": "Strategic Planning",
    "snake": "Strategic Planning",
    "chess_ranger": "Strategic Planning",
    "pacman": "Strategic Planning",
}


def cognitive_category(game_name: Any, fallback: Any = None) -> str:
    game = str(game_name or "")
    if game in COGNITIVE_CATEGORIES:
        return COGNITIVE_CATEGORIES[game]
    return str(fallback or "Other")
