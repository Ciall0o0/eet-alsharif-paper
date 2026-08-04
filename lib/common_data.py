"""Shared data loading + stratified split for eet-rl-research.

Loads augmented_dataset.npz directly (no dependency on eet/datasets layout).
Both DDQN (89-dim) and LSTM+PPO (89-dim) training/validation use this so the
comparison is fair: same data split, same no-destination input space.
"""
from __future__ import annotations
from pathlib import Path


import numpy as np
from sklearn.model_selection import train_test_split

PROJ = Path(__file__).resolve().parents[1]
AUG_PATH = PROJ / "augmented_dataset.npz"


def load_augmented():
    """Return dict with keys event_sequences(N,L,10), event_lengths(N,), labels(N,)."""
    assert AUG_PATH.exists(), f"Missing {AUG_PATH}; run data_augmentation.py first"
    d = np.load(str(AUG_PATH), allow_pickle=True)
    return {
        "event_sequences": d["event_sequences"],
        "event_lengths": d["event_lengths"].astype(np.int64),
        "labels": np.squeeze(d["labels"]).astype(np.int64),
    }


def get_splits(test_size: float = 0.15, seed: int = 42):
    """Stratified train/val split. Returns (data_dict, train_idx, val_idx)."""
    data = load_augmented()
    labels = data["labels"]
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=test_size, random_state=seed, stratify=labels
    )
    return data, train_idx, val_idx
