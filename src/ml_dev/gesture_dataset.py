from pathlib import Path

import cbor2
import numpy as np


LABEL_NAMES = ["idle", "snake", "updown", "wave"]


def load_gesture_data(
    data_root: Path, training: bool, window_size: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    split = "training" if training else "testing"

    sample_fragments, label_fragments = [], []

    for label_idx, name in enumerate(LABEL_NAMES):
        for file_path in (data_root / split).glob(f"{name}.*.cbor"):
            with file_path.open("rb") as in_file:
                raw_obj = cbor2.load(in_file)
                signal = np.array(raw_obj["payload"]["values"], dtype=np.float32)
                windows = _extract_running_windows(signal, window_size, stride)
                labels = np.array([label_idx] * len(windows))

                sample_fragments.append(windows)
                label_fragments.append(labels)

    samples = np.concatenate(sample_fragments)
    labels = np.concatenate(label_fragments)

    return samples, labels


def _extract_running_windows(
    signal: np.ndarray, window_size: int, stride: int
) -> np.ndarray:
    signal_length, _ = signal.shape

    if signal_length - window_size < 0:
        raise ValueError(
            f"Signal with length {signal_length} cannot divided into windows of size {window_size}."
        )

    start_indices = range(0, signal_length - window_size + 1, stride)
    windows = [signal[i : i + window_size] for i in start_indices]

    return np.stack(windows)
