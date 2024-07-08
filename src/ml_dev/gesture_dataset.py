from pathlib import Path

import cbor2
import numpy as np
import tensorflow as tf


LABEL_NAMES = ["idle", "snake", "updown", "wave"]


def load_gesture_dataset(
    data_root: Path, train_split: bool, window_size: int = 125, stride: int = 1
) -> tf.data.Dataset:
    split = "training" if train_split else "testing"

    sample_fragments, label_fragments = [], []

    for label_idx, name in enumerate(LABEL_NAMES):
        for file_path in (data_root / split).glob(f"{name}.*.cbor"):
            with file_path.open("rb") as in_file:
                raw_obj = cbor2.load(in_file)
                signal = np.array(raw_obj["payload"]["values"], dtype=np.float32)
                windows = _extract_running_windows(signal, window_size, stride)
                window_labels = np.array([label_idx] * len(windows), dtype=np.int32)

                sample_fragments.append(windows)
                label_fragments.append(window_labels)

    samples = tf.convert_to_tensor(np.concatenate(sample_fragments))
    labels = tf.convert_to_tensor(np.concatenate(label_fragments))

    return tf.data.Dataset.from_tensor_slices((samples, labels))


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
