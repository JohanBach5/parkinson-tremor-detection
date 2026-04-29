import numpy as np
import pandas as pd

from src.data.preprocessor import SENSOR_COLUMNS


class SlidingWindowSegmentor:

    def __init__(self, config: dict) -> None:
        """
        Store the config dictionary.
        Extract and store as instance variables:
        - window_size_sec from config["segmentation"]["window_size_sec"]
        - overlap from config["segmentation"]["overlap"]
        - label_threshold from config["segmentation"]["label_threshold"]
        - target_fs from config["sampling"]["target_fs"]
        Compute and store window_size and step_size in samples:
        - window_size = int(window_size_sec * target_fs)
        - step_size = int(window_size * (1 - overlap))
        """
        self.config = config
        self.window_size_sec = self.config["segmentation"]["window_size_sec"]
        self.overlap = self.config["segmentation"]["overlap"]
        self.label_threshold = self.config["segmentation"]["label_threshold"]
        self.target_fs = self.config["sampling"]["target_fs"]
        self.window_size = int(self.window_size_sec * self.target_fs)
        self.step_size = int(self.window_size * (1 - self.overlap))

    def segment_subject(
            self,
            df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Segment one subject's DataFrame into sliding windows.
        Extract the sensor data as a numpy array from SENSOR_COLUMNS.
        Extract the label array from the label column.
        Call _validate_window_params() first.
        Call _create_windows() to get the windowed signal array.
        Call _assign_labels() to get the label per window.
        Return a tuple of (windows, labels) where:
        - windows shape is [n_windows, window_size, n_channels]
        - labels shape is [n_windows]
        """
        signal = df[SENSOR_COLUMNS].values
        label_array = df["label"].values

        self._validate_window_params(len(signal), self.window_size, self.step_size)
        windows = self._create_windows(signal, self.window_size, self.step_size)
        labels = self._assign_labels(label_array, self.window_size, self.step_size, self.label_threshold)

        return windows, labels

    @staticmethod
    def _create_windows(
            signal: np.ndarray,
            window_size: int,
            step_size: int
    ) -> np.ndarray:
        """
        Slice the signal into overlapping windows using a loop.
        Each window is signal[start:start+window_size] where
        start advances by step_size each iteration.
        Stop when start + window_size exceeds signal length.
        Return a numpy array of shape [n_windows, window_size, n_channels].
        """
        windows = []
        start = 0

        while start + window_size <= len(signal):
            window = signal[start: start + window_size]
            windows.append(window)
            start += step_size

        return np.array(windows)

    @staticmethod
    def _assign_labels(
            label_array: np.ndarray,
            window_size: int,
            step_size: int,
            threshold: float = 0.5
    ) -> np.ndarray:
        """
        Assign one label per window using majority voting.
        For each window, compute the fraction of samples with label == 2
        (FoG episode).
        If fraction > threshold assign label 1 (FoG).
        Otherwise assign label 0 (normal).
        Return a numpy array of shape [n_windows] with binary labels.
        """
        labels = []
        start = 0

        while start + window_size <= len(label_array):
            window_labels = label_array[start: start + window_size]
            fog_count = np.sum(window_labels == 2)
            fraction = fog_count / len(window_labels)
            label = int(fraction > threshold)
            labels.append(label)
            start += step_size

        return np.array(labels)

    @staticmethod
    def _validate_window_params(
            signal_length: int,
            window_size: int,
            step_size: int
    ) -> None:
        """
        Raise a ValueError if window_size >= signal_length.
        Raise a ValueError if step_size <= 0.
        Raise a ValueError if step_size >= window_size.
        Otherwise return None.
        """
        if window_size >= signal_length:
            raise ValueError(
                f"window_size ({window_size}) must be smaller than "
                f"signal_length ({signal_length})"
            )

        if step_size <= 0:
            raise ValueError(
                f"step_size ({step_size}) must be greater than 0"
            )

        if step_size >= window_size:
            raise ValueError(
                f"step_size ({step_size}) must be smaller than "
                f"window_size ({window_size})"
            )

        return None


def segment_all_subjects(
    data: dict[str, pd.DataFrame],
    config: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function that creates a SlidingWindowSegmentor instance
    and calls segment_subject() on every DataFrame in the data dictionary.
    Concatenate all windows, labels, and subject IDs across subjects.
    Return a tuple of (all_windows, all_labels, all_subject_ids).
    """
    segmentor = SlidingWindowSegmentor(config)

    all_windows = []
    all_labels = []
    all_subject_ids = []

    for subject_id, df in data.items():
        subject_windows, subject_labels = segmentor.segment_subject(df)
        n_windows = len(subject_windows)

        all_windows.append(subject_windows)
        all_labels.append(subject_labels)
        all_subject_ids.extend([subject_id] * n_windows)

    return (
        np.concatenate(all_windows, axis=0),
        np.concatenate(all_labels, axis=0),
        np.array(all_subject_ids)
    )


