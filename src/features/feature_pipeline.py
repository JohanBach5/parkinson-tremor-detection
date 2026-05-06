import os
import numpy as np

from src.features.time_domain import extract_time_domain_features
from src.features.frequency_domain import extract_frequency_domain_features
from src.data.preprocessor import SENSOR_COLUMNS


def save_feature_matrix(
        X: np.ndarray,
        y: np.ndarray,
        output_path: str
) -> None:
    """
    Save the feature matrix X and labels y to disk as a .npz file.
    Use np.savez() with keys 'X' and 'y'.
    Create the output directory if it does not exist.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, X=X, y=y)


def load_feature_matrix(
        input_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a feature matrix and labels from a .npz file saved by
    save_feature_matrix().
    Use np.load() to load the file.
    Return a tuple of (X, y).
    """
    data = np.load(input_path)
    return data["X"], data["y"]


class FeaturePipeline:

    def __init__(self, config: dict) -> None:
        """
        Store the config dictionary.
        Extract and store as instance variables:
        - target_fs from config["sampling"]["target_fs"]
        - tremor_band_low from config["features"]["tremor_band_low"]
        - tremor_band_high from config["features"]["tremor_band_high"]
        """
        self.config = config
        self.target_fs = self.config["sampling"]["target_fs"]
        self.tremor_band_low = self.config["features"]["tremor_band_low"]
        self.tremor_band_high = self.config["features"]["tremor_band_high"]

    def extract(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract features from all windows.
        Loop over every window and call extract_single_window().
        Stack results into a 2D feature matrix.
        Input shape: [n_windows, n_samples, n_channels]
        Return shape: [n_windows, n_features]
        """
        result = []

        for window in windows:
            extracted_window = self.extract_single_window(window)
            result.append(extracted_window)

        X = np.stack(result)

        # replace NaN and inf values with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X

    def extract_single_window(self, window: np.ndarray) -> np.ndarray:
        """
        Extract all features from a single window.
        Call extract_time_domain_features() and
        extract_frequency_domain_features() and concatenate results.
        Input shape: [n_samples, n_channels]
        Return shape: [n_features]
        """
        time_domain_feature = extract_time_domain_features(window)
        frequency_domain_features = extract_frequency_domain_features(
            window,
            self.target_fs,
            self.tremor_band_low,
            self.tremor_band_high
        )

        return np.concatenate([time_domain_feature, frequency_domain_features])

    @staticmethod
    def get_feature_names() -> list[str]:
        """
        Return a list of human-readable feature names in the same
        order as the columns in the feature matrix.
        Time domain names follow pattern: {feature}_{channel}
        e.g. 'mean_ankle_acc_x', 'std_hip_acc_y'
        Frequency domain names follow pattern: {feature}_{channel}
        e.g. 'tremor_band_power_ankle_acc_x'
        This list must match exactly the output of extract() column by column.
        """
        names = []

        # time domain — per channel features
        time_features = ["mean", "std", "rms", "zcr"]
        for feature in time_features:
            for channel in SENSOR_COLUMNS:
                names.append(f"{feature}_{channel}")

        # sma — single scalar
        names.append("sma")

        # time domain — remaining per channel features
        time_features_2 = ["skewness", "kurtosis", "peak_to_peak"]
        for feature in time_features_2:
            for channel in SENSOR_COLUMNS:
                names.append(f"{feature}_{channel}")

        # frequency domain — per channel features
        freq_features = [
            "tremor_band_power",
            "dominant_frequency",
            "spectral_entropy",
            "spectral_edge_frequency"
        ]
        for feature in freq_features:
            for channel in SENSOR_COLUMNS:
                names.append(f"{feature}_{channel}")

        return names
