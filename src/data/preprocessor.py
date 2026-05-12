import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.signal import resample as scipy_resample


def get_sensor_columns(config: dict) -> list[str]:
    """
    Build sensor column names dynamically from config.
    For Daphnet: ankle, hip, wrist × x, y, z → 9 columns
    For CIS-PD: wrist × x, y, z → 3 columns
    """
    sensors = config["dataset"]["sensors"]
    axes = config["dataset"]["sensor_axes"]
    return [f"{sensor}_acc_{axis.lower()}" for sensor in sensors for axis in axes]


METADATA_COLUMNS = ["timestamp", "label", "subject_id", "session_id"]


class SignalPreprocessor:

    def __init__(self, config: dict) -> None:
        """
        Store the config dictionary.
        Extract and store as instance variables:
        - target_fs from config["sampling"]["target_fs"]
        - lowcut from config["preprocessing"]["bandpass_lowcut"]
        - highcut from config["preprocessing"]["bandpass_highcut"]
        - filter_order from config["preprocessing"]["filter_order"]
        - normalization_method from config["preprocessing"]["normalization_method"]
        """
        self.config = config
        self.sensor_columns = get_sensor_columns(config)
        self.target_fs = self.config["sampling"]["target_fs"]
        self.lowcut = self.config["preprocessing"]["bandpass_lowcut"]
        self.highcut = self.config["preprocessing"]["bandpass_highcut"]
        self.filter_order = self.config["preprocessing"]["filter_order"]
        self.normalization_method = self.config["preprocessing"]["normalization_method"]

    def process_subject(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point for preprocessing one subject's DataFrame.
        Call resample(), then bandpass_filter(), then normalize()
        in that exact order.
        Return the fully preprocessed DataFrame.
        """
        df = self.resample(df, self.target_fs)
        df = self.bandpass_filter(df)
        df = self.normalize(df)

        return df

    def bandpass_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a Butterworth bandpass filter to all sensor columns.
        Use self.lowcut, self.highcut, self.filter_order, self.target_fs.
        Call _compute_filter_coefficients() to get b and a coefficients.
        Call _apply_filter_to_all_axes() to apply the filter.
        Do not filter the label, subject_id, or session_id columns.
        Return the filtered DataFrame.
        """
        b, a = self._compute_filter_coefficients(
            self.lowcut, self.highcut, self.target_fs, self.filter_order
        )
        df = self._apply_filter_to_all_axes(df, b, a)

        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize all sensor columns using the method defined in
        self.normalization_method.
        For 'zscore': subtract mean and divide by std per column.
        For 'minmax': scale each column to [0, 1] range.
        Do not normalize the label, subject_id, or session_id columns.
        Return the normalized DataFrame.
        """
        if self.normalization_method == "zscore":
            df[self.sensor_columns] = (
                                              df[self.sensor_columns] - df[self.sensor_columns].mean()
                                      ) / df[self.sensor_columns].std()

        elif self.normalization_method == "minmax":
            df[self.sensor_columns] = (
                                              df[self.sensor_columns] - df[self.sensor_columns].min()
                                      ) / (df[self.sensor_columns].max() - df[self.sensor_columns].min())

        else:
            raise ValueError(
                f"Unrecognised normalization method: {self.normalization_method}. "
                f"Options are: 'zscore', 'minmax'."
            )

        return df

    def resample(self, df: pd.DataFrame, target_fs: int) -> pd.DataFrame:
        """
        Resample the signal to target_fs if the current sampling rate
        differs from target_fs.
        For Daphnet this will be a no-op since it is already at 64 Hz.
        Use scipy.signal.resample to resample each sensor column.
        Do not resample the label, subject_id, or session_id columns.
        Return the resampled DataFrame.
        """
        current_fs = int(1000 / df["timestamp"].diff().median())

        if abs(current_fs - target_fs) > 2:
            num_samples = int(len(df) * target_fs / current_fs)
            df[self.sensor_columns] = scipy_resample(
                df[self.sensor_columns].values, num_samples
            )

        return df

    def _apply_filter_to_all_axes(
            self,
            df: pd.DataFrame,
            b: np.ndarray,
            a: np.ndarray
    ) -> pd.DataFrame:
        """
        Apply the filter defined by coefficients b and a to every
        sensor column in the DataFrame using scipy.signal.filtfilt.
        filtfilt applies the filter forward and backward to avoid
        phase distortion.
        Return the filtered DataFrame.
        """
        for col in self.sensor_columns:
            df[col] = filtfilt(b, a, df[col].values)

        return df

    @staticmethod
    def _compute_filter_coefficients(
            lowcut: float,
            highcut: float,
            fs: int,
            order: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Butterworth bandpass filter coefficients.
        Use scipy.signal.butter with btype='band'.
        Normalize frequencies by Nyquist frequency (fs / 2).
        Return the tuple (b, a) of filter coefficients.
        """
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(order, [low, high], btype='band', output='ba')  # noqa

        return b, a


def preprocess_all_subjects(
        data: dict[str, pd.DataFrame],
        config: dict
) -> dict[str, pd.DataFrame]:
    """
    Convenience function that creates a SignalPreprocessor instance
    and calls process_subject() on every DataFrame in the data dictionary.
    Return a new dictionary with the same keys but preprocessed DataFrames.
    """
    preprocessed_data = {}
    preprocessor = SignalPreprocessor(config)

    for session_key, df in data.items():
        preprocessed_data[session_key] = preprocessor.process_subject(df)

    return preprocessed_data
