import numpy as np

EPSILON = 1e-8  # small value to avoid division by zero


def compute_mean(window: np.ndarray) -> np.ndarray:
    """
    Compute the mean of each channel across the time axis.
    Input shape: [n_samples, n_channels]
    Return shape: [n_channels]
    """
    return np.mean(window, axis=0)


def compute_std(window: np.ndarray) -> np.ndarray:
    """
    Compute the standard deviation of each channel across the time axis.
    Input shape: [n_samples, n_channels]
    Return shape: [n_channels]
    """
    return np.std(window, axis=0)


def compute_rms(window: np.ndarray) -> np.ndarray:
    """
    Compute the Root Mean Square of each channel.
    Formula: sqrt(mean(x^2)) per channel.
    Input shape: [n_samples, n_channels]
    Return shape: [n_channels]
    """
    return np.sqrt(np.mean(window ** 2, axis=0))


def compute_zero_crossing_rate(window: np.ndarray) -> np.ndarray:
    """
    Compute the zero crossing rate of each channel.
    A zero crossing occurs when consecutive samples have opposite signs.
    Use np.sign() and np.diff() to detect sign changes.
    Normalize by dividing by the number of samples.
    Input shape: [n_samples, n_channels]
    Return shape: [n_channels]
    """
    signs = np.sign(window)
    diff = np.diff(signs, axis=0)
    crossings = np.abs(diff) > 0
    count = np.sum(crossings, axis=0)

    return count / len(window)


def compute_signal_magnitude_area(window: np.ndarray) -> float:
    """
    Compute the Signal Magnitude Area across all channels.
    Formula: sum of absolute values across all samples and channels,
    divided by number of samples.
    Input shape: [n_samples, n_channels]
    Return shape: scalar float
    """
    return np.sum(np.abs(window)) / len(window)


def compute_skewness(window: np.ndarray) -> np.ndarray:
    """
    Compute the skewness of each channel.
    Skewness measures asymmetry of the signal distribution.
    Formula: mean((x - mean)^3) / std^3 per channel.
    Use EPSILON in the denominator to avoid division by zero.
    Input shape: [n_samples, n_channels]
    Return shape: [n_channels]
    """
    mean = compute_mean(window)
    std = compute_std(window)

    return np.mean((window - mean) ** 3, axis=0) / (std ** 3 + EPSILON)


def compute_kurtosis(window: np.ndarray) -> np.ndarray:
    """
    Compute the kurtosis of each channel.
    Kurtosis measures the "tailedness" of the signal distribution.
    Formula: mean((x - mean)^4) / std^4 per channel.
    Use EPSILON in the denominator to avoid division by zero.
    Input shape: [n_samples, n_channels]
    Return shape: [n_channels]
    """
    mean = compute_mean(window)
    std = compute_std(window)

    return np.mean((window - mean) ** 4, axis=0) / (std ** 4 + EPSILON)


def compute_peak_to_peak(window: np.ndarray) -> np.ndarray:
    """
    Compute the peak to peak amplitude of each channel.
    Formula: max(x) - min(x) per channel.
    Input shape: [n_samples, n_channels]
    Return shape: [n_channels]
    """
    return np.max(window, axis=0) - np.min(window, axis=0)


def extract_time_domain_features(window: np.ndarray) -> np.ndarray:
    """
    Aggregate all time domain features into a single flat feature vector.
    Call every function above and concatenate their outputs using np.concatenate.
    Input shape: [n_samples, n_channels]
    Return shape: [n_features] — a single flat 1D array
    """
    return np.concatenate([
        compute_mean(window),
        compute_std(window),
        compute_rms(window),
        compute_zero_crossing_rate(window),
        np.array([compute_signal_magnitude_area(window)]),
        compute_skewness(window),
        compute_kurtosis(window),
        compute_peak_to_peak(window)
    ])
