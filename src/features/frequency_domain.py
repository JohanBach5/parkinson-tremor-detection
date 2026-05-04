import numpy as np
from scipy.signal import welch

EPSILON = 1e-8  # small value to avoid division by zero


def compute_fft_magnitude(
        window: np.ndarray,
        fs: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the FFT magnitude spectrum of each channel.
    Use np.fft.rfft() which returns only positive frequencies.
    Use np.fft.rfftfreq() to compute the corresponding frequency axis.
    Normalize magnitude by dividing by number of samples.
    Input shape: [n_samples, n_channels]
    Return: tuple of (freqs, magnitude) where
    - freqs shape: [n_freqs] — same for all channels
    - magnitude shape: [n_freqs, n_channels]
    """
    fft_vals = np.fft.rfft(window, axis=0)
    magnitude = np.abs(fft_vals) / len(window)
    freqs = np.fft.rfftfreq(len(window), d=1/fs)

    return freqs, magnitude


def compute_power_spectral_density(
        window: np.ndarray,
        fs: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Power Spectral Density of each channel using Welch's method.
    Use scipy.signal.welch() with nperseg=min(len(window), 128).
    Input shape: [n_samples, n_channels]
    Return: tuple of (freqs, psd) where
    - freqs shape: [n_freqs]
    - psd shape: [n_freqs, n_channels]
    """
    all_psd = []
    freqs = None

    for ch in range(window.shape[1]):
        freqs, psd = welch(window[:, ch], fs=fs, nperseg=min(len(window), 128))
        all_psd.append(psd)

    psd = np.stack(all_psd, axis=1) # noqa

    return freqs, psd


def compute_tremor_band_power(
        freqs: np.ndarray,
        psd: np.ndarray,
        low_hz: float = 4.0,
        high_hz: float = 6.0
) -> np.ndarray:
    """
    Compute the total power in the tremor frequency band (4-6 Hz)
    for each channel.
    Create a boolean mask where freqs >= low_hz and freqs <= high_hz.
    Sum the PSD values within this band per channel.
    Input:
    - freqs shape: [n_freqs]
    - psd shape: [n_freqs, n_channels]
    Return shape: [n_channels]
    """
    tremor_band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    tremor_band_psd = psd[tremor_band_mask, :]

    return np.sum(tremor_band_psd, axis=0)


def compute_dominant_frequency(
        freqs: np.ndarray,
        psd: np.ndarray
) -> np.ndarray:
    """
    Find the frequency with the highest power for each channel.
    Use np.argmax() along axis=0 to find the index of max PSD per channel.
    Return the corresponding frequency value from freqs.
    Input:
    - freqs shape: [n_freqs]
    - psd shape: [n_freqs, n_channels]
    Return shape: [n_channels]
    """
    max_idx = np.argmax(psd, axis=0)

    return freqs[max_idx]


def compute_spectral_entropy(
        psd: np.ndarray
) -> np.ndarray:
    """
    Compute the spectral entropy of each channel.
    Normalize PSD to get a probability distribution per channel:
        p = psd / (sum(psd) + EPSILON)
    Then compute entropy:
        entropy = -sum(p * log(p + EPSILON)) per channel
    Input shape: [n_freqs, n_channels]
    Return shape: [n_channels]
    """
    p = psd / (np.sum(psd, axis=0) + EPSILON)

    return -np.sum(p * np.log(p + EPSILON), axis=0)


def compute_spectral_edge_frequency(
        freqs: np.ndarray,
        psd: np.ndarray,
        edge: float = 0.95
) -> np.ndarray:
    """
    Find the frequency below which 'edge' fraction (e.g. 95%) of total
    spectral power is contained, for each channel.
    Steps per channel:
    1. Compute cumulative sum of PSD along frequency axis
    2. Normalize by total power
    3. Find the first frequency index where cumulative power >= edge
    4. Return the corresponding frequency from freqs
    Input:
    - freqs shape: [n_freqs]
    - psd shape: [n_freqs, n_channels]
    Return shape: [n_channels]
    """
    result = []

    cumulative_power = np.cumsum(psd, axis=0)
    total_power = cumulative_power[-1, :]
    normalized = cumulative_power / (total_power + EPSILON)

    for ch in range(psd.shape[1]):
        idx = np.argmax(normalized[:, ch] >= edge)
        result.append(freqs[idx])

    return np.array(result)


def extract_frequency_domain_features(
        window: np.ndarray,
        fs: int,
        low_hz: float = 4.0,
        high_hz: float = 6.0
) -> np.ndarray:
    """
    Aggregate all frequency domain features into a single flat feature vector.
    Steps:
    1. Call compute_power_spectral_density() to get freqs and psd
    2. Call compute_tremor_band_power()
    3. Call compute_dominant_frequency()
    4. Call compute_spectral_entropy()
    5. Call compute_spectral_edge_frequency()
    Concatenate all results into a single flat 1D array.
    Input shape: [n_samples, n_channels]
    Return shape: [n_features]
    """
    freqs, psd = compute_power_spectral_density(window, fs)

    return np.concatenate([
        compute_tremor_band_power(freqs, psd, low_hz, high_hz),
        compute_dominant_frequency(freqs, psd),
        compute_spectral_entropy(psd),
        compute_spectral_edge_frequency(freqs, psd)
    ])
