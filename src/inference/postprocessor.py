import numpy as np


def smooth_predictions(
        predictions: np.ndarray,
        window_size: int = 5
) -> np.ndarray:
    """
    Apply majority vote smoothing over a rolling window of predictions.
    """
    smoothed = predictions.copy()
    half = window_size // 2

    for i in range(len(predictions)):
        start = max(0, i - half)
        end = min(len(predictions), i + half + 1)
        neighbourhood = predictions[start:end]
        smoothed[i] = 1 if np.sum(neighbourhood) > len(neighbourhood) / 2 else 0

    return smoothed


def apply_confidence_threshold(
        probabilities: np.ndarray,
        threshold: float = 0.6
) -> np.ndarray:
    """
    Convert probabilities to binary predictions using a custom threshold.
    """
    return (probabilities[:, 1] >= threshold).astype(int)


def postprocess(
        predictions: np.ndarray,
        probabilities: np.ndarray,
        threshold: float = 0.6,
        smooth_window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply confidence threshold then smooth predictions.
    """
    predictions = apply_confidence_threshold(probabilities, threshold)
    predictions = smooth_predictions(predictions, smooth_window)
    return predictions, probabilities
