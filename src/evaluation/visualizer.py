import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold_id: str,
        output_dir: str
) -> None:
    """
    Plot and save a confusion matrix for one LOSO fold.
    """
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Normal", "FoG"],
        yticklabels=["Normal", "FoG"],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix — {fold_id}"
    )

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", color="black", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{fold_id}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
        fold_results: list[dict],
        output_dir: str
) -> None:
    """
    Plot ROC curves for all LOSO folds on the same figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    aucs = []
    for fold in fold_results:
        y_true = fold["y_test"]
        y_proba = fold["y_proba"]

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)
        ax.plot(fpr, tpr, alpha=0.3, linewidth=1, color="steelblue")

    mean_auc = np.mean(aucs) if aucs else 0.0
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC Curves — All Folds (Mean AUC = {mean_auc:.3f})"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_signal_with_predictions(
        signal: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fs: int,
        output_dir: str,
        subject_id: str
) -> None:
    """
    Plot raw accelerometer signal overlaid with true and predicted labels.
    """
    os.makedirs(output_dir, exist_ok=True)

    time = np.arange(len(signal)) / fs

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, signal[:, 0], linewidth=0.8, color="black", label="ankle_acc_x")

    window_size = len(signal) // len(y_true)
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        start = i * window_size / fs
        end = (i + 1) * window_size / fs
        if true == 1:
            ax.axvspan(start, end, alpha=0.2, color="green", label="True FoG" if i == 0 else "")
        if pred == 1:
            ax.axvspan(start, end, alpha=0.2, color="red", label="Predicted FoG" if i == 0 else "")

    ax.set(xlabel="Time (s)", ylabel="Acceleration",
           title=f"Signal with Predictions — {subject_id}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"signal_{subject_id}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
        feature_importances: np.ndarray,
        feature_names: list[str],
        output_dir: str,
        top_n: int = 20
) -> None:
    """
    Plot a horizontal bar chart of the top_n most important features.
    """
    os.makedirs(output_dir, exist_ok=True)

    indices = np.argsort(feature_importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_values = feature_importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top_values[::-1], color="steelblue")
    ax.set(
        yticks=range(top_n),
        yticklabels=top_names[::-1],
        xlabel="Importance",
        title=f"Top {top_n} Feature Importances"
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
        train_losses: list[float],
        val_losses: list[float],
        output_dir: str
) -> None:
    """
    Plot training and validation loss curves over epochs.
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss", color="steelblue")
    ax.plot(epochs, val_losses, label="Val loss", color="orange")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training Curves")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_results_summary(
        aggregated_metrics: dict,
        model_type: str,
        output_dir: str
) -> None:
    """
    Plot a horizontal bar chart summarising all aggregated metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["sensitivity", "specificity", "f1", "auc_roc", "cohen_kappa"]
    means = [aggregated_metrics[f"{m}_mean"] for m in metrics]
    stds = [aggregated_metrics[f"{m}_std"] for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(metrics, means, xerr=stds, color="steelblue",
                   error_kw={"capsize": 5})
    ax.set(
        xlim=[0, 1],
        xlabel="Score",
        title=f"Results Summary — {model_type}"
    )

    for bar, mean in zip(bars, means):
        ax.text(mean + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{mean:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"results_summary_{model_type}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()