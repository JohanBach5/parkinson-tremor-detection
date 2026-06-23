import argparse
import pickle
import os
import yaml
import numpy as np

from src.data.loader import get_loader
from src.data.preprocessor import preprocess_all_subjects
from src.data.segmentor import segment_all_subjects
from src.features.feature_pipeline import FeaturePipeline, save_feature_matrix
from src.training.trainer import LOSOTrainer, aggregate_fold_results
from src.evaluation.visualizer import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_results_summary
)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Parkinson Disease Detection Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/daphnet_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "preprocess", "train", "evaluate"],
        help="Which stage to run"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load and return config from a YAML file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_preprocessing_pipeline(config: dict) -> tuple:
    """
    Run the full preprocessing pipeline.
    Load → preprocess → segment → extract features.
    Return (X, y, all_subject_ids).
    """
    dataset_name = config["dataset"]["name"]

    print("\n📂 Loading data...")
    loader = get_loader(dataset_name, config)
    raw_data = loader.load_all_subjects()
    preprocessed_data = preprocess_all_subjects(raw_data, config)
    print(f"   {len(raw_data)} sessions loaded and preprocessed")

    print("\n🔪 Segmenting...")
    all_windows, all_labels, all_subject_ids = segment_all_subjects(
        preprocessed_data, config
    )
    print(f"   {all_windows.shape[0]} windows — shape {all_windows.shape}")

    print("\n⚙️  Extracting features...")
    pipeline = FeaturePipeline(config)
    X = pipeline.extract(all_windows)
    y = all_labels
    print(f"   Feature matrix shape: {X.shape}")

    output_path = f"{config['paths']['segments_dir']}/features.npz"
    save_feature_matrix(X, y, output_path)
    print(f"   Feature matrix saved to {output_path}")

    return X, y, all_subject_ids


def run_training_pipeline(
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        config: dict
) -> dict:
    """
    Run the training pipeline.
    Create LOSOTrainer, run LOSO training.
    Return results dict.
    """
    model_type = config["training"]["model_type"]
    task_type = config["training"].get("task_type", "classification")
    n_subjects = len(np.unique(subject_ids))

    print(f"\n🚀 Training {model_type} ({task_type}) with LOSO ({n_subjects} folds)...")
    trainer = LOSOTrainer(config)
    results = trainer.run(X, y, subject_ids)
    print(f"   Training complete — {len(results['fold_results'])} folds")

    return results


def run_evaluation_pipeline(
        results: dict,
        config: dict
) -> None:
    """
    Run the evaluation pipeline.
    Aggregate fold results, print metrics, generate plots.
    """
    figures_dir = config["paths"]["figures_dir"]
    model_type = results["model_type"]
    task_type = config["training"].get("task_type", "classification")
    fold_results = results["fold_results"]

    # aggregate metrics
    aggregated = aggregate_fold_results(fold_results)

    # print results
    print(f"\n📊 Results — {model_type} ({task_type})")

    if task_type == "classification":
        print(f"   Sensitivity:  {aggregated['sensitivity_mean']:.3f} ± {aggregated['sensitivity_std']:.3f}")
        print(f"   Specificity:  {aggregated['specificity_mean']:.3f} ± {aggregated['specificity_std']:.3f}")
        print(f"   F1 Score:     {aggregated['f1_mean']:.3f} ± {aggregated['f1_std']:.3f}")
        print(f"   AUC-ROC:      {aggregated['auc_roc_mean']:.3f} ± {aggregated['auc_roc_std']:.3f}")
        print(f"   Cohen Kappa:  {aggregated['cohen_kappa_mean']:.3f} ± {aggregated['cohen_kappa_std']:.3f}")
    elif task_type == "regression":
        print(f"   MSE:                  {aggregated['mse_mean']:.3f} ± {aggregated['mse_std']:.3f}")
        print(f"   RMSE:                 {aggregated['rmse_mean']:.3f} ± {aggregated['rmse_std']:.3f}")
        print(f"   MAE:                  {aggregated['mae_mean']:.3f} ± {aggregated['mae_std']:.3f}")
        print(f"   Pearson r:            {aggregated['pearson_r_mean']:.3f} ± {aggregated['pearson_r_std']:.3f}")
        print(f"   Spearman r:           {aggregated['spearman_r_mean']:.3f} ± {aggregated['spearman_r_std']:.3f}")
        print(f"   Subject weighted MSE: {aggregated['subject_weighted_mse_mean']:.3f} ± {aggregated['subject_weighted_mse_std']:.3f}")

    # per fold breakdown
    print(f"\n📋 Per fold:")
    for fold in fold_results:
        m = fold["metrics"]
        if task_type == "classification":
            print(f"   {fold['fold_subject']} → "
                  f"sensitivity: {m['sensitivity']:.3f} | "
                  f"specificity: {m['specificity']:.3f} | "
                  f"f1: {m['f1']:.3f} | "
                  f"auc: {m['auc_roc']:.3f}")
        elif task_type == "regression":
            print(f"   {fold['fold_subject']} → "
                  f"mse: {m['mse']:.3f} | "
                  f"mae: {m['mae']:.3f} | "
                  f"pearson_r: {m['pearson_r']:.3f} | "
                  f"spearman_r: {m['spearman_r']:.3f}")

    # plots
    print(f"\n🎨 Generating plots...")

    if task_type == "classification":
        for fold in fold_results:
            plot_confusion_matrix(
                fold["y_test"],
                fold["y_pred"],
                fold["fold_subject"],
                figures_dir
            )
        plot_roc_curve(fold_results, figures_dir)

    # feature importance for tree based models
    if model_type in ("random_forest", "xgboost"):
        first_fold = fold_results[0]["fold_subject"]
        model_path = os.path.join(config["paths"]["models_dir"], f"{first_fold}.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        pipeline = FeaturePipeline(config)
        plot_feature_importance(
            model.feature_importances_,
            pipeline.get_feature_names(),
            figures_dir
        )

    plot_results_summary(aggregated, model_type, figures_dir)
    print(f"   Plots saved to {figures_dir}")


def main() -> None:
    """
    Main entry point.
    """
    args = parse_args()
    config = load_config(args.config)

    print(f"🧠 Parkinson Disease Detection Pipeline")
    print(f"   Config: {args.config}")
    print(f"   Stage:  {args.stage}")

    results = None

    if args.stage in ("all", "preprocess"):
        X, y, subject_ids = run_preprocessing_pipeline(config)

    if args.stage in ("all", "train"):
        if args.stage == "train":
            from src.features.feature_pipeline import load_feature_matrix
            X, y = load_feature_matrix(
                f"{config['paths']['segments_dir']}/features.npz"
            )
            subject_ids = None
        results = run_training_pipeline(X, y, subject_ids, config)

    if args.stage in ("all", "evaluate"):
        if results is None:
            raise ValueError("No results to evaluate — run train stage first")
        run_evaluation_pipeline(results, config)


if __name__ == "__main__":
    main()