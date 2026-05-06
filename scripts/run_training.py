import yaml
import numpy as np

from src.data.loader import get_loader
from src.data.preprocessor import preprocess_all_subjects
from src.data.segmentor import segment_all_subjects
from src.features.feature_pipeline import FeaturePipeline
from src.training.trainer import LOSOTrainer, aggregate_fold_results


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # -------------------------------------------------------------------------
    # 1. Load config
    # -------------------------------------------------------------------------
    config = load_config("configs/daphnet_config.yaml")
    print("✅ Config loaded")

    # -------------------------------------------------------------------------
    # 2. Load and preprocess data
    # -------------------------------------------------------------------------
    loader = get_loader("daphnet", config)
    raw_data = loader.load_all_subjects()
    preprocessed_data = preprocess_all_subjects(raw_data, config)
    print("✅ Data loaded and preprocessed")

    # -------------------------------------------------------------------------
    # 3. Segment
    # -------------------------------------------------------------------------
    all_windows, all_labels, all_subject_ids = segment_all_subjects(
        preprocessed_data, config
    )
    print(f"✅ Segmentation complete — {all_windows.shape[0]} windows")

    # -------------------------------------------------------------------------
    # 4. Extract features
    # -------------------------------------------------------------------------
    pipeline = FeaturePipeline(config)
    X = pipeline.extract(all_windows)
    y = all_labels
    print(f"✅ Features extracted — shape {X.shape}")

    # -------------------------------------------------------------------------
    # 5. Train
    # -------------------------------------------------------------------------
    print(f"\n🚀 Starting LOSO training with {config['training']['model_type']}...")
    print(f"   This will train {len(np.unique(all_subject_ids))} folds\n")

    trainer = LOSOTrainer(config)
    results = trainer.run(X, y, all_subject_ids)

    # -------------------------------------------------------------------------
    # 6. Aggregate and print results
    # -------------------------------------------------------------------------
    aggregated = aggregate_fold_results(results["fold_results"])

    print(f"\n✅ Training complete — {len(results['fold_results'])} folds")
    print(f"   Model: {results['model_type']}")
    print(f"\n📊 Results:")
    print(f"   Sensitivity:   {aggregated['sensitivity_mean']:.3f} ± {aggregated['sensitivity_std']:.3f}")
    print(f"   Specificity:   {aggregated['specificity_mean']:.3f} ± {aggregated['specificity_std']:.3f}")
    print(f"   F1 Score:      {aggregated['f1_mean']:.3f} ± {aggregated['f1_std']:.3f}")
    print(f"   AUC-ROC:       {aggregated['auc_roc_mean']:.3f} ± {aggregated['auc_roc_std']:.3f}")
    print(f"   Cohen Kappa:   {aggregated['cohen_kappa_mean']:.3f} ± {aggregated['cohen_kappa_std']:.3f}")

    # -------------------------------------------------------------------------
    # 7. Per fold breakdown
    # -------------------------------------------------------------------------
    print(f"\n📋 Per fold results:")
    for fold in results["fold_results"]:
        m = fold["metrics"]
        print(f"   {fold['fold_subject']} → "
              f"sensitivity: {m['sensitivity']:.3f} | "
              f"specificity: {m['specificity']:.3f} | "
              f"f1: {m['f1']:.3f} | "
              f"auc: {m['auc_roc']:.3f}")


if __name__ == "__main__":
    main()