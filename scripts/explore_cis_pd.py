import yaml
import numpy as np
from src.data.loader import get_loader
from src.data.preprocessor import preprocess_all_subjects
from src.data.segmentor import segment_all_subjects
from src.features.feature_pipeline import FeaturePipeline


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # -------------------------------------------------------------------------
    # 1. Load config
    # -------------------------------------------------------------------------
    config = load_config("configs/cis_pd_config.yaml")
    print("✅ Config loaded")

    # -------------------------------------------------------------------------
    # 2. Load labels overview
    # -------------------------------------------------------------------------
    loader = get_loader("cis_pd", config)
    labels = loader.get_labels()

    print(f"\n✅ Labels loaded")
    print(f"   Total valid measurements: {len(labels)}")
    print(f"   Unique subjects: {labels['subject_id'].nunique()}")
    print(f"   Tremor score distribution:")
    print(f"   {labels['tremor'].value_counts().sort_index().to_dict()}")

    # -------------------------------------------------------------------------
    # 3. Load a small subset — first 5 measurements only
    # -------------------------------------------------------------------------
    print(f"\n⏳ Loading first 5 measurements...")
    loader.measurement_ids = loader.measurement_ids[:5]
    raw_data = loader.load_all_subjects()

    print(f"   Loaded {len(raw_data)} measurements")
    first_key = list(raw_data.keys())[0]
    first_df = raw_data[first_key]
    print(f"\n   Sample measurement: {first_key}")
    print(f"   Shape: {first_df.shape}")
    print(f"   Columns: {list(first_df.columns)}")
    print(f"   Duration: {len(first_df) / config['sampling']['target_fs']:.1f} seconds")
    print(f"   Tremor score: {first_df['tremor_score'].iloc[0]}")
    print(f"\n   First 3 rows:")
    print(f"   {first_df.head(3)}")

    # -------------------------------------------------------------------------
    # 4. Preprocess
    # -------------------------------------------------------------------------
    print(f"\n⏳ Preprocessing...")
    preprocessed_data = preprocess_all_subjects(raw_data, config)
    print(f"   ✅ Preprocessing complete")
    print(f"   Sample mean after zscore (should be ~0):")
    sample = preprocessed_data[first_key]
    print(f"   {sample[['wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z']].mean().round(4).to_dict()}")

    # -------------------------------------------------------------------------
    # 5. Segment
    # -------------------------------------------------------------------------
    print(f"\n⏳ Segmenting...")
    all_windows, all_labels, all_subject_ids = segment_all_subjects(
        preprocessed_data, config
    )
    print(f"   ✅ Segmentation complete")
    print(f"   Total windows: {all_windows.shape[0]}")
    print(f"   Window shape: {all_windows.shape[1]} samples × {all_windows.shape[2]} channels")
    print(f"   Labels shape: {all_labels.shape}")
    print(f"   Label range: {all_labels.min():.2f} – {all_labels.max():.2f}")

    # -------------------------------------------------------------------------
    # 6. Feature extraction
    # -------------------------------------------------------------------------
    print(f"\n⏳ Extracting features...")
    pipeline = FeaturePipeline(config)
    X = pipeline.extract(all_windows)
    print(f"   ✅ Feature extraction complete")
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   NaN count: {np.isnan(X).sum()}")


if __name__ == "__main__":
    main()