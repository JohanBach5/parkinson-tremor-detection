from src.data.cis_pd_loader import CISPDLoader
from src.data.daphnet_loader import DaphnetLoader


def get_loader(dataset_name: str, config: dict) -> DaphnetLoader | CISPDLoader:
    if dataset_name == "daphnet":
        raw_data_dir = config["paths"]["raw_data_dir"]
        return DaphnetLoader(raw_data_dir, config)
    elif dataset_name == "cis_pd":
        raw_data_dir = config["paths"]["raw_data_dir"]
        labels_path = config["paths"]["labels_path"]
        return CISPDLoader(raw_data_dir, labels_path, config)
    else:
        raise ValueError(
            f"Unrecognised dataset name: {dataset_name}. "
            f"Options are: 'daphnet', 'cis_pd'."
        )
