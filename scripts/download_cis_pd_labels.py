import argparse
import shutil
from pathlib import Path

import synapseclient


FILES = {
    "labels": {
        "syn21291578": "CIS-PD_Training_Data_IDs_Labels.csv",
        "syn21291582": "CIS-PD_Test_Data_IDs_Labels.csv",
    },
    "clinical": {
        "syn21291808": "CIS-PD_Demographics.csv",
        "syn21291805": "CIS-PD_UPDRS_Part3.csv",
    },
}


def download_files(output_root: Path) -> None:
    syn = synapseclient.Synapse()
    syn.login()

    for subdir, mapping in FILES.items():
        target_dir = output_root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        for syn_id, filename in mapping.items():
            target_path = target_dir / filename
            if target_path.exists():
                print(f"[skip] {target_path} already exists")
                continue

            print(f"[download] {syn_id} -> {target_path}")
            entity = syn.get(syn_id, downloadLocation=str(target_dir))
            # syn.get may return a different filename than we want; normalize it
            downloaded = Path(entity.path)
            if downloaded.name != filename:
                shutil.move(str(downloaded), str(target_path))

    print("\nDone. Directory tree:")
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            size_kb = path.stat().st_size / 1024
            print(f"  {path}  ({size_kb:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cis_pd"),
        help="Root directory for downloaded files (default: data/cis_pd)",
    )
    args = parser.parse_args()
    download_files(args.output_root)


if __name__ == "__main__":
    main()