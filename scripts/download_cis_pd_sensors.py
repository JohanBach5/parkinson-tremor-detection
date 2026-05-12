import argparse
import tarfile
from pathlib import Path

import synapseclient


SENSOR_TARBALL_SYN_ID = "syn26435215"


def download_and_extract(output_dir: Path, keep_tarball: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    syn = synapseclient.Synapse()
    syn.login()

    # Size check before committing to download
    entity_meta = syn.get(SENSOR_TARBALL_SYN_ID, downloadFile=False)
    size_gb = entity_meta._file_handle["contentSize"] / 1e9
    print(f"Tarball size: {size_gb:.2f} GB")

    print(f"Downloading {SENSOR_TARBALL_SYN_ID} to {output_dir} ...")
    entity = syn.get(SENSOR_TARBALL_SYN_ID, downloadLocation=str(output_dir))
    tarball_path = Path(entity.path)
    print(f"Downloaded to: {tarball_path}")

    print(f"Extracting {tarball_path.name} ...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        members = tar.getmembers()
        print(f"Archive contains {len(members)} members")
        tar.extractall(path=output_dir)
    print(f"Extracted to: {output_dir}")

    if not keep_tarball:
        tarball_path.unlink()
        print(f"Removed tarball: {tarball_path}")

    print("\nTop-level contents of extraction directory:")
    for p in sorted(output_dir.iterdir())[:20]:
        marker = "/" if p.is_dir() else ""
        print(f"  {p.name}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cis_pd/sensor_data"),
        help="Where to extract the sensor data (default: data/cis_pd/sensor_data)",
    )
    parser.add_argument(
        "--keep-tarball",
        action="store_true",
        help="Keep the downloaded tarball after extraction",
    )
    args = parser.parse_args()
    download_and_extract(args.output_dir, keep_tarball=args.keep_tarball)


if __name__ == "__main__":
    main()