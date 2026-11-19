"""Turn the downloaded dataset ZIP into a training-ready folder.

Output layout:
    data/processed/
    ├── metadata.csv          # one row per photo, with the regression target
    └── images/
        ├── 1_underripe/*.jpg
        ├── 2_breaking/*.jpg
        ├── 3_ripe_first_stage/*.jpg
        ├── 4_ripe_second_stage/*.jpg
        └── 5_overripe/*.jpg

Usage:
    python -m avocado_ai.data.prepare [--zip data/raw/hass_avocado_dataset.zip]
                                      [--output data/processed]
"""

from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from avocado_ai.data.metadata import add_regression_target, load_raw_metadata
from avocado_ai.domain import RipenessStage

DEFAULT_ZIP = Path("data/raw/hass_avocado_dataset.zip")
DEFAULT_OUTPUT = Path("data/processed")
SPREADSHEET_NAME = "Avocado Ripening Dataset.xlsx"


def stage_folder(stage: int) -> str:
    return f"{stage}_{RipenessStage(stage).name.lower()}"


def prepare(zip_path: Path, output_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Run `python -m avocado_ai.data.download` first."
        )

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()

        spreadsheet = next((m for m in members if m.endswith(SPREADSHEET_NAME)), None)
        if spreadsheet is None:
            raise FileNotFoundError(f"{SPREADSHEET_NAME} not found inside {zip_path}")
        metadata = add_regression_target(load_raw_metadata(io.BytesIO(archive.read(spreadsheet))))

        # File names in the spreadsheet have no extension: index photos by stem.
        photos = {
            PurePosixPath(m).stem: m for m in members if m.lower().endswith((".jpg", ".jpeg"))
        }

        image_paths: list[str | None] = []
        copied = skipped = 0
        for row in metadata.itertuples():
            member = photos.get(row.file_name)
            if member is None:
                image_paths.append(None)
                continue
            relative = Path("images") / stage_folder(row.stage) / PurePosixPath(member).name
            target = output_dir / relative
            if target.exists():
                skipped += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                copied += 1
            image_paths.append(relative.as_posix())

    metadata["image_path"] = image_paths
    missing = int(metadata["image_path"].isna().sum())
    metadata = metadata.dropna(subset=["image_path"])

    metadata_path = output_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    print(f"Photos copied: {copied:,} | already present: {skipped:,} | missing: {missing:,}")
    print(f"Fruits: {metadata['group_id'].nunique():,} | rows: {len(metadata):,}")
    print("Stage distribution:")
    for stage, count in metadata["stage"].value_counts().sort_index().items():
        print(f"  {stage_folder(stage):<22} {count:>6,}")
    print(f"Metadata written to {metadata_path}")
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Hass avocado dataset.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, dest="zip_path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.zip_path, args.output)


if __name__ == "__main__":
    main()
