"""Download the Hass Avocado Ripening Photographic Dataset (~400 MB).

Source: Mendeley Data, dataset 3xd9n945v8 (version 1)
        https://data.mendeley.com/datasets/3xd9n945v8/1

Usage:
    python -m avocado_ai.data.download [--output data/raw/hass_avocado_dataset.zip]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

DATASET_URL = "https://data.mendeley.com/public-api/zip/3xd9n945v8/download/1"
DEFAULT_OUTPUT = Path("data/raw/hass_avocado_dataset.zip")
CHUNK_SIZE = 1 << 20  # 1 MiB


def download(url: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        print(f"[skip] {destination} already exists (use --force to re-download)")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        received = 0
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                handle.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r  {received / 2**20:,.0f} / {total / 2**20:,.0f} MiB", end="")
        print()

    partial.replace(destination)  # only expose the file once it is complete
    print(f"[done] Saved to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Hass avocado dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args()
    download(DATASET_URL, args.output, force=args.force)


if __name__ == "__main__":
    main()
