"""Download the original Scania CSV files from the UCI repository."""

from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "data"
BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/00421"
FILES = ("aps_failure_training_set.csv", "aps_failure_test_set.csv")


def main():
    DESTINATION.mkdir(exist_ok=True)
    for name in FILES:
        destination = DESTINATION / name
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Already present: {destination}")
            continue
        print(f"Downloading {name}...")
        urlretrieve(f"{BASE}/{name}", destination)
        print(f"Saved {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

