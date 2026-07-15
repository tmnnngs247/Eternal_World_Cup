from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SOURCE_FILE = PROCESSED / "app_players.csv"
OUTPUT_DIR = ROOT / "assets" / "player_faces"

MAX_WORKERS = 6
LATEST_SEASON_ONLY = True


def clean_text(value: object) -> str:
    text = str(value or "").strip()

    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""

    return text


def choose_filename(row: pd.Series) -> str:
    sofifa_id = pd.to_numeric(
        pd.Series([row.get("sofifa_id")]),
        errors="coerce",
    ).iloc[0]

    if pd.notna(sofifa_id):
        return f"{int(sofifa_id)}.png"

    player_season_id = clean_text(row.get("player_season_id"))

    safe_name = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in player_season_id
    )

    return f"{safe_name or 'unknown_player'}.png"


def download_image(url: str, destination: Path) -> tuple[bool, str]:
    if destination.exists() and destination.stat().st_size > 0:
        return True, "exists"

    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = destination.with_suffix(".tmp")

    command = [
        "curl",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "--retry-delay",
        "1",
        "--connect-timeout",
        "15",
        "--max-time",
        "45",
        "--output",
        str(temporary_path),
        url,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        temporary_path.unlink(missing_ok=True)
        return False, result.stderr.strip() or f"curl exit {result.returncode}"

    if not temporary_path.exists() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        return False, "empty download"

    temporary_path.replace(destination)
    return True, "downloaded"


def prepare_downloads(players: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "image_url",
        "player_season_id",
    }

    missing_columns = required_columns.difference(players.columns)

    if missing_columns:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    work = players.copy()

    work["image_url"] = work["image_url"].map(clean_text)

    work = work[
        work["image_url"].str.startswith(("http://", "https://"))
    ].copy()

    if LATEST_SEASON_ONLY and "season_year" in work.columns:
        season_year = pd.to_numeric(
            work["season_year"],
            errors="coerce",
        )

        latest_season = season_year.max()

        work = work[
            season_year.eq(latest_season)
        ].copy()

    work["filename"] = work.apply(
        choose_filename,
        axis=1,
    )

    work = work.drop_duplicates(
        subset=["filename"],
        keep="first",
    )

    return work


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"{SOURCE_FILE} not found. Run the pipeline first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    players = pd.read_csv(
        SOURCE_FILE,
        low_memory=False,
    )

    downloads = prepare_downloads(players)

    print(f"Images to process: {len(downloads):,}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Latest season only: {LATEST_SEASON_ONLY}")

    results: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}

        for _, row in downloads.iterrows():
            destination = OUTPUT_DIR / row["filename"]

            future = executor.submit(
                download_image,
                row["image_url"],
                destination,
            )

            futures[future] = {
                "short_name": clean_text(row.get("short_name")),
                "sofifa_id": row.get("sofifa_id"),
                "season_label": clean_text(row.get("season_label")),
                "image_url": row["image_url"],
                "local_image_path": str(
                    destination.relative_to(ROOT)
                ),
            }

        total = len(futures)

        for index, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            metadata = futures[future]
            success, status = future.result()

            results.append(
                {
                    **metadata,
                    "success": success,
                    "status": status,
                }
            )

            if index % 100 == 0 or index == total:
                successful = sum(
                    bool(result["success"])
                    for result in results
                )

                print(
                    f"Processed {index:,}/{total:,} "
                    f"({successful:,} successful)"
                )

    manifest = pd.DataFrame(results)

    manifest_path = (
        PROCESSED
        / "player_face_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    successful = int(
        manifest["success"].sum()
    )

    failed = len(manifest) - successful

    print("\nFinished")
    print(f"Successful: {successful:,}")
    print(f"Failed: {failed:,}")
    print(f"Manifest: {manifest_path}")

    if failed:
        print("\nMost common failures:")

        print(
            manifest.loc[
                ~manifest["success"],
                "status",
            ]
            .value_counts()
            .head(10)
            .to_string()
        )


if __name__ == "__main__":
    main()