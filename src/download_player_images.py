from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "assets" / "player_faces"
SOURCE_FILE = PROCESSED / "app_players.csv"

MAX_WORKERS = 8
TIMEOUT_SECONDS = 20
RETRIES = 3

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/150.0 Safari/537.36"
)


def clean_text(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def choose_filename(row: pd.Series) -> str:
    sofifa_id = pd.to_numeric(
        pd.Series([row.get("sofifa_id")]),
        errors="coerce",
    ).iloc[0]

    if pd.notna(sofifa_id):
        return f"{int(sofifa_id)}.png"

    fallback = clean_text(row.get("player_season_id")) or "unknown_player"
    safe = "".join(
        char if char.isalnum() or char in {"_", "-"} else "_"
        for char in fallback
    )
    return f"{safe}.png"


def download_image(url: str, destination: Path) -> tuple[bool, str]:
    if destination.exists() and destination.stat().st_size > 0:
        return True, "exists"

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Referer": "https://sofifa.com/",
        },
    )

    last_error = ""

    for attempt in range(1, RETRIES + 1):
        try:
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                content_type = response.headers.get("Content-Type", "")
                payload = response.read()

            if not payload:
                raise ValueError("Empty response")

            if "image" not in content_type.lower():
                raise ValueError(
                    f"Unexpected content type: {content_type or 'unknown'}"
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            temp_path = destination.with_suffix(".tmp")
            temp_path.write_bytes(payload)
            temp_path.replace(destination)
            return True, "downloaded"

        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < RETRIES:
                sleep(attempt * 1.5)

    return False, last_error


def prepare_downloads(players: pd.DataFrame) -> pd.DataFrame:
    required = {"image_url", "player_season_id"}
    missing = required.difference(players.columns)

    if missing:
        raise KeyError(
            "Missing required columns in app_players.csv: "
            + ", ".join(sorted(missing))
        )

    work = players.copy()
    work["image_url"] = work["image_url"].map(clean_text)
    work = work[
        work["image_url"].str.startswith(("http://", "https://"))
    ].copy()

    if "season_year" in work.columns:
        latest_season = pd.to_numeric(
            work["season_year"],
            errors="coerce",
        ).max()

        work = work[
            pd.to_numeric(work["season_year"], errors="coerce").eq(latest_season)
        ].copy()

    work["filename"] = work.apply(choose_filename, axis=1)
    return work.drop_duplicates("filename", keep="first")


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"{SOURCE_FILE} not found. Run the data pipeline first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    players = pd.read_csv(SOURCE_FILE, low_memory=False)
    downloads = prepare_downloads(players)

    if downloads.empty:
        print("No valid image URLs found.")
        return

    print(f"Images to process: {len(downloads):,}")
    print(f"Output folder: {OUTPUT_DIR}")

    results = []

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
                "local_image_path": str(destination.relative_to(ROOT)),
            }

        total = len(futures)

        for index, future in enumerate(as_completed(futures), start=1):
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
                successes = sum(bool(item["success"]) for item in results)
                print(
                    f"Processed {index:,}/{total:,} "
                    f"({successes:,} successful)"
                )

    manifest = pd.DataFrame(results)
    manifest_path = PROCESSED / "player_face_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    successes = int(manifest["success"].sum())
    failures = len(manifest) - successes

    print("\nFinished")
    print(f"Successful: {successes:,}")
    print(f"Failed: {failures:,}")
    print(f"Manifest: {manifest_path}")

    if failures:
        print("\nMost common failure statuses:")
        print(
            manifest.loc[~manifest["success"], "status"]
            .value_counts()
            .head(10)
            .to_string()
        )


if __name__ == "__main__":
    main()
