import argparse
import os
import time
import requests
from pathlib import Path
from tqdm import tqdm


API_KEY = "s2k-w3I59p4P2DDRQUHRoYFBYnBFXKikO6drraIaZvJ3"
BASE_URL = "https://api.semanticscholar.org/datasets/v1/release"
MAX_RETRIES = 5


def fetch_dataset_files(release_id: str, dataset_name: str) -> list[str]:
    headers = {"x-api-key": API_KEY}
    url = f"{BASE_URL}/{release_id}/dataset/{dataset_name}"

    for attempt in range(MAX_RETRIES):
        response = requests.get(url, headers=headers)
        if response.status_code == 429:
            wait = 2 ** attempt
            print(f"Rate limited. Retrying in {wait}s... (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        response.raise_for_status()
        data = response.json()
        if "files" not in data:
            raise ValueError(f"No files found in response: {data}")
        return data["files"]

    raise RuntimeError(f"Failed to fetch dataset after {MAX_RETRIES} attempts")


def download_file(url: str, download_dir: str, index: int, total: int):
    filename = url.split("?")[0].split("/")[-1]
    dest = os.path.join(download_dir, filename)

    if os.path.exists(dest):
        print(f"[{index}/{total}] Skipping {filename} (already exists)")
        return

    for attempt in range(MAX_RETRIES):
        try:
            label = f"[{index}/{total}]"
            print(f"{label} Downloading {filename}..." + (f" (retry {attempt})" if attempt else ""))
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code == 429:
                    wait = 2 ** attempt
                    print(f"{label} Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                total_bytes = int(r.headers.get("Content-Length", 0))
                with open(dest, "wb") as f, tqdm(
                    total=total_bytes or None,
                    unit="MB",
                    unit_scale=1 / (1024 * 1024),
                    unit_divisor=1,
                    desc=f"{label} {filename}",
                    dynamic_ncols=True,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        bar.update(len(chunk))
            print(f"{label} Done -> {dest}")
            return
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"[{index}/{total}] Error: {e}. Retrying in {wait}s...")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Download Semantic Scholar dataset files")
    parser.add_argument("--release", default="latest", help="Release ID (e.g. 2023-10-31) or 'latest'")
    parser.add_argument("--dataset", default="papers", help="Dataset name (e.g. papers, s2orc, citations)")
    parser.add_argument("--output-dir", default="./data", help="Directory to save downloaded files")
    args = parser.parse_args()

    release_id = args.release
    if release_id == "latest":
        release_id = requests.get(f"{BASE_URL}/latest").json()["release_id"]
        print(f"Latest release: {release_id}")

    print(f"Fetching file list for dataset '{args.dataset}' from release '{release_id}'...")
    files = fetch_dataset_files(release_id, args.dataset)
    total = len(files)
    print(f"Found {total} files.")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(files, 1):
        download_file(url, args.output_dir, i, total)

    print("All downloads complete.")


if __name__ == "__main__":
    main()
