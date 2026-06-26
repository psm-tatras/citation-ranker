import argparse
import asyncio
import os
import re
import time
import requests
import aiohttp
from pathlib import Path
from tqdm import tqdm


API_KEY = "s2k-w3I59p4P2DDRQUHRoYFBYnBFXKikO6drraIaZvJ3"
BASE_URL = "https://api.semanticscholar.org/datasets/v1/release"
MAX_RETRIES = 5
GZ_FILENAME_REGEX = re.compile(
    r"^\d{8}_\d{6}_\d{5}_[a-z0-9]{5}_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.gz$"
)


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


async def download_file(session: aiohttp.ClientSession, url: str, download_dir: str, temp_dir: str, index: int, total: int) -> bool:
    filename = url.split("?")[0].split("/")[-1]
    if not GZ_FILENAME_REGEX.match(filename):
        raise ValueError(f"Unexpected filename from URL, expected dataset .gz file: {filename}")

    dest = os.path.join(download_dir, filename)
    temp_dest = os.path.join(temp_dir, filename)

    if os.path.exists(dest):
        print(f"[{index}/{total}] Skipping {filename} (already downloaded in {download_dir})")
        return True

    for attempt in range(MAX_RETRIES):
        label = f"[{index}/{total}]"
        print(f"{label} Downloading {filename}..." + (f" (retry {attempt})" if attempt else ""))
        try:
            timeout = aiohttp.ClientTimeout(total=600)
            async with session.get(url, timeout=timeout) as r:
                if r.status == 429:
                    wait = 2 ** attempt
                    print(f"{label} Rate limited. Waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                total_bytes = int(r.headers.get("Content-Length", 0))
                with open(temp_dest, "wb") as f, tqdm(
                    total=total_bytes or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"{label} {filename}",
                    dynamic_ncols=True,
                ) as bar:
                    async for chunk in r.content.iter_chunked(1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bar.update(len(chunk))
            os.replace(temp_dest, dest)
            print(f"{label} Done -> {dest}")
            return True
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            if os.path.exists(temp_dest):
                os.remove(temp_dest)
            if attempt == MAX_RETRIES - 1:
                print(f"{label} Failed to download {filename} after {MAX_RETRIES} attempts: {e}. Skipping.")
                return False
            wait = 2 ** attempt
            print(f"{label} Error: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
        except Exception as e:
            if os.path.exists(temp_dest):
                os.remove(temp_dest)
            print(f"{label} Unexpected error downloading {filename}: {e}. Skipping.")
            return False
    print(f"[{index}/{total}] Failed to download {filename} after {MAX_RETRIES} attempts; skipping.")
    return False


async def download_all_files(urls: list[str], download_dir: str, concurrency: int):
    temp_dir = os.path.join(download_dir, ".tmp_downloads")
    os.makedirs(temp_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def sem_download(url: str, index: int):
            async with semaphore:
                success = await download_file(session, url, download_dir, temp_dir, index, len(urls))
                if not success:
                    print(f"[{index}/{len(urls)}] Skipped corrupted or unreachable file: {url}")

        await asyncio.gather(*(sem_download(url, i) for i, url in enumerate(urls, start=1)))


def main():
    parser = argparse.ArgumentParser(description="Download Semantic Scholar dataset files")
    parser.add_argument("--release", default="latest", help="Release ID (e.g. 2023-10-31) or 'latest'")
    parser.add_argument("--dataset", default="papers", help="Dataset name (e.g. papers, s2orc, citations)")
    parser.add_argument("--output-dir", default="./data", help="Directory to save downloaded files")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of files to download concurrently")
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

    asyncio.run(download_all_files(files, args.output_dir, args.concurrency))

    print("All downloads complete.")


if __name__ == "__main__":
    main()
