import json
import sys
import os
from tqdm import tqdm
import asyncio
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor
import logging
from dotenv import load_dotenv

load_dotenv()

AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME") or os.getenv("S3_BUCKET", "")
AWS_DIRECTORY = os.getenv("AWS_DIRECTORY", "")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# SOURCE_INDEX_FILE = "/Work/Experiments/citation-processor-ranker/data/unarxive/processed/source_index_2024.jsonl"
PAPER_INDEX_FILE = "/Work/IISER/citation-data/SEKU-CRD-v1/paper_index_2024.jsonl"
# FULL_PAPERS_FILE = "/Work/Experiments/citation-processor-ranker/data/unarxive/processed/full_papers_2024.jsonl"


def _deep_size(obj, seen=None):
    """Recursively sum sys.getsizeof over obj and its contents (dict/list/tuple/set), so nested container overhead is counted, not just the outer shell. Shared sub-objects are counted once."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_size(k, seen)
            size += _deep_size(v, seen)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            size += _deep_size(item, seen)
    return size


def load_paper_index(paper_index_file=PAPER_INDEX_FILE,max_files=-1):
    """Load paper_index_file into a dict keyed by paper_id, and print its deep in-memory size (recursively sums sys.getsizeof over every nested value, not just the outer dict shell)."""
    paper_index = {}
    if max_files == -1:
        max_files = 200000 # large number, covers all files
    count = 0
    with open(paper_index_file, 'r') as f:
        for line in tqdm(f, desc="Loading paper index", unit="line"):
            rec = json.loads(line)
            paper_index[rec['paper_id']] = rec
            count += 1
            if count >= max_files:
                break

    size_bytes = _deep_size(paper_index)
    print(f"Loaded {len(paper_index)} papers from {paper_index_file} "
          f"({size_bytes / (1024 ** 2):.1f} MB in memory)")
    return paper_index


def list_s3_paper_ids(
    bucket_name=AWS_BUCKET_NAME,
    directory=AWS_DIRECTORY,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    region=AWS_REGION,
    s3_client=None,
):
    """
    List paper_ids already uploaded to s3://bucket_name/directory, by paginating
    the full listing and stripping the directory prefix + '.json' suffix off
    each key. Used by build_s3_index to skip files that already exist in S3.
    """
    if s3_client is None:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    prefix = directory.rstrip('/') + '/'
    existing_ids = set()
    continuation_token = None

    while True:
        list_kwargs = {'Bucket': bucket_name, 'Prefix': prefix}
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token

        response = s3_client.list_objects_v2(**list_kwargs)

        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.json'):
                existing_ids.add(key[len(prefix):-len('.json')])

        if response.get('IsTruncated'):
            continuation_token = response.get('NextContinuationToken')
        else:
            break

    return existing_ids


def get_title_and_abstract(paper):
    """
    Extract (title, abstract) from a paper record as returned by
    get_paper_from_s3 / get_papers_from_s3.

    Title comes from metadata.title. Abstract is preferred from the top-level
    'abstract' field (a {'section': 'Abstract', 'text': ...} dict in this
    corpus's schema), falling back to metadata.abstract (raw arXiv string)
    if that's missing. Returns (None, None) if paper is None (not found/failed).
    """
    if paper is None:
        return None, None

    metadata = paper.get('metadata') or {}
    title = metadata.get('title')
    if title:
        title = title.strip()

    abstract_field = paper.get('abstract')
    if isinstance(abstract_field, dict):
        abstract = abstract_field.get('text')
    elif isinstance(abstract_field, str):
        abstract = abstract_field
    else:
        abstract = metadata.get('abstract')

    if abstract:
        abstract = abstract.strip()

    return title, abstract


def get_paper_from_s3(
    paper_id,
    bucket_name=AWS_BUCKET_NAME,
    directory=AWS_DIRECTORY,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    region=AWS_REGION,
    s3_client=None,
):
    """
    Fetch and parse one paper's JSON content from S3 (paper_id is the filename,
    without the .json extension). Returns the parsed dict, or None if the key
    doesn't exist in the bucket.
    """
    if s3_client is None:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    key = f"{directory}/{paper_id}.json"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        body = response['Body'].read()
        return json.loads(body)
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code in ('NoSuchKey', '404'):
            return None
        raise


async def get_papers_from_s3(
    paper_ids,
    bucket_name=AWS_BUCKET_NAME,
    directory=AWS_DIRECTORY,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    region=AWS_REGION,
    num_workers=8,
):
    """
    Fetch multiple papers' JSON content from S3 concurrently.

    boto3's low-level client is thread-safe, so a single shared client is used
    across all fetches; each fetch runs get_paper_from_s3() in a worker thread
    (asyncio.to_thread) under a semaphore capping concurrency at num_workers.

    Args:
        paper_ids: Iterable of paper_ids to fetch (S3 filenames, no extension)
        num_workers: Max concurrent S3 GET requests in flight

    Returns:
        dict mapping paper_id -> parsed JSON dict, or None for any paper_id
        not found in S3 / failed to fetch
    """
    logger = logging.getLogger(__name__)

    s3_config = Config(max_pool_connections=num_workers, retries={'max_attempts': 3})
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=s3_config,
    )

    semaphore = asyncio.Semaphore(num_workers)

    async def fetch_one(paper_id):
        async with semaphore:
            try:
                return paper_id, await asyncio.to_thread(
                    get_paper_from_s3,
                    paper_id,
                    bucket_name=bucket_name,
                    directory=directory,
                    s3_client=s3_client,
                )
            except Exception as e:
                logger.error(f"Failed to fetch {paper_id}: {str(e)}")
                return paper_id, None

    tasks = [asyncio.ensure_future(fetch_one(pid)) for pid in paper_ids]
    results = {}
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching papers"):
        paper_id, data = await fut
        results[paper_id] = data

    return results


# def build_s3_index(
#     full_papers_file=FULL_PAPERS_FILE,
#     bucket_name=AWS_BUCKET_NAME,
#     directory=AWS_DIRECTORY,
#     num_workers=4,
#     max_files=-1,
#     access_key=AWS_ACCESS_KEY_ID,
#     secret_key=AWS_SECRET_ACCESS_KEY,
#     region=AWS_REGION,
#     skip_existing=True,
# ):
#     """
#     Upload papers from JSONL file to S3 asynchronously.

#     Args:
#         full_papers_file: Path to JSONL file containing papers
#         bucket_name: S3 bucket name
#         directory: Directory inside bucket (e.g., 'papers')
#         num_workers: Number of concurrent upload workers
#         max_files: Max files to upload (-1 for all)
#         access_key: AWS access key
#         secret_key: AWS secret key
#         region: AWS region
#         skip_existing: If True (default), list what's already in the S3
#             directory first and skip those paper_ids instead of re-uploading

#     Returns:
#         dict with upload stats: {uploaded, failed, skipped, total_size_gb}
#     """
#     # Configure S3 client with connection pooling
#     s3_config = Config(
#         max_pool_connections=num_workers,
#         retries={'max_attempts': 3}
#     )
#     s3_client = boto3.client(
#         's3',
#         aws_access_key_id=access_key,
#         aws_secret_access_key=secret_key,
#         region_name=region,
#         config=s3_config
#     )

#     logger = logging.getLogger(__name__)
#     logging.basicConfig(level=logging.INFO)

#     existing_ids = set()
#     if skip_existing:
#         print("Listing files already uploaded to S3...")
#         existing_ids = list_s3_paper_ids(
#             bucket_name=bucket_name, directory=directory, s3_client=s3_client
#         )
#         print(f"Found {len(existing_ids)} papers already in S3 -- these will be skipped")

#     uploaded = 0
#     failed = 0
#     skipped = 0
#     total_size = 0
    
#     def upload_paper(paper_record, paper_id):
#         """Upload single paper to S3. Returns (success, size_bytes)."""
#         nonlocal uploaded, failed, total_size
        
#         try:
#             paper_json = json.dumps(paper_record)
#             paper_size = len(paper_json.encode('utf-8'))
            
#             s3_key = f"{directory}/{paper_id}.json"
            
#             s3_client.put_object(
#                 Bucket=bucket_name,
#                 Key=s3_key,
#                 Body=paper_json,
#                 ContentType='application/json'
#             )
            
#             uploaded += 1
#             total_size += paper_size
#             return True, paper_size
        
#         except Exception as e:
#             logger.error(f"Failed to upload {paper_id}: {str(e)}")
#             failed += 1
#             return False, 0
    
#     # Stream file and upload with ThreadPoolExecutor (memory-safe, no batching in RAM)
#     with ThreadPoolExecutor(max_workers=num_workers) as executor:
#         futures = []
#         file_count = 0
        
#         with open(full_papers_file, 'r') as f:
#             pbar = tqdm(desc="Processing papers", unit="paper")
            
#             for line in f:
#                 if max_files != -1 and file_count >= max_files:
#                     break
                
#                 try:
#                     paper = json.loads(line)
#                 except json.JSONDecodeError:
#                     skipped += 1
#                     pbar.update(1)
#                     continue
                
#                 paper_id = paper.get('paper_id')
#                 if not paper_id:
#                     skipped += 1
#                     pbar.update(1)
#                     continue

#                 if skip_existing and paper_id in existing_ids:
#                     skipped += 1
#                     pbar.update(1)
#                     continue

#                 # Submit upload job (non-blocking)
#                 future = executor.submit(upload_paper, paper, paper_id)
#                 futures.append(future)
#                 file_count += 1
#                 pbar.update(1)
                
#                 # Keep memory low: don't queue more than 2× num_workers futures
#                 if len(futures) >= num_workers * 2:
#                     # Wait for first future to complete (ThreadPoolExecutor, not async)
#                     try:
#                         futures[0].result(timeout=60)
#                     except Exception as e:
#                         logger.error(f"Upload future failed: {str(e)}")
#                     futures = futures[1:]
            
#             # Wait for remaining uploads
#             for future in futures:
#                 try:
#                     future.result(timeout=60)
#                 except Exception as e:
#                     logger.error(f"Upload future failed: {str(e)}")
            
#             pbar.close()
    
#     total_gb = total_size / (1024 ** 3)
    
#     stats = {
#         'uploaded': uploaded,
#         'failed': failed,
#         'skipped': skipped,
#         'total_size_gb': round(total_gb, 2),
#         'total_files_processed': file_count
#     }
    
#     print("\n" + "="*60)
#     print(f"Upload Complete: {uploaded} papers uploaded ({total_gb:.2f} GB)")
#     print(f"Failed: {failed} | Skipped: {skipped} | Total processed: {file_count}")
#     print("="*60)
    
#     return stats


def count_s3_files(
    bucket_name=AWS_BUCKET_NAME,
    directory=AWS_DIRECTORY,
    access_key=AWS_ACCESS_KEY_ID,
    secret_key=AWS_SECRET_ACCESS_KEY,
    region=AWS_REGION
):
    """
    Count JSON files in S3 directory.
    
    Args:
        bucket_name: S3 bucket name
        directory: Directory inside bucket (prefix)
        access_key: AWS access key
        secret_key: AWS secret key
        region: AWS region
    
    Returns:
        dict with count stats: {total_files, total_size_gb}
    """
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    total_files = 0
    total_size = 0
    continuation_token = None
    
    try:
        while True:
            # List objects with pagination
            list_kwargs = {
                'Bucket': bucket_name,
                'Prefix': directory + '/'
            }
            
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token
            
            response = s3_client.list_objects_v2(**list_kwargs)
            
            if 'Contents' not in response:
                break
            
            for obj in response['Contents']:
                if obj['Key'].endswith('.json'):
                    total_files += 1
                    total_size += obj['Size']
            
            # Check if there are more results
            if response.get('IsTruncated'):
                continuation_token = response.get('NextContinuationToken')
            else:
                break
        
        total_gb = total_size / (1024 ** 3)
        
        stats = {
            'total_files': total_files,
            'total_size_gb': round(total_gb, 2)
        }
        
        print("\n" + "="*60)
        print(f"S3 Directory: s3://{bucket_name}/{directory}")
        print(f"JSON Files: {total_files}")
        print(f"Total Size: {total_gb:.2f} GB")
        print("="*60)
        
        return stats
    
    except Exception as e:
        print(f"Error counting files: {str(e)}")
        return {'total_files': 0, 'total_size_gb': 0}


# if __name__ == "__main__":
#     # Hardcoded parameters for S3 upload
#     stats = build_s3_index(
#         full_papers_file=FULL_PAPERS_FILE,
#         bucket_name=AWS_BUCKET_NAME,
#         directory=AWS_DIRECTORY,
#         num_workers=4,
#         max_files=-1,
#         access_key=AWS_ACCESS_KEY_ID,
#         secret_key=AWS_SECRET_ACCESS_KEY,
#         region=AWS_REGION,
#         skip_existing=True
#     )
#     print("\nFinal stats:", stats)