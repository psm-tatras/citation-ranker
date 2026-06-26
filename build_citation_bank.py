import gzip
import shutil
import os
from src.utils.ss_dataset import *
import json
import time
from tqdm import tqdm

if __name__ == "__main__":
    MAX_GZS = -1  # Set to -1 to process all .gz files, or specify a positive integer to limit the number of files processed
    RATIO_THRESHOLD = 0.8
    MIN_CITATIONS = 5
    CORPUS_INDEX_PATH = "/store/psm-citation-ranking/dataset/processed/s2orc-citation-ranker/corpus_index.jsonl"
    DATA_DIR = "/store/psm-citation-ranking/dataset/s2orc-04-06-2026"
    TEMP_PATH = "/store/psm-citation-ranking/dataset/processed/temp-citation-extraction.jsonl"
    OUTPUT_BANK = f"/store/psm-citation-ranking/dataset/processed/citation-bank/23062026-{MAX_GZS}-80p.jsonl"
    GZ_DICT_PATH = f"/store/psm-citation-ranking/dataset/processed/citation-bank/23062026-{MAX_GZS}-80p-gz_dict.jsonl"
    PROCESSED_LOG_PATH = "/store/psm-citation-ranking/dataset/processed/citation-bank/processed_gz.txt"
    corpus_index = {}
    print("Loading corpus index in memory")
    with open(CORPUS_INDEX_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading corpus index", unit=" lines"):
            entry = json.loads(line)
            corpus_index[entry["corpus_id"]] = entry["source"]
    print(f"Loaded {len(corpus_index)} entries")

    if os.path.exists(PROCESSED_LOG_PATH):
        with open(PROCESSED_LOG_PATH, 'r') as f:
            processed = {line.strip() for line in f if line.strip()}
    else:
        processed = set()
    print(f"Skipping {len(processed)} already-processed gz files")

    gz_files = sorted(
        os.path.join(root, fname)
        for root, _, files in os.walk(DATA_DIR)
        for fname in files
        if fname.endswith(".gz") and fname not in processed
    )[:MAX_GZS]

    global_count = 0
    gz_dict = {}
    with open(OUTPUT_BANK, "w") as outfile:
        for i, file_path in enumerate(gz_files):
            fname = os.path.basename(file_path)
            print(f"---Extracting {fname}...")
            try:
                with gzip.open(file_path, 'rb') as f_in, open(TEMP_PATH, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                print(f"Error extracting {fname}: {e}")
                continue
            print(f"---Scanning {fname} for matched citations")

            count = 0
            start_time = time.time()
            with open(TEMP_PATH, 'r', encoding='utf-8') as f:
                pbar = tqdm(f, desc=f"Scanning {fname}", unit=" lines")
                for line in pbar:
                    try:
                        paper_data = json.loads(line)
                        corpus_id = paper_data["corpusid"]
                        citations = eval(paper_data['content']['annotations']['bibentry'])
                        total_citations = len(citations)
                        found = 0
                        all_citation_ids = []
                        for item in citations:
                            attributes = item['attributes']
                            if not 'matched_paper_id' in attributes.keys():
                                continue
                            matched_paper_id = attributes['matched_paper_id']
                            if matched_paper_id is not None:
                                matched_paper_id = int(matched_paper_id)
                                if matched_paper_id in corpus_index:
                                    all_citation_ids.append({
                                        "matched_paper_id": matched_paper_id,
                                        "source_gz": corpus_index[matched_paper_id]
                                    })
                                    found += 1
                        ratio = found / total_citations if total_citations > 0 else 0
                        if ratio >= RATIO_THRESHOLD and total_citations >= MIN_CITATIONS:
                            outfile.write(json.dumps({
                                "corpus_id": corpus_id,
                                "citations": all_citation_ids,
                                "ratio": ratio,
                                "total_citations": total_citations,
                                "source_gz": fname
                            }) + "\n")
                            gz_dict.setdefault(fname, []).append(corpus_id)
                            for citation in all_citation_ids:
                                gz_dict.setdefault(citation["source_gz"], []).append(corpus_id)
                            count += 1
                            pbar.set_postfix_str(f"ID:{corpus_id} | {found}/{total_citations} | {ratio:.2%} | Extracted:{count} | {time.time() - start_time:.1f}s")
                    except Exception as e:
                       pass
            global_count += count
            remaining = len(gz_files) - i - 1
            tqdm.write(f"---Finished {fname}. Extracted: {count} | Global: {global_count} | Remaining: {remaining}")
            with open(PROCESSED_LOG_PATH, 'a') as log:
                log.write(fname + "\n")

    print(f"Writing gz_dict ({len(gz_dict)} entries) to {GZ_DICT_PATH}")
    with open(GZ_DICT_PATH, "w") as gz_out:
        for gz_name, corpus_ids in gz_dict.items():
            gz_out.write(json.dumps({"gz": gz_name, "corpus_ids": corpus_ids}) + "\n")
    print("Done.")

    del corpus_index, gz_dict, gz_files