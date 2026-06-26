from utils.ss_dataset import *
import json
import time
import numpy as np
import gzip
import shutil
import os
import traceback


def human_readable_bytes(num_bytes):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PiB"


def print_progress_bar(iteration, total, prefix='', suffix='', length=40, fill='█'):
    if total <= 0:
        spinner_index = iteration % (length + 1)
        bar = fill * spinner_index + '-' * (length - spinner_index)
        print(f"{prefix} |{bar}| {suffix}", end='\r', flush=True)
        return

    percent = min(max(iteration / float(total), 0.0), 1.0)
    filled_length = int(length * percent)
    bar = fill * filled_length + '-' * (length - filled_length)
    readable_current = human_readable_bytes(iteration)
    readable_total = human_readable_bytes(total)
    print(
        f"{prefix} |{bar}| {percent*100:5.1f}% {readable_current}/{readable_total} {suffix}",
        end='\r',
        flush=True,
    )


def load_processed_files(processed_files_path):
    if not os.path.exists(processed_files_path):
        return set()
    with open(processed_files_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_processed_file(processed_files_path, fname):
    with open(processed_files_path, "a", encoding="utf-8") as f:
        f.write(fname + "\n")
        f.flush()


def extract_selective(data_dir):
    """
    This will extract all corpus ids
    and their sources in a single file. This is useful for building an index of corpus ids to sources.
    
    """
    for root,_,files in os.walk(data_dir):
        for fname in files:
            file_path = os.path.join(root,fname)
            ext = file_path[-2:]
            if ext != "gz":
                continue
            output_path = "/store/psm-citation-ranking/dataset/processed/temp.jsonl"
            output_bank = "/Work/Experiments/IISER/citation-data/extracted/" + fname + "_citations.jsonl"
            print("---Starting extraction from ",fname)
            with gzip.open(file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                MAX_DOCS = np.inf

            with open(output_path, 'r', encoding='utf-8') as f, open(output_bank, "w") as outfile:
                count = 0
                successfull_count = 0
                start_time = time.time()
                for line in f:
                    try:
                        paper_data = json.loads(line)
                        corpus_id = paper_data["corpusid"]
                        raw_content = paper_data["content"]
                        success,record = extract_from_jsonl(raw_content,corpus_id)
                        if success:
                            record.update({"source":fname})
                            outfile.write(json.dumps(record) + "\n")
                            successfull_count += 1
                        count += 1
                        if count >= MAX_DOCS:
                            break
                    except Exception as e:
                        print(f"Can not process this entry",tr.format_exc())
                end_time = time.time()
                total_time = end_time - start_time
                print(f"Processed Total = {count} entries. Successful = {successfull_count} | Time taken [{total_time} seconds] ")
                print("------------------------------------------------------------------")

def extract_all(data_dir):
    global_count = 0
    processed_files_path = os.path.join(data_dir, "processed_files.txt")
    processed_files = load_processed_files(processed_files_path)

    for root,_,files in os.walk(data_dir):
        for fname in sorted(files):
            if fname in processed_files:
                print(f"Skipping already processed file: {fname}")
                continue

            file_path = os.path.join(root,fname)
            ext = file_path[-2:]
            if ext != "gz":
                continue

            output_path = "/store/psm-citation-ranking/dataset/processed/temp.jsonl"
            print(f"---Starting extraction from {fname}...")
            try:
                with gzip.open(file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    MAX_DOCS = np.inf

                total_bytes = os.path.getsize(output_path)
                corpus_index_path = "/store/psm-citation-ranking/dataset/processed/corpus_index.jsonl"
                with open(output_path, 'r', encoding='utf-8') as f, open(corpus_index_path, "a") as index_file:
                    count = 0
                    successfull_count = 0
                    start_time = time.time()
                    print(f"Processing large file {fname} ({human_readable_bytes(total_bytes)})...")
                    last_reported = 0
                    for line in f:
                        try:
                            paper_data = json.loads(line)
                            corpus_id = paper_data["corpusid"]
                            index_file.write(json.dumps({"corpus_id": corpus_id, "source": fname}) + "\n")
                            count += 1
                            successfull_count += 1
                            global_count += 1
                        except Exception:
                            print(f"Can not process this entry: {traceback.format_exc()}")

                        try:
                            current_pos = f.tell()
                        except OSError:
                            current_pos = f.buffer.tell()

                        if current_pos - last_reported >= 4 * 1024 * 1024 or count % 1000 == 0 or current_pos >= total_bytes:
                            print_progress_bar(
                                current_pos,
                                total_bytes,
                                prefix=f"[{fname}]",
                                suffix=f"{count} records",
                                length=50,
                            )
                            last_reported = current_pos

                    print()  # newline after progress bar
                    end_time = time.time()
                    total_time = end_time - start_time
                    print(f"Finished {fname}: {count} entries, successful {successfull_count}, time {total_time:.2f}s")
                    print("------------------------------------------------------------------")

                append_processed_file(processed_files_path, fname)
                processed_files.add(fname)
            except Exception as e:
                print(f"Failed processing file {fname}: {e}\n{traceback.format_exc()}")
                print("Moving on to next file.")

    print(f"Processed Total = {global_count} entries.")


if __name__ == "__main__":
    DATA_DIR = "/store/psm-citation-ranking/dataset/s2orc-04-06-2026"
    extract_all(DATA_DIR)
    
    # # Reading few single metadata or pdf_parses file
    # file_path = "/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/20260313_115134_00043_vcx4r_0a24153e-8434-4220-9c25-922408f36db8.jsonl"
    # output_bank = "/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/archive.jsonl"
    
