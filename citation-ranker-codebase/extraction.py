from utils.ss_dataset import *
import json
import time
import numpy as np
import gzip
import shutil
import os


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
            output_path = "/Work/Experiments/IISER/citation-data/temp.jsonl"
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
    for root,_,files in os.walk(data_dir):
        for fname in files:
            file_path = os.path.join(root,fname)
            ext = file_path[-2:]
            if ext != "gz":
                continue
            output_path = "/Work/Experiments/IISER/citation-data/temp.jsonl"
            print("---Starting extraction from ",fname)
            with gzip.open(file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                MAX_DOCS = np.inf

            corpus_index_path = "/Work/Experiments/IISER/citation-data/corpus_index.jsonl"
            with open(output_path, 'r', encoding='utf-8') as f, open(corpus_index_path, "a") as index_file:
                count = 0
                successfull_count = 0
                start_time = time.time()
                for line in f:
                    try:
                        paper_data = json.loads(line)
                        corpus_id = paper_data["corpusid"]
                        index_file.write(json.dumps({"corpus_id": corpus_id, "source": fname}) + "\n")
                        count += 1
                        successfull_count += 1
                        global_count += 1
                    except Exception as e:
                        print(f"Can not process this entry",tr.format_exc())
                end_time = time.time()
                total_time = end_time - start_time
                print(f"Processed Total = {count} entries. Successful = {successfull_count} | Time taken [{total_time} seconds] ")
                print("------------------------------------------------------------------")
    print(f"Processed Total = {global_count} entries.")


if __name__ == "__main__":
    DATA_DIR = "/Work/Experiments/IISER/s2orc"
    extract_all(DATA_DIR)
    
    # # Reading few single metadata or pdf_parses file
    # file_path = "/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/20260313_115134_00043_vcx4r_0a24153e-8434-4220-9c25-922408f36db8.jsonl"
    # output_bank = "/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/archive.jsonl"
    
