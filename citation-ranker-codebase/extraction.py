from utils.ss_dataset import *
import json
import time
import numpy as np
import gzip
import shutil
import os



if __name__ == "__main__":
    DATA_DIR = "/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/main"
    for root,_,files in os.walk(DATA_DIR):
        for fname in files:
            file_path = os.path.join(root,fname)
            output_path = "/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/temp.jsonl"
            output_bank = "/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/extracted/" + fname + "_citations.jsonl"
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


    # # Reading few single metadata or pdf_parses file
    # file_path = "/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/20260313_115134_00043_vcx4r_0a24153e-8434-4220-9c25-922408f36db8.jsonl"
    # output_bank = "/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/archive.jsonl"
    
