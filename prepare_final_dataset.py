import gzip
import os
from src.utils.ss_dataset import *
import json
import time
from tqdm import tqdm

if __name__== "__main__":
    CORPUSID_MAIN = '/store/psm-citation-ranking/dataset/processed/s2orc-citation-ranker-240626/23062026--1-80p.csv'
    CORPUSID_CITATIONS = '/store/psm-citation-ranking/dataset/processed/s2orc-citation-ranker-240626/23062026--1-80p.jsonl'
    COLLECTION_THRESHOLD = 1
    OUTPUT_SOURCE_CORPUSIDS = '/store/psm-citation-ranking/dataset/processed/s2orc-citation-ranker-240626/source_list_corpusids.jsonl'
    MAIN_GZ_PATH = "/store/psm-citation-ranking/dataset/s2orc-04-06-2026"
    FINAL_DATASET = "/store/psm-citation-ranking/dataset/processed/s2orc-citation-ranker-240626/final_dataset.jsonl"

     #-----PHASE 0-----------#
     # create CSV from JSONL file for final dataset
    export_citation_bank_csv(CORPUSID_CITATIONS, CORPUSID_MAIN)
    
    #-----PHASE 1-----------#
    # first collect target papers, later we will collect their citations
    # colecting only those papers where the ratio of citations is above a COLLECTION_THRESHOLD
    target_corpusids = []
    source_dict = {}
    with open(CORPUSID_MAIN, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            corpusid = row["corpusid"]
            row["ratio"] = float(row["ratio"])
            if row["ratio"] >= COLLECTION_THRESHOLD:
                source_gz = row['source_gz']
                target_corpusids.append(corpusid)
                if source_gz not in source_dict: # this gz not added before
                    source_dict[source_gz] = [corpusid]
                else:
                    all_ids = source_dict[source_gz]
                    if corpusid not in all_ids: # this corpusid is not added before
                        source_dict[source_gz].append(corpusid)
    target_corpusids = set(target_corpusids)
    print(f"Total corpusids: {len(target_corpusids)}")
    print(f"Total {len(source_dict)} GZs collected")

    #-----PHASE 2-----------#
    # now, we will collect the citations for these target corpusids
    # for this we consult CORPUSID_CITATIONS which is a JSONL file where each line is a JSON object 
    cited_corpusids = []
    with open(CORPUSID_CITATIONS, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            corpusid = str(data['corpus_id'])
            if corpusid in target_corpusids: # this is a set O(1) lookup
                citations = data['citations'] # this is a list of dicts, each dict has two keys:'matched_paper_id' and 'source_gz'
                for a_citation in citations:
                    cited_id = a_citation['matched_paper_id']
                    cited_corpusids.append(cited_id)
                    source_gz = a_citation['source_gz']
                    if source_gz not in source_dict: # this gz not added before
                        source_dict[source_gz] = [cited_id]
                    else:
                        all_ids = source_dict[source_gz]
                        if cited_id not in all_ids: # this corpusid is not added before
                            source_dict[source_gz].append(cited_id)
                # print(f"CorpusID: {corpusid}, Citations: {len(citations)}")
    cited_corpusids = set(cited_corpusids)
    print(f"Total corpusids: {len(cited_corpusids)}")
    print(f"Total {len(source_dict)} GZs collected")

    # verify the volume of corpus ids to process
    v_cids = []
    v_sgzs = 0
    for k in source_dict.keys():
        cids = source_dict[k]
        v_cids.extend(cids)
    u_cids = len(list(set(v_cids)))
    print(f"Total {len(v_cids)} to be loaded in Phase 3, Unique {u_cids}")


    with open(OUTPUT_SOURCE_CORPUSIDS, "w") as gz_out:
        for gz_name, corpus_ids in source_dict.items():
            gz_out.write(json.dumps({"source_gz": gz_name, "corpus_ids": corpus_ids}) + "\n")



    #-----PHASE 3-----------#
    # now start reading the OUTPUT_SOURCE_CORPUSIDS
    # for each gz, stream-read directly (no temp extraction to disk)
    # if any record found, copy in new jsonl
    with open(OUTPUT_SOURCE_CORPUSIDS, 'r', encoding='utf-8') as f:
        gz_entries = [json.loads(line) for line in f]

    print(f"Total {len(gz_entries)} entries for final dataset")
    total_collected = 0
    with open(FINAL_DATASET, "w") as outfile:
        # for entry in tqdm(gz_entries, desc="Processing GZs", unit="gz"):
        for source_gz in tqdm(source_dict.keys()):
            # source_gz = entry["source_gz"]
            entry = [str(x) for x in source_dict[source_gz]]
            target_ids = set(entry) # set for O(1) lookup
            file_path = os.path.join(MAIN_GZ_PATH, source_gz)
            try:
                collected = 0
                total_targets = len(target_ids)
                with gzip.open(file_path, 'rt', encoding='utf-8') as ftemp:
                    pbar = tqdm(ftemp, desc=f"Scanning {source_gz}", unit=" lines", leave=False)
                    for p_line in pbar:
                        try:
                            paper_data = json.loads(p_line)
                            paper_id = str(paper_data["corpusid"]) # comes in integer
                            if paper_id in target_ids: # yes, we have this in our target set, need to copy contents
                                paper_type = "main" if paper_id in target_corpusids else "cited"
                                paper_data["paper_type"] = paper_type
                                outfile.write(json.dumps(paper_data) + "\n")
                                collected += 1
                                pbar.set_postfix_str(f"collected:{collected}/{total_targets}")
                            if collected >= total_targets:
                                tqdm.write(f"All IDs collected : Total {collected}, not checking the rest")
                                break
                        except Exception as e:
                            tqdm.write(f"Error processing a record in {source_gz}: {e}")
                            continue
                    # processing one GZ complete
                    total_collected += collected
                    tqdm.write(f"All IDs collected so far : Total {total_collected}")
            except Exception as e:
                tqdm.write(f"Error extracting {source_gz}: {e}")
                continue