import asyncio
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

project_root = Path.cwd()
if not (project_root / "utils").is_dir():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env", override=True)

from utils.seku_crd_corpus_utils import *
from prediction.predict_ner import *
from prediction.predict_ner_v2 import *
from prediction.predict_ner_v3 import UnifiedQwenPipeline,postprocess_json_with_types
from utils.embedding_utils import *
from utils.prompts import LABELS
import random

TEST_VOLUME = int(os.environ["TEST_VOLUME"]) # for how many papers we are computing citation relevance
LOCAL_MODEL_DIR = os.environ["NER_MODEL_LOCAL_DIR"]
WANDB_PROJECT = os.environ["WANDB_PROJECT"]
WANDB_ARTIFACT = os.environ["WANDB_ADAPTER_ARTIFACT_NAME"]
MAX_REFERENCES = int(os.environ["MAX_REFERENCES"]) # Prediction is in GPU, to quickly test take only few references
RESULT_DIR = os.environ["RESULT_DIR"]
MAX_MAIN_PAPERS = int(os.environ["MAX_MAIN_PAPERS"])
PROCESSED_LOG_PATH = f"{RESULT_DIR}/processed_main.jsonl"
FAILED_LOG_PATH = f"{RESULT_DIR}/failed_main.jsonl"


def _load_processed_ids(path):
    """{main_paper_id: report_path} lines already written by a prior run."""
    ids = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.update(json.loads(line).keys())
    return ids


def _append_jsonl(path, main_paper_id, value):
    with open(path, "a") as f:
        f.write(json.dumps({main_paper_id: value}) + "\n")






# but for a list of {paper_id,text} extracts NEs async
# uses the class-based model's real batched generation (apredict) instead of one generate() call per paper
async def get_entities_from_papers(items,ner_model,batch_size=8,max_new_tokens=2048):
    text_by_id = {item["paper_id"]:item["text"] for item in items}
    ft_items = [{"paper_id":item["paper_id"],"text_to_extract":item["text"]} for item in items]
    predictions = await ner_model.apredict(ft_items,batch_size=batch_size,max_new_tokens=max_new_tokens)

    result_aug_map = {}
    all_windows_map = {}
    for pred in predictions:
        paper_id = pred["paper_id"]
        entities = pred["entities"]
        if not entities: # apredict returns {} for paper_ids that failed to parse
            result_aug_map[paper_id] = {}
            all_windows_map[paper_id] = {}
            continue
        text = text_by_id[paper_id]
        result_aug = augment_context_by_label(entities,text,window=0) # grouped entities + context
        all_windows = {}
        for r in result_aug:
            w_text = ""
            for rt in result_aug[r]:
                w_text += rt["text"]
            all_windows[r] = w_text
        result_aug_map[paper_id] = result_aug
        all_windows_map[paper_id] = all_windows
    return result_aug_map,all_windows_map


async def generate_relevance_report(main_paper_id):
    """Extract NEs from main_paper_id and its references, score reference
    relevance against the main paper, and write the report to
    RESULT_DIR/<main_paper_id>.json. Returns True on success, False if
    anything failed along the way. Already-processed ids (tracked in
    PROCESSED_LOG_PATH) are skipped so a rerun resumes rather than
    redoing work; failures are logged to FAILED_LOG_PATH."""
    if main_paper_id in processed_ids:
        print(f"{main_paper_id} already processed, skipping")
        return True
    try:
        main_data = paper_index[main_paper_id] # data from paper index file
        title = main_data["title"]
        # print(f'Title: {title}')
        main_paper_data = get_paper_from_s3(main_paper_id)
        print("Retrieved main paper data from S3...Extracting titles (not needed) and abstract")
        main_title,main_abstract = get_title_and_abstract(main_paper_data)
        print("Extracted title and abstract from raw data...Finding Named Entities")

        # extract main paper data
        abstracts = [{"paper_id":main_paper_id,"text":main_abstract}]
        result_aug_map,all_windows_map = await get_entities_from_papers(abstracts,ner_model)
        print("Entity Extraction Findinings complete")

        # gather reference papers
        in_corpus_refs_ids = main_data["in_corpus_references"] # list of IDs of [C]
        nb_all_refs_r = main_data["total_bib_entries"]
        # we need to check if these ids truly exists in our Full Paper Data
        # our full paper data is in S3 😊, so we will get it from there.

        # for quick computation, we will take few references, not ALL
        in_corpus_refs_ids = in_corpus_refs_ids[:min(len(in_corpus_refs_ids),MAX_REFERENCES)]
        print(f"We will extract entities from {len(in_corpus_refs_ids)} citations")
        results = await get_papers_from_s3(in_corpus_refs_ids)
        found_full_data = 0

        # print(f"Checking refernces for Main paper {main_paper_id}\n{main_title}")
        # print("="*5)
        # ref_ne_contexts = {}
        list_of_texts = []
        references_nes = {}
        for paper_id, data in results.items():
            if data is not None:
                ref_title,ref_abstract = get_title_and_abstract(data)
                list_of_texts.append({"paper_id":paper_id,"text":ref_abstract})
                references_nes[paper_id] = {"abstract":ref_abstract,"title":ref_title}
                found_full_data += 1
            else:
                print(f"{paper_id} Not found ☹️")
        full_data_ratio = found_full_data / nb_all_refs_r
        # print(f"{full_data_ratio*100:0.2f}% references have full data")
        # print(f"Collected {len(list_of_texts)} for NE extraction")

        # extract entities from reference papers
        ref_result_aug_map,ref_all_windows_map = await get_entities_from_papers(list_of_texts,ner_model,batch_size=6,max_new_tokens=2048)
        # build a dict for next steps
        for paper_id in ref_all_windows_map:
            data = ref_all_windows_map[paper_id]
            references_nes[paper_id].update({"nes":data})

        # create pairs for embedding computation
        all_pairs = {}
        for paper_id in all_windows_map: # main paper data
            # print(paper_id)
            # print("="*5)
            main_nes = all_windows_map[paper_id] # main paper entities
            for d in main_nes:
                d_dict = {}
                for ref_id in references_nes: # for each ref paper
                    if "nes" in references_nes[ref_id]:
                        ref_nes = references_nes[ref_id]["nes"] # this is a dict of entity labels
                        if d in ref_nes:
                            d_dict[ref_id] = ref_nes[d]
                all_pairs[d] = {"base_text":main_nes[d],"ref_texts":d_dict}
        # print("All pairs collected")

        # report generation
        report = {}
        for lbl in all_pairs:
            base_text = all_pairs[lbl]["base_text"]
            ref_texts = all_pairs[lbl]["ref_texts"]
            rel_score_lbl = reference_relevance_computation(base_text,ref_texts,embedding_model)
            # print(lbl)
            # print("=="*5)
            for pid in rel_score_lbl:
                # print(f"paper {pid} relevance {rel_score_lbl[pid]:0.2f}")
                if pid not in report:
                    report[pid] = {lbl:rel_score_lbl[pid]}
                else:
                    report[pid].update({lbl:rel_score_lbl[pid]})
        # showing the report
        # for ref_id in report:
        #     # print(ref_id)
        #     # print("--"*5)
        #     for lbl in report[ref_id]:
        #         print(f"{lbl}: {report[ref_id][lbl]}")
        #     print("=="*5)

        extended_report = {}
        main_nes = all_windows_map[main_paper_id] # main paper entities
        extended_report[main_paper_id] = {"title":main_title,"abstract":main_abstract,"nes":main_nes} # add record for main paper
        ref_records = {}
        for ref_id in references_nes:
            # ref_title = references_nes[ref_id]["title"]
            ref_nes = references_nes[ref_id]["nes"] # this is a dict of entity labels
            ref_records[ref_id] = {"nes":ref_nes,
                                "title":references_nes[ref_id]["title"],
                                "abstract":references_nes[ref_id]["abstract"],
                                "relevance_score":report[ref_id]}
        extended_report[main_paper_id]["references"] = ref_records

        report_path = f"{RESULT_DIR}/{main_paper_id}.json"
        with open(report_path,"w") as f:
            json.dump(extended_report,f,indent=3)
        print(f"Record created for {main_paper_id}")
        print("-----------------END----------------------")

        _append_jsonl(PROCESSED_LOG_PATH, main_paper_id, report_path)
        processed_ids.add(main_paper_id)
        return True
    except Exception as e:
        print(f"Failed to generate report for {main_paper_id}: {e}")
        _append_jsonl(FAILED_LOG_PATH, main_paper_id, str(e))
        return False

# Lazy-loaded globals -- populated by _load_pipeline(), which only main()
# calls. generate_relevance_report()/get_entities_from_papers() read these as
# module globals, so they must be set here (not as locals of main()) for that
# lookup to work. collect_statistics() never calls _load_pipeline(), so it
# stays fast and doesn't load the models or the (multi-GB) paper index.
ner_model = None
embedding_model = None
processed_ids = None
paper_index = None
sample_main_paper_ids = None


def _load_pipeline():
    global ner_model, embedding_model, processed_ids, paper_index, sample_main_paper_ids

    processed_ids = _load_processed_ids(PROCESSED_LOG_PATH) # main_paper_ids already done in a prior run

    # load models
    ner_model = FineTunedNERModel(local_dir=LOCAL_MODEL_DIR,wandb_project=WANDB_PROJECT,artifact_name=WANDB_ARTIFACT) # for ner
    print("NER model loaded")
    embedding_model = EmbeddingModel()
    embedding_model.load_embedding_model()
    print('Embedding model loaded')

    # Load main papers
    paper_index = load_paper_index(max_files=MAX_MAIN_PAPERS) # turn off the max-files if you want to load all papers from dataset
    all_main_paper_ids = list(paper_index.keys())
    nb_main_papers = len(all_main_paper_ids)
    print(f'Total {nb_main_papers} Main papers')

    sample_main_paper_ids = random.sample(all_main_paper_ids, TEST_VOLUME)
    print(f'Total {len(sample_main_paper_ids)} Main papers')


async def main():
    _load_pipeline()
    count = 0
    for main_paper_id in sample_main_paper_ids:
        success = await generate_relevance_report(main_paper_id)
        count += 1
        if not success:
            print(f"Skipped {main_paper_id}")
        else:
            print(f"Completed {count}/{len(sample_main_paper_ids)}")


def type_classification(output_dir):
    pipeline = UnifiedQwenPipeline(ner_adapter_dir=f"{LOCAL_MODEL_DIR}/adapter_model")
    for root,_,files in os.walk(output_dir):
        for fname in files:
            if fname in ("processed_main.jsonl","failed_main.jsonl") or not fname.endswith(".json"):
                continue # tracking logs, not report files
            fpath = os.path.join(root,fname)
            with open(fpath) as f:
                data = json.load(f)
            modified,status = postprocess_json_with_types(pipeline, data)
            if status:
                # print("found aspect class")
                with open(fpath,"w") as f:
                    json.dump(modified,f,indent=3)
            else:
                print(f"main contribution not found for {fname}")

def collect_statistics(output_dir):
    print("Collecting LBL stat")
    histogram = {lbl:0 for lbl in LABELS}
    ref_count = 0
    for root,_,files in os.walk(output_dir):
        for fname in files:
            if fname in ("processed_main.jsonl","failed_main.jsonl") or not fname.endswith(".json"):
                continue # tracking logs, not report files
            fpath = os.path.join(root,fname)
            with open(fpath) as f:
                data = json.load(f)
            main_data = data[list(data.keys())[0]]
            all_ref_data = main_data["references"]
            for pid in all_ref_data:
                ref_data = all_ref_data[pid]
                ref_data_nes = ref_data["nes"]
                for lbl in ref_data_nes:
                    histogram[lbl] += 1
                ref_count += 1
    avg_histogram = {lbl: histogram[lbl]/ref_count for lbl in LABELS}
    print(histogram)
    print(avg_histogram)

if __name__ == "__main__":
    # asyncio.run(main())
    # collect_statistics(output_dir=RESULT_DIR)
    type_classification(output_dir=RESULT_DIR)