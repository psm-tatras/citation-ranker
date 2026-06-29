import json
from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv
import os
import sys
status = load_dotenv()
print("ENV loaded ",status)

def get_collection(
    db_name: str = "research_db",
    collection_name: str = "citations_main",
) -> Collection:
    """Connect to MongoDB using env variables and return the collection."""
    client = MongoClient(
        host=os.getenv("MONGO_HOST", "localhost"),
        port=int(os.getenv("MONGO_PORT", 27017)),
        username=os.getenv("MONGO_USERNAME"),
        password=os.getenv("MONGO_PASSWORD")
    )
    db = client[db_name]

    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
        db[collection_name].create_index("corpus_id", unique=True)
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")

    return db[collection_name]

def insert_record(collection: Collection, data: dict):
    """Insert a single record into the collection."""
    corpus_id = data["corpus_id"]
    # record = {
    #     "corpus_id": corpus_id,
    #     "title": data["title"],
    #     "abstract": data["abstract"],
    #     "citation": data["citation"]
    # }
    result = collection.insert_one(data)
    print(f"Inserted corpus_id: {corpus_id} -> _id: {result.inserted_id}")


def insert_many_records(collection: Collection, records: dict):
    """Insert multiple records at once from a {corpus_id: {...}} dict."""
    docs = [
        {
            "corpus_id": corpus_id,
            "title": data["title"],
            "abstract": data["abstract"],
            "citation": data["citation"]
        }
        for corpus_id, data in records.items()
    ]
    result = collection.insert_many(docs)
    print(f"Inserted {len(result.inserted_ids)} records.")


def insert_from_jsonl(collection: Collection, jsonl_path: str):
    """Stream a JSONL file and insert records one by one (memory efficient)."""
    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            # for corpus_id, data in record.items():
            insert_record(collection, record)
            count += 1
    print(f"Done. Total inserted: {count}")


def search_by_corpus_id(collection: Collection, corpus_id: str) -> dict | None:
    """Search a record by corpus_id."""
    record = collection.find_one({"corpus_id": corpus_id})
    
    if record:
        # print(f"Found record for corpus_id: {corpus_id}")
        return record
    else:
        # print(f"No record found for corpus_id: {corpus_id}")
        return None

def get_collection_stats(collection: Collection):
    stats = collection.database.command("collstats", collection.name, scale=1024*1024)
    print(f"Storage size : {stats['storageSize']} MB")
    print(f"Total size   : {stats['totalSize']} MB")
    print(f"Index size   : {stats['totalIndexSize']} MB")
    print(f"Doc count    : {stats['count']}")


def find_existing_citations(collection: Collection,corpus_id:int):
    found_citations = []
    total_citation = 0
    paper = search_by_corpus_id(collection,corpus_id)
    if paper:
        citations_in_paper = paper["citation"]
        
        for citation_id in citations_in_paper.keys():
            matched_paper_id = citations_in_paper[citation_id]["matched_paper_id"]
            total_citation += 1
            if matched_paper_id is not None:
                cited_paper = search_by_corpus_id(collection,int(matched_paper_id))
                if cited_paper:
                    found_citations.append(matched_paper_id)
    return found_citations,total_citation

def scan_corpus_ids_stream(collection: Collection):
    """Memory efficient - processes one at a time."""
    cursor = collection.find({}, {"corpus_id": 1, "_id": 0}).batch_size(500)
    f = open("/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/all_matched_citations.csv","w")
    count = 0
    for doc in cursor:
        try:
            corpus_id = doc["corpus_id"]
            found_citations,total_citation = find_existing_citations(collection,corpus_id=corpus_id)
            if len(found_citations) > 0:
                # print("found match in ",corpus_id)
                success = len(found_citations) / total_citation
                f.write(f"{corpus_id},{len(found_citations)},{total_citation},{success}\n")
            count += 1
            # print(f"Total scanned: {count}")
            sys.stdout.write(f"\rTotal processed {count}")
            sys.stdout.flush()
        except Exception as e:
            print(e)
        # if count >=1000:
        #     break
    f.close()


if __name__ == "__main__":
    collection = get_collection()
    get_collection_stats(collection)
    
    # Insert from a JSONL file (memory efficient for large files)
    # data_dir = "/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/extracted"
    # for root,_,fnames in os.walk(data_dir):
    #     for fname in fnames:
    #         path = os.path.join(root,fname)
    #         if fname.split(".")[-1] == "jsonl":
    #             print("Insert from ",fname)
    #             insert_from_jsonl(collection, path)
            # break
    scan_corpus_ids_stream(collection)