import json
from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv
import os
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
        print(f"Found record for corpus_id: {corpus_id}")
        return record
    else:
        print(f"No record found for corpus_id: {corpus_id}")
        return None

def get_collection_stats(collection: Collection):
    stats = collection.database.command("collstats", collection.name, scale=1024*1024)
    print(f"Storage size : {stats['storageSize']} MB")
    print(f"Total size   : {stats['totalSize']} MB")
    print(f"Index size   : {stats['totalIndexSize']} MB")
    print(f"Doc count    : {stats['count']}")

# ── Usage ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    collection = get_collection()
    get_collection_stats(collection)
    # Insert a single record
    # insert_record(collection, "27603005", {
    #     "title": "Some Paper Title",
    #     "abstract": "Some abstract text...",
    #     "citation": [{"ref_id": "b0", "paper_id": "123456"}]
    # })

    # # Insert multiple records from a dict
    # records = {
    #     "27603005": {"title": "Title A", "abstract": "Abstract A", "citation": []},
    #     "72705592": {"title": "Title B", "abstract": "Abstract B", "citation": []},
    # }
    # insert_many_records(collection, records)

    # Insert from a JSONL file (memory efficient for large files)
    # insert_from_jsonl(collection, "/home/parthosarothi/IISER-Work/citation-ranker/citation-ranker/20260313_115134_00043_vcx4r_0a24153e-8434-4220-9c25-922408f36db8.gz_citations.jsonl")