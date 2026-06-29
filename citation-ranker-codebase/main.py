from utils.mongo_utils import *

if __name__ == "__main__":
    ARCHIVE = "/media/parthosarothi/Studio/UbuntuWorkSpace/SemanticScholarDataset/extracted"
    collection = get_collection()
    for root,_,fnames in os.walk(ARCHIVE):
        for fn in fnames:
            if ".jsonl" in fn:
                fpath = os.path.join(root,fn)
                insert_from_jsonl(collection, fpath)
    get_collection_stats(collection)
