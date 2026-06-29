import traceback as tr

import re



def extract_info(raw_text:str,annotations:dict,info_type:str):
    try:
        annotations = eval(annotations[info_type]) # this can be more than one entries
        attribute_texts = []
        # print(f'Total {len(annotations)} markers')
        for item in annotations:
            start = int(item['start'])
            end = int(item["end"])
            data = raw_text[start:end]
            attribute_texts.append(data)
    except Exception as e:
        print("Can not extract info ",tr.format_exc())
    return attribute_texts

def extract_info_w_attributes(raw_text:str,raw_annotations:any,info_type:str):
    status = False
    attribute_texts = []
    try:
        if isinstance(raw_annotations,str):
            raw_annotations = eval(raw_annotations)
        annotations = raw_annotations[info_type]
        if isinstance(annotations,str):
            annotations = eval(annotations) # this can be more than one entries
        if annotations is not None:
            # print(f'Total {len(annotations)} markers')
            for item in annotations: # here we can have an attribute key as well
                start = int(item['start'])
                end = int(item["end"])
                if 'attributes' in item.keys():
                    attribute = item['attributes']
                    attribute_vals = []
                    for k in attribute.keys():
                        at_val = attribute[k]
                        attribute_vals.append(at_val)
                else:
                    attribute_vals = None
                data = raw_text[start:end]
                attribute_texts.append((attribute_vals,data))
            status = True
    except Exception as e:
        print("Can not extract info ",tr.format_exc())
    return status, attribute_texts

def clean_n_augment(info:list):
    clean_info = ""
    for item in info:
        clean_info += item[-1] + "\n" # basically ignore 1st part which is attribute and value is null
    return clean_info

def process_bibref(info:list):
    bib_refs = {}
    try:
        for item in info:
            if len(item) == 2 and item[0] is not None:
                key = item[0][0].replace("[","").replace("]","").strip() # like b0,b1 etc.
                value = item[1] # like [1],[2] etc.
                bib_refs[key] = value
    except Exception as e:
        print(f"Can not extract info from bibref {info}",tr.format_exc())
    return bib_refs



def extract_citation_ids(lst):
    ref_id = None
    paper_id = None
    try:
        for item in lst:
            item = str(item)
            if re.match(r'^b\d+$', item):     # matches b0, b5, b12 ...
                ref_id = item
            elif re.match(r'^\d+$', item):    # matches any pure numeric string
                paper_id = item
    except Exception as e:
        print(f"Can not extract info from bibentries {lst}",tr.format_exc())
    return {"ref_id": ref_id, "paper_id": paper_id}


def process_bibentries(info:list,guide:dict):
    bib_entries = {}
    well_formed_entries = 0
    try:
        for item in info:
            # now each item is likely to be (['b1', 36244258], 'paper name and details)
            ref = item[0] # should be either ['b8', 5380403] or ['b6']
            if ref is not None and isinstance(ref,list) and len(ref) == 2: # we have a matched paper id
                all_ids = extract_citation_ids(ref)
                matched_paper_id = all_ids["paper_id"]
                bib_ref = all_ids["ref_id"].strip() # 'b6
                if bib_ref in guide.keys(): # we know how it is cited in paper
                    bib_render = guide[bib_ref] # [2] or [3] like this
                    bib_entries[bib_ref] = {"cite_key":bib_render,"cite_text":item[1],"matched_paper_id":matched_paper_id}
            else:
                well_formed_entries += 1
        total_citations = len(bib_entries)
        # if total_citations == 0:
        #     print(f"No citations found. Entries without paper id {well_formed_entries}/{total_citations}")
        # else:
        #     print(f"Total {total_citations} citation entries found")
    except Exception as e:
        print(f"Can not extract info from bibentries {info}",tr.format_exc())
    return bib_entries,total_citations


def extract_from_jsonl(raw_content:dict,corpus_id:str):
    record = {}
    success = False
    try:
        all_text = raw_content["text"]
        all_annotations = raw_content["annotations"]
        status,title = extract_info_w_attributes(all_text,all_annotations,"title")
        # process title
        clean_title = clean_n_augment(title)
        if status: # title extraction is success
            status,abstract = extract_info_w_attributes(all_text,all_annotations,"abstract")
            if status: # abstract extraction is success
                clean_abstract = clean_n_augment(abstract)
                status,bibref = extract_info_w_attributes(all_text,all_annotations,"bibref") # these are refs like [1],[2] etc
                if status:
                    all_bibrefs = process_bibref(bibref) # this is a dict like {"b0":[1]...}
                    status,bibentry = extract_info_w_attributes(all_text,all_annotations,"bibentry") # these are entries like paper details,author etc
                    raw_citations = len(bibentry)
                    all_bibentries,nb_citations = process_bibentries(bibentry,all_bibrefs)
                    if nb_citations > 0:
                        record = {"corpus_id":corpus_id, "title": clean_title, "abstract": clean_abstract,"citation":all_bibentries}
                        success = True
                    else:
                        record = {}
                        success = False
                        reason = "No Citation"
                else:
                    reason = "Citation format"
                    print(f"finished clean extraction from {corpus_id}, #valid citations {nb_citations}, #total citations {raw_citations}")
    except Exception as e:
        print(f"Can not extract this info",tr.format_exc())
    return success,record