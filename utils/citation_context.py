"""Locate citation contexts of a cited paper (by arXiv id) inside an unarXive paper JSON."""
import json
import re
from utils.seku_crd_corpus_utils import *

class CitationContextExtractor:
    """Find the sentences citing a given arXiv paper, with surrounding context."""

    MARKER = r"\{\{(?:cite|figure|table):[^}]*\}\}"
    # a marker optionally followed by more ", marker" / " marker" (grouped citations)
    MARKER_GROUP_RE = re.compile(r"\s*" + MARKER + r"(?:\s*[,;]?\s*" + MARKER + r")*")
    # whitespace / commas / markers that may follow a sentence-final punctuation mark
    TRAIL_RE = re.compile(r"(?:\s|,|" + MARKER + r")*")
    # leftover of custom \section macros, e.g. "section1-0.09in0.02in"
    _UNIT = r"-?[\d.]+(?:in|pt|em|ex|cm|mm)"
    HEADING_ARTIFACT_RE = re.compile(r"(?:sub)*section\d" + _UNIT + _UNIT)
    ABBREVIATIONS = {
        "fig", "figs", "ref", "refs", "eq", "eqs", "sec", "secs", "tab", "no", "nos",
        "al", "e.g", "i.e", "cf", "vs", "etc", "approx", "resp", "ch", "vol", "pp", "dr", "prof",
    }
    ARXIV_DOI_PREFIX = "10.48550/arxiv."
    # fallback: arXiv id stated in the raw bib text when structured ids are empty
    ARXIV_ID_IN_TEXT_RE = re.compile(
        r"arxiv:\s*([a-z.\-]+/\d{7}|\d{4}\.\d{4,5})", re.IGNORECASE)

    def __init__(self, S=2):
        """S: sentences before/after the citing sentence (window = up to 2S+1)."""
        self.S = S

    # ---------- public API ----------

    def extract(self, paper, arxiv_id):
        """Return one dict per citing sentence of `arxiv_id` in `paper` (dict or JSON path)."""
        paper = self.load(paper)
        ref_ids = set(self.find_ref_ids(paper, arxiv_id))
        if not ref_ids:
            return []

        results, seen = [], set()
        for section_name, section in paper["sections"].items():
            spans = [c for c in section["cite_spans"] if c["ref_id"] in ref_ids]
            if not spans:
                continue
            text = section["text"]
            sentences = self.split_sentences(text)

            for c in spans:
                pos = self._locate(text, "{{cite:" + c["ref_id"] + "}}", c["start"])
                if pos < 0:
                    continue
                idx = next((k for k, (s, e) in enumerate(sentences) if s <= pos < e), None)
                if idx is None:
                    continue
                lo, hi = max(0, idx - self.S), min(len(sentences) - 1, idx + self.S)
                window = [self.clean(text[s:e]) for s, e in sentences[lo:hi + 1]]
                centre = idx - lo
                # same sentence cited twice, or duplicated paragraph in the source
                key = (section_name, window[centre])
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "section": section_name,
                    "ref_id": c["ref_id"],
                    "sentence_index": idx,
                    "before": window[:centre],
                    "center": self.clean(text[sentences[idx][0]:sentences[idx][1]], ref_ids),
                    "after": window[centre + 1:],
                    "context": " ".join(window),
                })
        return results

    def extract_many(self, paper, arxiv_ids):
        """Return {arxiv_id: [contexts]} for several cited papers of the same paper."""
        paper = self.load(paper)
        return {aid: self.extract(paper, aid) for aid in arxiv_ids}

    # ---------- reference lookup ----------

    @staticmethod
    def load(paper):
        if isinstance(paper, str):
            with open(paper, encoding="utf-8") as f:
                return json.load(f)
        return paper

    @staticmethod
    def normalise_arxiv_id(aid):
        aid = aid.strip().lower()
        aid = re.sub(r"^(https?://)?(arxiv\.org/abs/|arxiv:)", "", aid)
        return re.sub(r"v\d+$", "", aid)

    def find_ref_ids(self, paper, arxiv_id):
        """Return bib_entries keys whose ids point to the given arXiv id."""
        target = self.normalise_arxiv_id(arxiv_id)
        ref_ids = []
        for ref_id, bib in paper["bib_entries"].items():
            ids = bib.get("ids", {})
            doi = (ids.get("doi") or "").lower().replace("https://doi.org/", "")
            candidates = set()
            if doi.startswith(self.ARXIV_DOI_PREFIX):
                candidates.add(doi[len(self.ARXIV_DOI_PREFIX):])
            if ids.get("arxiv_id"):
                candidates.add(ids["arxiv_id"])
            for c in bib.get("contained_arXiv_ids") or []:
                # entries may be span dicts instead of id strings
                if isinstance(c, dict):
                    c = c.get("id")
                if c:
                    candidates.add(c)
            candidates.update(
                m.group(1) for m in self.ARXIV_ID_IN_TEXT_RE.finditer(bib.get("bib_entry_raw", "")))
            if target in {self.normalise_arxiv_id(c) for c in candidates}:
                ref_ids.append(ref_id)
        return ref_ids

    # ---------- sentence handling ----------

    def _is_abbreviation(self, text, dot_pos):
        m = re.search(r"([A-Za-z][A-Za-z.]*)$", text[:dot_pos])
        if not m:
            return False
        word = m.group(1)
        if len(word) == 1 and word.isupper():  # author initial, e.g. "K."
            return True
        return word.lower() in self.ABBREVIATIONS

    def split_sentences(self, text):
        """Return list of (start, end) char spans on the original text (markers kept)."""
        spans, start, i, n = [], 0, 0, len(text)
        while i < n:
            ch = text[i]
            if ch in ".!?":
                if ch == "." and self._is_abbreviation(text, i):
                    i += 1
                    continue
                j = i + 1
                while j < n and text[j] in ")]\"'":
                    j += 1
                k = self.TRAIL_RE.match(text, j).end()
                if k >= n:
                    break
                gap, nxt = text[j:k], text[k]
                prev = text[i - 1] if i > 0 else ""
                if gap:  # whitespace or markers after the punctuation
                    boundary = nxt.isupper() or nxt.isdigit() or nxt == "\\"
                else:    # paragraphs glued without space: "superconductivity.The"
                    boundary = nxt.isupper() and (prev.islower() or prev in ")}")
                if boundary:
                    spans.append((start, k))
                    start = i = k
                    continue
            i += 1
        if start < n:
            spans.append((start, n))

        # heading artifacts always start a new sentence
        cuts = [m.start() for m in self.HEADING_ARTIFACT_RE.finditer(text)]
        split = []
        for s, e in spans:
            for c in (c for c in cuts if s < c < e):
                split.append((s, c))
                s = c
            split.append((s, e))
        return [(s, e) for s, e in split if self.clean(text[s:e])]

    def clean(self, sentence, cite_ids=None):
        """Strip markers; a marker group containing any of `cite_ids` becomes [Citation]."""
        s = self.HEADING_ARTIFACT_RE.sub("", sentence)
        if cite_ids:
            s = self.MARKER_GROUP_RE.sub(
                lambda m: " [Citation]" if any("{{cite:" + r + "}}" in m.group() for r in cite_ids) else "", s)
            s = re.sub(r"([(\[])\s+\[Citation\]", r"\1[Citation]", s)
        else:
            s = self.MARKER_GROUP_RE.sub("", s)
        s = re.sub(r"\(\s*\)|\[\s*\]", "", s)
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s+([.,;:!?)])", r"\1", s)
        s = re.sub(r"(?<!\.)\.\.(?!\.)", ".", s)  # "Ref.{{cite}}." -> "Ref.."
        return s.strip()

    @staticmethod
    def _locate(text, marker, hint):
        if text[hint:hint + len(marker)] == marker:
            return hint
        positions = [m.start() for m in re.finditer(re.escape(marker), text)]
        return min(positions, key=lambda p: abs(p - hint)) if positions else -1

def find_citation_context(main_paper_id:str,bulk_paper_index:dict,citation_context_extractor:CitationContextExtractor,output_dir:str=None,max_refs:int=32):
    """Collect citation contexts for a main paper's in-corpus references.

    Args:
        main_paper_id: Corpus id of the main paper.
        bulk_paper_index: {paper_id: record} from load_paper_index.
        citation_context_extractor: CitationContextExtractor instance.
        output_dir: If set, the result is also written to <output_dir>/<main_paper_id>.json.
        max_refs: Max references to process; -1 for all.

    Returns:
        (status, {main_paper_id: {"references": {...}, "abstract": ...}})

    Note:
        Writes the main paper to ./temp_paper_text.json, so concurrent calls
        are not safe.
    """
    status = False
    final_dict = {}
    try:
        citation_contexts = {}
        # read main paper content
        # 1) paper title, abstract and in corpus refrences
        main_paper_w_refs = bulk_paper_index[main_paper_id] # data from paper index file, contains title, abstract and citations
        # 2) paper text as in Latex content from S3, and dump in a temp file
        main_paper_text = get_paper_from_s3(main_paper_id)
        main_paper_title,main_paper_abstract = get_title_and_abstract(main_paper_text)
        fname = "temp_paper_text.json"
        with open(fname,"w") as f:
            json.dump(main_paper_text,f,indent=2)
        # now we collect all the refs
        in_corpus_refs_ids = main_paper_w_refs["in_corpus_references"] # list of reference paper IDs
        # but may be, not all references
        if max_refs != -1:
            in_corpus_refs_ids = in_corpus_refs_ids[:min(len(in_corpus_refs_ids),max_refs)]
        # errors are caught per reference so one bad reference doesn't drop the rest
        successfull = 0
        for ref_paper_id in in_corpus_refs_ids:
            try:
                result = citation_context_extractor.extract(fname, ref_paper_id) # this will be list of dicts for every citation context
                if result: # non empty list
                    # collect abstract of reference paper
                    ref_paper_text = get_paper_from_s3(ref_paper_id)
                    ref_paper_title,ref_paper_abstract = get_title_and_abstract(ref_paper_text)
                    citation_contexts.update({ref_paper_id:{"citation_context":result,"abstract":ref_paper_abstract}})
                    successfull += 1
            except Exception as e:
                print(f"  ref {ref_paper_id}: {e}")
                continue
        # done, collected from all refs
        final_dict = {main_paper_id:{"references":citation_contexts,"abstract":main_paper_abstract}}
        print(f"Extracted context for {main_paper_id} success [{successfull}/{len(in_corpus_refs_ids)}]")
        if output_dir:
            # dump in output path
            op_fname = os.path.join(output_dir,f"{main_paper_id}.json")
            with open(op_fname,"w") as f:
                json.dump(final_dict,f,indent=2)
        status = True
    except Exception as e:
        print(e)
    return status,final_dict

if __name__ == "__main__":
    import sys
    path, aid = sys.argv[1], sys.argv[2]
    s = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    print(json.dumps(CitationContextExtractor(S=s).extract(path, aid), indent=2, ensure_ascii=False))
