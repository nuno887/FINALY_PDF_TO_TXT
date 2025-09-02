import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import spacy
from spacy.matcher import Matcher

INPUT_ROOT = Path("output")                  # where <pdf-stem>/completo.txt lives
OUTPUT_ROOT = Path("output_EXTRACTED_DOCUMENTS")
DEBUG = True  # set to False to silence debug prints

# ---------------------------
# spaCy model
# ---------------------------
KIND_WORDS = [
    "despacho", "aviso", "declaração", "declaracao", "edital",
    "deliberação", "deliberacao", "contrato", "resolução", "resolucao",
    "revogação", "revogacao", "caducidade", "ato", "acto"
]
KIND_SET = set(KIND_WORDS)

NON_MODIFIER_TOKENS = {
    "de", "da", "do", "das", "dos", "e", "em", "para", "no", "na", "nos", "nas"
}
N_TOKENS = {"n.º", "nº", "n.o", "n.°", "n°", "n."}

def load_pt_model():
    for name in ("pt_core_news_lg", "pt_core_news_md", "pt_core_news_sm"):
        try:
            return spacy.load(name)
        except Exception:
            continue
    raise RuntimeError("No Portuguese spaCy model found. Install e.g. pt_core_news_lg.")

NLP = load_pt_model()

def build_org_matcher(nlp) -> Matcher:
    org_matcher = Matcher(nlp.vocab)
    org_patterns = [
        [{"IS_UPPER": True, "OP": "+"}],
        [{"IS_UPPER": True, "OP": "+"}, {"TEXT": "-"}, {"IS_UPPER": True, "OP": "+"}],
    ]
    for p in org_patterns:
        org_matcher.add("ORG_HEAD", [p])
    return org_matcher

ORG_MATCHER = build_org_matcher(NLP)

# ---------------------------
# Helpers (no regex; original text preserved)
# ---------------------------
def norm_ws(s: str) -> str:
    return " ".join(s.replace("\u00A0", " ").split()).lower()

def is_org_line(text: str) -> bool:
    t = text.strip()
    if not t or t == "Sumário":
        return False
    doc = NLP(t)
    return len(ORG_MATCHER(doc)) > 0

def consume_org_block(lines: List[str], i: int) -> Tuple[str, int]:
    parts = [lines[i].strip()]
    j = i + 1
    while j < len(lines):
        nxt = lines[j]
        if not nxt.strip():
            break
        if extract_act_from_line(nxt):
            break
        if is_org_line(nxt):
            parts.append(nxt.strip())
            j += 1
            if len(parts) >= 3:
                break
            continue
        break
    return " ".join(parts), j

def digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

# ---------------------------
# ACT parsing & detection (header-only rule; no matcher gate)
# ---------------------------
def parse_act_tokens(doc):
    """
    Extract (kind_base, kind_modifier, number, year, kind_full, last_idx_used) from ONE LINE doc.
    last_idx_used = index of last header token. Kind must be at the start (only punctuation before).
    """
    kind_base = None
    number = None
    year = None
    last_idx_used = -1

    kind_idx = None
    for i, tok in enumerate(doc):
        if tok.text.lower() in KIND_SET:
            if all((t.is_punct or (not t.text.strip())) for t in doc[:i]):
                kind_base = tok.text
                kind_idx = i
                last_idx_used = i
            break
    if kind_idx is None:
        return None

    modifier_tokens: List[str] = []
    i = kind_idx + 1
    while i < len(doc):
        t = doc[i]
        txt = t.text
        low = txt.lower()
        if txt in N_TOKENS:
            last_idx_used = i
            break
        if any(ch.isdigit() for ch in txt) or txt == "/":
            break
        if t.is_alpha and low not in NON_MODIFIER_TOKENS:
            modifier_tokens.append(txt)
            last_idx_used = i
        i += 1

    j = kind_idx + 1
    while j < len(doc) and number is None:
        tok = doc[j]
        txt = tok.text
        if any(ch.isdigit() for ch in txt):
            if "/" in txt:
                left, _, right = txt.partition("/")
                number = digits_only(left) or number
                y = digits_only(right)
                if y:
                    year = y
                last_idx_used = max(last_idx_used, j)
            else:
                number = digits_only(txt) or number
                if j + 2 < len(doc) and doc[j+1].text == "/" and any(ch.isdigit() for ch in doc[j+2].text):
                    year = digits_only(doc[j+2].text)
                    last_idx_used = max(last_idx_used, j + 2)
                else:
                    k = j + 1
                    while k < len(doc):
                        if any(ch.isdigit() for ch in doc[k].text):
                            year = digits_only(doc[k].text)
                            last_idx_used = max(last_idx_used, k)
                            break
                        k += 1
            last_idx_used = max(last_idx_used, j)
        j += 1

    if not (kind_base and number and year):
        return None

    kind_modifier = " ".join(modifier_tokens).strip()
    kind_full = (kind_base + (" " + kind_modifier if kind_modifier else "")).strip()
    return kind_base, kind_modifier, number, year, kind_full, last_idx_used

def extract_act_from_line(text: str):
    """
    Return (kind_base, kind_modifier, number, year, kind_full) ONLY if the line is a pure header:
    no alpha/digit tokens after the header (punct allowed).
    """
    t = text.strip()
    if not t or t == "Sumário":
        return None
    doc = NLP(t)

    parsed = parse_act_tokens(doc)
    if not parsed:
        return None
    kind_base, kind_modifier, number, year, kind_full, last_idx = parsed

    for tok in doc[last_idx + 1:]:
        if tok.is_alpha or any(ch.isdigit() for ch in tok.text):
            return None
    return (kind_base, kind_modifier, number, year, kind_full)

# ---------------------------
# Pass 1: Sumário (find range + parse items)
# ---------------------------
def find_sumario_range(lines: List[str]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "Sumário":
            start = i
            break
    if start is None:
        return None, None, None

    first_org = None
    first_org_idx = None
    j = start + 1
    while j < len(lines):
        if is_org_line(lines[j]):
            first_org, next_after_org = consume_org_block(lines, j)
            first_org_idx = j
            break
        j += 1
    if first_org is None:
        end = min(len(lines), start + 120)
        return start, end, None

    target = norm_ws(first_org)
    end = None
    k = next_after_org
    while k < len(lines):
        one = norm_ws(lines[k].strip())
        if one == target:
            end = k
            break
        if k + 1 < len(lines):
            two = norm_ws(lines[k].strip() + " " + lines[k+1].strip())
            if two == target:
                end = k
                break
        k += 1
    if end is None:
        k = next_after_org
        while k < len(lines):
            if is_org_line(lines[k]):
                end = k
                break
            k += 1
    if end is None:
        end = len(lines)
    return start, end, first_org

def parse_sumario_items(sum_lines: List[str]) -> List[Dict[str, Any]]:
    items = []
    current_org = None
    i = 0
    while i < len(sum_lines):
        line = sum_lines[i]
        if is_org_line(line):
            combined_org, j = consume_org_block(sum_lines, i)
            current_org = combined_org
            i = j
            continue

        parsed = extract_act_from_line(line)
        if parsed:
            kind_base, kind_modifier, number, year, kind_full = parsed
            act_text = line.strip()
            lead = []
            j = i + 1
            while j < len(sum_lines):
                nxt = sum_lines[j]
                if not nxt.strip():
                    break
                if is_org_line(nxt) or extract_act_from_line(nxt):
                    break
                lead.append(nxt.rstrip())
                if len(lead) >= 3:
                    break
                j += 1

            items.append({
                "section": current_org,
                "kind_base": kind_base,
                "kind_modifier": kind_modifier,
                "kind_full": kind_full,
                "number": number,
                "year": year,
                "act_text": act_text,
                "title": " ".join(lead).strip(),
                "sumario_line_range": [i, max(i, j - 1)]
            })
            i = j
            continue

        i += 1
    return items

# ---------------------------
# Pass 2 helpers (Sumário ignored)
# ---------------------------
def build_body_header_index(lines: List[str], sum_start: Optional[int], sum_end: Optional[int]):
    """
    Returns:
      idx: dict with keys ("base"/"full", kind_lower, number, year) -> [line_idx, ...]
      entries: list of parsed body headers with line numbers (for debug)
    """
    def in_sumario(i):
        return sum_start is not None and sum_end is not None and sum_start <= i < sum_end

    idx: Dict[Tuple[str, str, str, str], List[int]] = {}
    entries: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        if in_sumario(i):
            continue
        parsed = extract_act_from_line(line)
        if not parsed:
            continue
        k_base, k_mod, num, yr, k_full = parsed
        key_base = ("base", k_base.lower(), num, yr)
        key_full = ("full", k_full.lower(), num, yr)
        idx.setdefault(key_base, []).append(i)
        if k_mod:
            idx.setdefault(key_full, []).append(i)
        entries.append({
            "line": i, "kind_base": k_base, "kind_modifier": k_mod, "kind_full": k_full,
            "number": num, "year": yr, "text": line.rstrip()
        })
    return idx, entries

def build_global_org_map(lines: List[str], body_start: int):
    org_positions_all: List[Tuple[int, str, int]] = []
    i = body_start
    while i < len(lines):
        if is_org_line(lines[i]):
            org_text, j = consume_org_block(lines, i)
            org_positions_all.append((i, org_text, j))
            i = j
        else:
            i += 1
    return org_positions_all

def find_org_for_line(org_positions_all: List[Tuple[int, str, int]], line_idx: int,
                      default_name: str, default_start: int, default_end: int):
    org_name = None
    org_start = None
    org_end = default_end
    for a, (s, name, _next_after) in enumerate(org_positions_all):
        if s <= line_idx:
            org_name = name
            org_start = s
            org_end = org_positions_all[a + 1][0] if a + 1 < len(org_positions_all) else default_end
        else:
            break
    if org_name is None:
        return default_name, default_start, default_end
    return org_name, org_start, org_end

# ---------------------------
# Per PDF-stem processing
# ---------------------------
def process_pdf_stem(stem_dir: Path) -> None:
    completo = stem_dir / "completo.txt"
    if not completo.exists():
        return

    out_dir = OUTPUT_ROOT / stem_dir.name
    docs_dir = out_dir / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(completo, out_dir / "completo.txt")
    lines = completo.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Pass 1: Sumário
    sum_start, sum_end, _ = find_sumario_range(lines)
    sumario_raw = lines[sum_start:sum_end] if (sum_start is not None and sum_end is not None) else []
    (out_dir / "sumario_raw.txt").write_text("\n".join(sumario_raw), encoding="utf-8")
    items = parse_sumario_items(sumario_raw)
    (out_dir / "sumario_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    if DEBUG:
        print(f"\n=== {stem_dir.name} ===")
        rng = f"{sum_start}..{(sum_end-1) if (sum_start is not None and sum_end is not None) else 'N/A'}"
        print(f"Sumário range: {rng}")
        print("ACTs (Sumário order):")
        for idx, it in enumerate(items, 1):
            print(f"  {idx:02d}. [{it.get('section')}] {it['kind_full']} {it['number']}/{it['year']}")

    # Pass 2: Body-only
    body_start = sum_end if sum_end is not None else 0
    org_positions_all = build_global_org_map(lines, body_start)
    header_index, header_entries = build_body_header_index(lines, sum_start, sum_end)

    if DEBUG:
        print("Body ORG anchors:")
        for s, name, _n in org_positions_all:
            print(f"  - line {s}: {name}")
        print(f"Body header anchors: {len(header_entries)}")

    # Anchor each sumário item globally (prefer full, fallback base), track used anchors
    used_anchors: set[int] = set()
    anchored_hits: List[Tuple[int, Dict[str, Any]]] = []  # (line, item)
    unmatched: List[Dict[str, Any]] = []

    for it in items:
        key_full = ("full", it["kind_full"].lower(), it["number"], it["year"])
        key_base = ("base", it["kind_base"].lower(), it["number"], it["year"])
        candidates = header_index.get(key_full, []) or header_index.get(key_base, [])
        anchor = None
        for ln in candidates:
            if ln not in used_anchors:
                anchor = ln
                break
        if anchor is None:
            unmatched.append({
                "section": it.get("section"),
                "kind": it["kind_base"].lower(),
                "act_raw": it["act_text"],
                "reason": "no_body_anchor_found_or_all_anchors_used"
            })
            if DEBUG:
                print(f"UNMATCHED: {it['kind_full']} {it['number']}/{it['year']}")
            continue
        used_anchors.add(anchor)
        anchored_hits.append((anchor, it))

    if not anchored_hits:
        (out_dir / "index.json").write_text(json.dumps({
            "pdf_stem": stem_dir.name,
            "source_completo": str(completo),
            "sumario_file": "sumario_raw.txt",
            "sumario_items_file": "sumario_items.json",
            "items": [],
            "unmatched": unmatched
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # Sort all anchors globally; cut ACT -> next ACT (last goes to EOF)
    anchored_hits.sort(key=lambda x: x[0])
    anchor_lines = [a for a, _ in anchored_hits]
    boundaries = anchor_lines[1:] + [len(lines)]

    used_names = set()
    def make_doc_filename(kind_full: str, number: str) -> str:
        base = f"{kind_full}-{number}.txt"
        safe = "".join(ch for ch in base if ch not in '\\/:*?"<>|')
        name = safe
        suffix = ord('a')
        while name in used_names:
            name = safe.replace(".txt", f"-{chr(suffix)}.txt")
            suffix += 1
        used_names.add(name)
        return name

    index = {
        "pdf_stem": stem_dir.name,
        "source_completo": str(completo),
        "sumario_file": "sumario_raw.txt",
        "sumario_items_file": "sumario_items.json",
        "items": [],
        "unmatched": unmatched
    }

    # Prep a quick function to get ORG for a given anchor (for metadata & optional prepend)
    def org_for_anchor(anchor: int):
        name, org_s, org_e = find_org_for_line(
            org_positions_all, anchor,
            default_name="(ORG não identificado)",
            default_start=body_start,
            default_end=len(lines)
        )
        return name, org_s, org_e

    for (start, it), end in zip(anchored_hits, boundaries):
        # body slice: ACT -> next ACT (or EOF)
        chunk_lines = lines[start:end]

        # ORG metadata (not used for slicing)
        body_org_name, _org_s, _org_e = org_for_anchor(start)

        # Prepend ORG heading if the first line isn't already it (purely cosmetic/labeling)
        doc_lines = []
        if not chunk_lines or norm_ws(chunk_lines[0]) != norm_ws(body_org_name):
            doc_lines.append(body_org_name)
            doc_lines.append("")
        doc_lines.extend(chunk_lines)

        fname = make_doc_filename(it["kind_full"], it["number"])
        (docs_dir / fname).write_text("\n".join(doc_lines), encoding="utf-8")

        if DEBUG:
            print(f"-> wrote {fname}  lines [{start}:{end})")

        # Simplified index item:
        index["items"].append({
            "section_sumario": it.get("section"),
            "section_body": body_org_name,
            "kind": it["kind_base"].lower(),                 # just 'despacho' etc.
            "act_raw": lines[start].strip(),                 # exact header from BODY
            "doc_file": f"docs/{fname}",
            "body_line_range": [start, end - 1],
            "sumario_line_range": it.get("sumario_line_range"),
        })

    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------
# Batch over all PDFs
# ---------------------------
def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not INPUT_ROOT.exists():
        print(f"Input root not found: {INPUT_ROOT.as_posix()}")
        return
    processed = 0
    for sub in sorted(p for p in INPUT_ROOT.iterdir() if p.is_dir()):
        if (sub / "completo.txt").exists():
            process_pdf_stem(sub)
            processed += 1
    print(f"Processed {processed} PDFs.")

if __name__ == "__main__":
    main()
