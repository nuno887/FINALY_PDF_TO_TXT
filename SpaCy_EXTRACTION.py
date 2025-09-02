import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import spacy
from spacy.matcher import Matcher

INPUT_ROOT = Path("output")                  # where <pdf-stem>/completo.txt lives
OUTPUT_ROOT = Path("output_EXTRACTED_DOCUMENTS")
DEBUG = True  # set to False to silence debug prints

# ---------------------------
# spaCy model + matchers
# ---------------------------
KIND_WORDS = [
    "despacho", "aviso", "declaração", "declaracao", "edital",
    "deliberação", "deliberacao", "contrato", "resolução", "resolucao",
    "revogação", "revogacao", "caducidade", "ato", "acto"
]
KIND_SET = set(KIND_WORDS)

def load_pt_model():
    for name in ("pt_core_news_lg", "pt_core_news_md", "pt_core_news_sm"):
        try:
            return spacy.load(name)
        except Exception:
            continue
    raise RuntimeError("No Portuguese spaCy model found. Install e.g. pt_core_news_lg.")

NLP = load_pt_model()

def build_matchers(nlp) -> Tuple[Matcher, Matcher]:
    org_matcher = Matcher(nlp.vocab)
    des_matcher = Matcher(nlp.vocab)

    # ORG: sequences of uppercase tokens; allow hyphenated too
    org_patterns = [
        [{"IS_UPPER": True, "OP": "+"}],
        [{"IS_UPPER": True, "OP": "+"}, {"TEXT": "-"}, {"IS_UPPER": True, "OP": "+"}],
    ]
    for p in org_patterns:
        org_matcher.add("ORG_HEAD", [p])

    # DES/ACT: line-based; optional words and optional "n.º"/variants; then number and optional "/ year"
    des_patterns = [
        [
            {"LOWER": {"IN": KIND_WORDS}},
            {"LOWER": {"IN": ["conjunto", "de", "da", "do", "retificação", "retificacao", "societário", "societario"]}, "OP": "*"},
            {"TEXT": {"IN": ["n.º", "nº", "n.o", "n.°"]}, "OP": "?"},
            {"LIKE_NUM": True},
            {"TEXT": "/", "OP": "?"},
            {"LIKE_NUM": True, "OP": "?"},
        ],
        [
            {"LOWER": {"IN": KIND_WORDS}},
            {"LOWER": {"IN": ["conjunto", "de", "da", "do", "retificação", "retificacao", "societário", "societario"]}, "OP": "*"},
            {"LIKE_NUM": True},
            {"TEXT": "/", "OP": "?"},
            {"LIKE_NUM": True, "OP": "?"},
        ],
    ]
    for p in des_patterns:
        des_matcher.add("DES_ACT", [p])

    return org_matcher, des_matcher

ORG_MATCHER, DES_MATCHER = build_matchers(NLP)

# ---------------------------
# Helpers (no regex; original text preserved)
# ---------------------------
def is_org_line(text: str) -> bool:
    """Decide if a single line is an ORG heading using spaCy matcher (no regex)."""
    t = text.strip()
    if not t or t == "Sumário":
        return False
    doc = NLP(t)
    return len(ORG_MATCHER(doc)) > 0

def parse_act_tokens(doc) -> Optional[Tuple[str, str, str]]:
    """
    Extract (kind, number, year) from a spaCy Doc representing ONE LINE.
    Handles:
      - 'Despacho n.º 215/2025'
      - 'Despacho 215/2025'
      - token '215/2025' as a single token (split by '/')
    """
    kind = None
    number = None
    year = None

    def digits_only(s: str) -> str:
        return "".join(ch for ch in s if ch.isdigit())

    i = 0
    while i < len(doc):
        tok = doc[i]
        low = tok.text.lower()
        if kind is None and low in KIND_SET:
            kind = tok.text  # keep original case/accents as seen on this line

        # number/year detection (robust to a single token like "215/2025")
        if number is None:
            txt = tok.text
            if any(ch.isdigit() for ch in txt):
                if "/" in txt:
                    left, _, right = txt.partition("/")
                    num = digits_only(left)
                    yr = digits_only(right)
                    if num:
                        number = num
                    if yr:
                        year = yr
                else:
                    number = digits_only(txt) or number

                # try to fetch year nearby if not set yet
                if year is None:
                    # if next tokens contain '/' and a numeric, capture year
                    if i + 2 < len(doc) and doc[i+1].text == "/" and any(ch.isdigit() for ch in doc[i+2].text):
                        year = digits_only(doc[i+2].text)
                    else:
                        # otherwise, take the next numeric token as year
                        j = i + 1
                        while j < len(doc):
                            if any(ch.isdigit() for ch in doc[j].text):
                                year = digits_only(doc[j].text)
                                break
                            j += 1
        i += 1

    if kind and number and year:
        return kind, number, year
    return None

def extract_act_from_line(text: str) -> Optional[Tuple[str, str, str]]:
    """Return (kind, number, year) if the line is an ACT header; else None."""
    t = text.strip()
    if not t or t == "Sumário":
        return None
    doc = NLP(t)
    if len(DES_MATCHER(doc)) == 0:
        return None
    return parse_act_tokens(doc)

# ---------------------------
# Sumário detection (as specified)
# ---------------------------
def find_sumario_range(lines: List[str]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    # 1) first line exactly "Sumário"
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "Sumário":
            start = i
            break
    if start is None:
        return None, None, None

    # 2) first ORG after "Sumário" (spaCy matcher)
    first_org = None
    first_org_idx = None
    for j in range(start + 1, len(lines)):
        if is_org_line(lines[j]):
            first_org = lines[j].strip()
            first_org_idx = j
            break
    if first_org is None:
        end = min(len(lines), start + 120)  # fallback snapshot
        return start, end, None

    # 3) Sumário ends immediately before the NEXT occurrence of that same ORG line
    end = None
    for k in range(first_org_idx + 1, len(lines)):
        if lines[k].strip() == first_org:
            end = k
            break
    if end is None:
        # Fallback: next ORG-like line after first_org
        for k in range(first_org_idx + 1, len(lines)):
            if is_org_line(lines[k]):
                end = k
                break
    if end is None:
        end = len(lines)

    return start, end, first_org

# ---------------------------
# Sumário parsing (multiple ORGs, multiple ACTs) — line-by-line
# ---------------------------
def parse_sumario_items(sum_lines: List[str]) -> List[Dict]:
    items = []
    current_org = None
    i = 0
    while i < len(sum_lines):
        line = sum_lines[i]

        if is_org_line(line):
            current_org = line.strip()
            i += 1
            continue

        parsed = extract_act_from_line(line)
        if parsed:
            kind, number, year = parsed
            act_text = line.strip()  # keep for metadata

            # Optional short lead: next 1–3 non-empty lines, stopping at blank/next ORG/next ACT
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
                "kind": kind,
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
# Body anchoring (line-by-line with spaCy token parsing)
# ---------------------------
def extract_kind_num_year_from_line(text: str) -> Optional[Tuple[str, str, str]]:
    """Parse a BODY line. If it is the ACT header line, return (kind_lower, number, year)."""
    parsed = extract_act_from_line(text)
    if not parsed:
        return None
    kind, number, year = parsed
    return (kind.lower(), number, year)

def find_act_line_in_section(lines: List[str], start_idx: int, end_idx: int,
                             kind: str, number: str, year: str) -> Optional[int]:
    target_kind = kind.lower()
    target_num = number
    target_year = year
    for i in range(start_idx, end_idx):
        parsed = extract_kind_num_year_from_line(lines[i])
        if parsed:
            pk, pn, py = parsed
            if pk == target_kind and pn == target_num and py == target_year:
                return i
    return None

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

    # Copy original completo.txt into the extracted folder
    shutil.copy2(completo, out_dir / "completo.txt")

    lines = completo.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Sumário extraction
    sum_start, sum_end, _ = find_sumario_range(lines)
    sumario_raw = lines[sum_start:sum_end] if (sum_start is not None and sum_end is not None) else []
    (out_dir / "sumario_raw.txt").write_text("\n".join(sumario_raw), encoding="utf-8")

    # Sumário items
    items = parse_sumario_items(sumario_raw)
    (out_dir / "sumario_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    # ORGs in sumário (unique, in order)
    orgs_unique: List[str] = []
    for it in items:
        org = it.get("section")
        if org and org not in orgs_unique:
            orgs_unique.append(org)

    if DEBUG:
        print(f"\n=== {stem_dir.name} ===")
        rng = f"{sum_start}..{(sum_end-1) if (sum_start is not None and sum_end is not None) else 'N/A'}"
        print(f"Sumário range: {rng}")
        print("ORGs (from Sumário):")
        for o in orgs_unique:
            print("  -", o)
        print("ACTs (from Sumário, in order):")
        for idx, it in enumerate(items, 1):
            print(f"  {idx:02d}. [{it.get('section')}] {it['kind']} {it['number']}/{it['year']} | act_line='{it.get('act_text','')}'")

    # Body starts at sum_end (or 0 if no Sumário)
    body_start = sum_end if sum_end is not None else 0

    # Find each ORG position in the body (exact equality to known ORG lines)
    org_positions: List[Tuple[int, Optional[str]]] = []
    for i in range(body_start, len(lines)):
        if lines[i].strip() in orgs_unique:
            org_positions.append((i, lines[i].strip()))
    org_positions.append((len(lines), None))  # sentinel end

    if DEBUG:
        print("ORG positions in BODY:")
        for pos, name in org_positions[:-1]:
            print(f"  - line {pos}: {name}")

    index = {
        "pdf_stem": stem_dir.name,
        "source_completo": str(completo),
        "sumario_file": "sumario_raw.txt",
        "sumario_items_file": "sumario_items.json",
        "items": [],
        "unmatched": []
    }

    # Group items by ORG (Sumário order preserved)
    items_by_org: Dict[str, List[Dict]] = {}
    for it in items:
        org = it.get("section") or ""
        items_by_org.setdefault(org, []).append(it)

    # Filename generator (use original kind text; make filesystem-safe)
    used_names = set()
    def make_doc_filename(kind: str, number: str) -> str:
        base = f"{kind}-{number}.txt"
        safe = "".join(ch for ch in base if ch not in '\\/:*?"<>|')
        name = safe
        suffix = ord('a')
        while name in used_names:
            name = safe.replace(".txt", f"-{chr(suffix)}.txt")
            suffix += 1
        used_names.add(name)
        return name

    # Walk ORG sections in body order
    for idx in range(len(org_positions) - 1):
        org_start, org_name = org_positions[idx]
        org_end, _ = org_positions[idx + 1]
        if org_name is None:
            continue

        expected_items = items_by_org.get(org_name, [])
        if not expected_items:
            continue

        if DEBUG:
            print(f"\n-- ORG section: {org_name}  lines {org_start}..{org_end-1}")
            print(f"   expected ACTs in this ORG: {len(expected_items)}")

        # Find ACT anchors inside this ORG section
        act_starts: List[Tuple[int, Dict]] = []
        search_start = org_start + 1  # skip the ORG header line
        for it in expected_items:
            if DEBUG:
                print(f"   searching: {it['kind']} {it['number']}/{it['year']}")
            pos = find_act_line_in_section(lines, search_start, org_end, it["kind"], it["number"], it["year"])
            if DEBUG:
                if pos is not None:
                    print(f"     FOUND at line {pos}: {lines[pos].rstrip()!r}")
                else:
                    print("     NOT FOUND")
            if pos is not None:
                act_starts.append((pos, it))
            else:
                index["unmatched"].append({
                    "section": it.get("section"),
                    "kind": it["kind"],
                    "number": it["number"],
                    "year": it["year"],
                    "title": it.get("title", ""),
                    "reason": "anchor_not_found_in_body_section"
                })

        if not act_starts:
            continue

        act_starts.sort(key=lambda x: x[0])
        boundaries = [p for p, _ in act_starts] + [org_end]

        # Slice each doc by line ranges; prepend ORG label
        for aidx, (start_pos, it) in enumerate(act_starts):
            start = start_pos
            end = boundaries[aidx + 1]
            chunk_lines = lines[start:end]

            doc_lines = []
            if not chunk_lines or chunk_lines[0].strip() != org_name.strip():
                doc_lines.append(org_name.strip())
                doc_lines.append("")
            doc_lines.extend(chunk_lines)

            fname = make_doc_filename(it["kind"], it["number"])
            (docs_dir / fname).write_text("\n".join(doc_lines), encoding="utf-8")

            if DEBUG:
                print(f"   -> wrote {fname}  lines [{start}:{end})")

            index["items"].append({
                "section": it.get("section"),
                "kind": it["kind"],
                "number": it["number"],
                "year": it["year"],
                "title": it.get("title", ""),
                "doc_file": f"docs/{fname}",
                "body_line_range": [start, end - 1],
                "sumario_line_range": it.get("sumario_line_range"),
            })

    if DEBUG and index["unmatched"]:
        print(f"\nUNMATCHED ({len(index['unmatched'])}):")
        for u in index["unmatched"]:
            print(f"  - [{u.get('section')}] {u['kind']} {u['number']}/{u['year']}  reason={u['reason']}")

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
