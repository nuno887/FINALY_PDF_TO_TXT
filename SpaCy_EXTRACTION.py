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
# spaCy model + matchers (ORG only)
# ---------------------------
KIND_WORDS = [
    "despacho", "aviso", "declaração", "declaracao", "edital",
    "deliberação", "deliberacao", "contrato", "resolução", "resolucao",
    "revogação", "revogacao", "caducidade", "ato", "acto"
]
KIND_SET = set(KIND_WORDS)

# words we DO NOT treat as modifiers (common connectors/prepositions)
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
    # ORG: sequences of uppercase tokens; allow hyphenated too
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
def is_org_line(text: str) -> bool:
    """Decide if a single line is an ORG heading using spaCy matcher (no regex)."""
    t = text.strip()
    if not t or t == "Sumário":
        return False
    doc = NLP(t)
    return len(ORG_MATCHER(doc)) > 0

def norm_ws(s: str) -> str:
    """Normalize for matching only: collapse spaces and lowercase."""
    return " ".join(s.replace("\u00A0", " ").split()).lower()

def consume_org_block(lines: List[str], i: int) -> Tuple[str, int]:
    """
    Given index i at the start of an ORG line, consume subsequent uppercase/ORG-like continuation
    lines (typically the next line for wrapped headings) and return (combined_org_text, next_index).
    """
    parts = [lines[i].strip()]
    j = i + 1
    while j < len(lines):
        nxt = lines[j]
        if not nxt.strip():
            break
        # stop if the next line looks like an ACT header
        if extract_act_from_line(nxt):
            break
        # continue if the next line is also an ORG-like uppercase line
        if is_org_line(nxt):
            parts.append(nxt.strip())
            j += 1
            # usually ORG wraps to just the next line; cap at 3 lines just in case
            if len(parts) >= 3:
                break
            continue
        break
    return " ".join(parts), j

def digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())

# ---------------------------
# ACT parsing & detection (no matcher gate; header-only rule)
# ---------------------------
def parse_act_tokens(doc):
    """
    Extract (kind_base, kind_modifier, number, year, kind_full, last_idx_used) from ONE LINE doc.
    last_idx_used = index of the last token that is part of the header (usually the year or '215/2025' token).
    Kind must appear at the start (only punctuation allowed before it).
    """
    kind_base = None
    number = None
    year = None
    last_idx_used = -1

    # find base kind at the start (allow only punctuation before it)
    kind_idx = None
    for i, tok in enumerate(doc):
        if tok.text.lower() in KIND_SET:
            # ensure all tokens before are punctuation (no letters/digits)
            if all((t.is_punct or (not t.text.strip())) for t in doc[:i]):
                kind_base = tok.text
                kind_idx = i
                last_idx_used = i
            break
    if kind_idx is None:
        return None

    # collect modifier tokens between base kind and the number / 'n.º'
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

    # find number/year (robust to '215/2025' as a single token)
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
                # try '/ year'
                if j + 2 < len(doc) and doc[j+1].text == "/" and any(ch.isdigit() for ch in doc[j+2].text):
                    year = digits_only(doc[j+2].text)
                    last_idx_used = max(last_idx_used, j + 2)
                else:
                    # otherwise, next numeric token as year
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
    - kind at the start (no letters/digits before it, punctuation allowed),
    - number/year present,
    - and no alphabetic OR numeric tokens after the header (punctuation allowed).
    """
    t = text.strip()
    if not t or t == "Sumário":
        return None
    doc = NLP(t)

    parsed = parse_act_tokens(doc)
    if not parsed:
        return None
    kind_base, kind_modifier, number, year, kind_full, last_idx = parsed

    # Must be alone: after the last header token, there should be no letters/digits
    for tok in doc[last_idx + 1:]:
        if tok.is_alpha or any(ch.isdigit() for ch in tok.text):
            return None

    return (kind_base, kind_modifier, number, year, kind_full)

# ---------------------------
# Sumário detection (with multi-line ORG)
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

    # 2) first ORG block after "Sumário"
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
        end = min(len(lines), start + 120)  # fallback snapshot
        return start, end, None

    # 3) Sumário ends immediately before the NEXT occurrence of that same ORG (1–2 line window match)
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
        # Fallback: next ORG-like line after j
        k = next_after_org
        while k < len(lines):
            if is_org_line(lines[k]):
                end = k
                break
            k += 1
    if end is None:
        end = len(lines)

    return start, end, first_org

# ---------------------------
# Sumário parsing (multiple ORGs, multiple ACTs) — line-by-line, with multi-line ORG combine
# ---------------------------
def parse_sumario_items(sum_lines: List[str]) -> List[Dict]:
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
# Body anchoring (line-by-line with spaCy token parsing)
# ---------------------------
def extract_kind_num_year_from_line(text: str) -> Optional[Tuple[str, str, str, str, str]]:
    """Parse a BODY line. If it is the ACT header line, return tuple; else None."""
    return extract_act_from_line(text)

def find_act_line_in_section(lines: List[str], start_idx: int, end_idx: int,
                             kind_base: str, kind_modifier: str, number: str, year: str) -> Optional[int]:
    tgt_base = kind_base.lower()
    tgt_full = (kind_base + (" " + kind_modifier if kind_modifier else "")).strip().lower()
    for i in range(start_idx, end_idx):
        parsed = extract_kind_num_year_from_line(lines[i])
        if not parsed:
            continue
        pk_base, pk_mod, pn, py, pk_full = parsed
        if pn == number and py == year:
            # Accept if base kind matches (always), or full kind matches (when present)
            if pk_base.lower() == tgt_base or pk_full.lower() == tgt_full:
                return i
    return None

# ---------------------------
# ORG positions in body (1–2 line windows)
# ---------------------------
def find_org_positions_in_body(lines: List[str], start_idx: int, orgs: List[str]) -> List[Tuple[int, str]]:
    positions: List[Tuple[int, str]] = []
    norm_map = {norm_ws(o): o for o in orgs}
    norms = set(norm_map.keys())
    i = start_idx
    L = len(lines)
    while i < L:
        l1 = norm_ws(lines[i].strip())
        matched = None
        if l1 in norms:
            matched = norm_map[l1]
        elif i + 1 < L:
            l12 = norm_ws(lines[i].strip() + " " + lines[i+1].strip())
            if l12 in norms:
                matched = norm_map[l12]
        if matched:
            positions.append((i, matched))
        i += 1
    return positions

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

    # ORGs in sumário (unique, in order; canonical combined text)
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
            print(f"  {idx:02d}. [{it.get('section')}] {it['kind_full']} {it['number']}/{it['year']} | act_line='{it.get('act_text','')}'")

    # Body starts at sum_end (or 0 if no Sumário)
    body_start = sum_end if sum_end is not None else 0

    # Find each ORG position in the body with multi-line aware matching
    org_positions: List[Tuple[int, Optional[str]]] = find_org_positions_in_body(lines, body_start, orgs_unique)
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

    # Filename generator (use kind_full; make filesystem-safe)
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
                print(f"   searching: {it['kind_full']} {it['number']}/{it['year']}")
            pos = find_act_line_in_section(
                lines, search_start, org_end,
                it["kind_base"], it["kind_modifier"], it["number"], it["year"]
            )
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
                    "kind_full": it["kind_full"],
                    "number": it["number"],
                    "year": it["year"],
                    "title": it.get("title", ""),
                    "reason": "anchor_not_found_in_body_section"
                })

        if not act_starts:
            continue

        act_starts.sort(key=lambda x: x[0])
        boundaries = [p for p, _ in act_starts] + [org_end]

        # Slice each doc by line ranges; prepend ORG label (canonical combined text)
        for aidx, (start_pos, it) in enumerate(act_starts):
            start = start_pos
            end = boundaries[aidx + 1]
            chunk_lines = lines[start:end]

            doc_lines = []
            if not chunk_lines or norm_ws(chunk_lines[0]) != norm_ws(org_name):
                doc_lines.append(org_name)
                doc_lines.append("")
            doc_lines.extend(chunk_lines)

            fname = make_doc_filename(it["kind_full"], it["number"])
            (docs_dir / fname).write_text("\n".join(doc_lines), encoding="utf-8")

            if DEBUG:
                print(f"   -> wrote {fname}  lines [{start}:{end})")

            index["items"].append({
                "section": it.get("section"),
                "kind_base": it["kind_base"],
                "kind_modifier": it["kind_modifier"],
                "kind_full": it["kind_full"],
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
            print(f"  - [{u.get('section')}] {u['kind_full']} {u['number']}/{u['year']}  reason={u['reason']}")

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
