from pathlib import Path
from typing import List, Tuple, Optional
import spacy
from spacy.matcher import Matcher

INPUT_ROOT = Path("output_EXTRACTED_DOCUMENTS")

# Try to load a Portuguese model (prefer large)
def load_pt_model():
    for name in ("pt_core_news_lg", "pt_core_news_md", "pt_core_news_sm"):
        try:
            return spacy.load(name)
        except Exception:
            continue
    raise RuntimeError("No Portuguese spaCy model found. Install pt_core_news_lg or similar.")

def build_matchers(nlp) -> Tuple[Matcher, Matcher]:
    org_matcher = Matcher(nlp.vocab)
    des_matcher = Matcher(nlp.vocab)

    # ORG pattern (user-provided idea, trimmed: spaces aren't tokens so we only check for uppercase tokens)
    # Accept 1+ uppercase tokens, optionally hyphen, then 1+ uppercase tokens (covers 'SECRETARIA REGIONAL ...', 'VICE-PRESIDÊNCIA ...')
    org_patterns = [
        [{"IS_UPPER": True, "OP": "+"}],  # one or more uppercase tokens
        [{"IS_UPPER": True, "OP": "+"}, {"TEXT": "-"}, {"IS_UPPER": True, "OP": "+"}],  # allow hyphenated
    ]
    for p in org_patterns:
        org_matcher.add("ORG_HEAD", [p])

    # DES/ACT patterns (user intent: header at start of line; we scan per-line so it's implicitly anchored)
    KIND_WORDS = [
        "despacho", "aviso", "declaração", "declaracao", "edital",
        "deliberação", "deliberacao", "contrato", "resolução", "resolucao",
        "revogação", "revogacao", "caducidade", "ato", "acto"
    ]

    # pattern with 'n.º' token (optional) then number '/' year (both LIKE_NUM)
    des_patterns = [
        [
            {"LOWER": {"IN": KIND_WORDS}},
            {"LOWER": {"IN": ["conjunto", "de", "da", "do", "retificação", "retificacao", "societário", "societario"]}, "OP": "*"},
            {"TEXT": {"IN": ["n.º", "nº", "n.o", "n.°"]}, "OP": "?"},
            {"LIKE_NUM": True},
            {"TEXT": "/", "OP": "?"},
            {"LIKE_NUM": True, "OP": "?"},
        ],
        # variant without the n.º token at all
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

def parse_act_tokens(doc) -> Optional[Tuple[str, str, str]]:
    """Extract (kind, number, year) from a line doc matched as DES_ACT, without regex."""
    # kind = first token that is in KIND_WORDS; number/year = first LIKE_NUM and next LIKE_NUM around optional '/'
    KIND_SET = {
        "despacho","aviso","declaração","declaracao","edital","deliberação","deliberacao",
        "contrato","resolução","resolucao","revogação","revogacao","caducidade","ato","acto"
    }
    kind = None
    num = None
    year = None
    i = 0
    while i < len(doc):
        t = doc[i]
        low = t.text.lower()
        if kind is None and low in KIND_SET:
            kind = t.text  # keep original case/accents as appears
        if num is None and t.like_num:
            num = t.text
            # look ahead for '/' and a following number as year
            if i + 2 < len(doc) and doc[i+1].text == "/" and doc[i+2].like_num:
                year = doc[i+2].text
            else:
                # fallback: next number token anywhere later
                j = i + 1
                while j < len(doc):
                    if doc[j].like_num:
                        year = doc[j].text
                        break
                    j += 1
        i += 1
    if kind and num and year:
        return (kind, num, year)
    return None

def scan_sumario_file(nlp, org_matcher: Matcher, des_matcher: Matcher, sumario_path: Path):
    lines = sumario_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    print(f"\n=== {sumario_path.parent.name} ===")
    print(f"Sumário lines: {len(lines)}")
    if not lines:
        print("  (empty sumário_raw.txt)")
        return

    orgs: List[Tuple[int, str]] = []
    acts: List[Tuple[int, str, str, str, str]] = []  # (line_idx, current_org, raw_line, kind, num/year)

    current_org: Optional[str] = None
    for idx, raw in enumerate(lines):
        text = raw.strip()
        if not text:
            continue
        if text == "Sumário":
            continue

        doc = nlp(text)
        # ORG detection
        org_matches = org_matcher(doc)
        if org_matches:
            current_org = text
            orgs.append((idx, current_org))
            continue

        # ACT detection
        des_matches = des_matcher(doc)
        if des_matches:
            parsed = parse_act_tokens(doc)
            if parsed:
                kind, num, yr = parsed
                acts.append((idx, current_org or "(no ORG set yet)", text, kind, f"{num}/{yr}"))
            else:
                acts.append((idx, current_org or "(no ORG set yet)", text, "(could not parse)", ""))

    print("ORGs detected:")
    if orgs:
        for i, o in orgs:
            print(f"  - line {i}: {o}")
    else:
        print("  (none)")

    print("ACTs detected (line index, ORG, raw line, parsed):")
    if acts:
        for i, org, raw, kind, numyr in acts:
            print(f"  - line {i}: [{org}] {raw}  -> {kind} {numyr}")
    else:
        print("  (none)")

def main():
    nlp = load_pt_model()
    org_matcher, des_matcher = build_matchers(nlp)

    root = INPUT_ROOT
    if not root.exists():
        print(f"Input folder not found: {root.as_posix()}")
        return
    found = 0
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        sf = sub / "sumario_raw.txt"
        if sf.exists():
            scan_sumario_file(nlp, org_matcher, des_matcher, sf)
            found += 1
    print(f"\nScanned {found} sumário files.")

if __name__ == "__main__":
    main()