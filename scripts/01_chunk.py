"""
scripts/01_chunk.py
-------------------
Reads the three clean text files (BNS, BNSS, BSA), splits each into
individual section chunks with contextual prefixes, and saves all
chunks to data/chunks.json.

Run once:  python scripts/01_chunk.py
"""

import json
import os
import re
import sys

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLEAN = os.path.join(BASE_DIR, "data", "clean")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "chunks.json")

# Map filename stems to official Act names
FILES = {
    "BNS_2023.txt": "BNS 2023",
    "BNSS_2023.txt": "BNSS 2023",
    "BSA_2023.txt": "BSA 2023",
}

# Regex that matches a section boundary line: digits + dot + space
SECTION_RE = re.compile(r"^(\d+)\.\s+(.+)")

# Regex that matches a chapter heading line
CHAPTER_RE = re.compile(r"^CHAPTER\s+[IVXLCDM0-9]+", re.IGNORECASE)

# Pattern to detect gazette footnote phantoms like "1. 1st July 2024"
PHANTOM_RE = re.compile(r"^\d+\.\s+\d+(st|nd|rd|th)\b", re.IGNORECASE)

# Regex to strip everything from the em-dash or period-dash onwards in a title
# Matches: .— or —— or –– or .( or just — at the first occurrence
TITLE_CLEAN_RE = re.compile(r"[.\s]*[—––]+.*$|\s*\.\s*—.*$|\s*\.—.*$|\s*\.\(.*$")


def clean_title(raw_title: str) -> str:
    """
    Extract only the clean short title from a section's first line.
    
    Examples:
        'Murder.—Except in the cases...' -> 'Murder'
        'Punishment for murder.—(1) Whoever...' -> 'Punishment for murder'
        'Short title, commencement and application.––(1) This Act...' -> 'Short title, commencement and application'
        'Definitions. –– In this Sanhita...' -> 'Definitions'
    """
    # Strip everything from the first em-dash variant onwards
    title = TITLE_CLEAN_RE.sub("", raw_title)
    # Also handle cases where the title ends with a period before the dash was stripped
    title = title.rstrip(". ")
    # If somehow we stripped everything, fall back to the original (cleaned)
    if not title.strip():
        title = raw_title.split(".")[0].split("—")[0].split("––")[0].strip()
    return title.strip()


def parse_sections(filepath: str, act_name: str) -> list[dict]:
    """Parse a single Act text file into section chunks."""
    with open(filepath, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    chunks: list[dict] = []
    current_chapter = "GENERAL"
    current_section_num = None
    current_section_title = None
    current_lines: list[str] = []

    def _flush():
        """Save the accumulated section as a chunk."""
        if current_section_num is None:
            return
        raw_text = "".join(current_lines).strip()
        if not raw_text:
            return

        # Build the contextual prefix
        act_short = act_name.split()[0]  # BNS / BNSS / BSA
        chunk_id = f"{act_short}_{current_section_num}"
        # Use the clean title (no leaked legal text) in the context prefix
        clean = clean_title(current_section_title)
        context_prefix = (
            f"[Context: This section is from {act_name}, "
            f"{current_chapter}. It covers Section "
            f"{current_section_num}: {clean}.]\n\n"
        )
        full_text = context_prefix + raw_text
        source_label = f"Section {current_section_num}, {act_name}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": full_text,
                "section_number": current_section_num,
                "section_title": clean_title(current_section_title),
                "chapter": current_chapter,
                "act": act_name,
                "source_label": source_label,
            }
        )

    for line in lines:
        stripped = line.strip()

        # ── detect chapter headings ──────────────────────────────────────
        if CHAPTER_RE.match(stripped):
            # Chapter heading: current line + possibly next line (title)
            current_chapter = stripped
            continue

        # If the previous line was a bare CHAPTER line, this line is the
        # chapter title — append it.
        if (
            current_chapter
            and not current_chapter.endswith(")")
            and stripped
            and not SECTION_RE.match(stripped)
            and not CHAPTER_RE.match(stripped)
            and "CHAPTER" not in stripped.upper()
            and not stripped[0].isdigit()
        ):
            # Heuristic: if current_chapter is just "CHAPTER XIV" with no
            # descriptive text after it, and this line looks like a title
            # (starts with upper-case letters, no digit), treat it as the
            # chapter title continuation.
            words = current_chapter.split()
            if len(words) <= 3:  # e.g. "CHAPTER XIV"
                current_chapter = current_chapter + " " + stripped
                continue

        # ── detect section boundaries ────────────────────────────────────
        sec_match = SECTION_RE.match(stripped)
        if sec_match:
            # Skip gazette footnote phantoms
            if PHANTOM_RE.match(stripped):
                continue

            # Flush the previous section
            _flush()

            current_section_num = sec_match.group(1)
            current_section_title = sec_match.group(2).rstrip(".")
            current_lines = [line]
            continue

        # ── accumulate content lines ─────────────────────────────────────
        if current_section_num is not None:
            current_lines.append(line)

    # Flush the last section in the file
    _flush()
    return chunks


def validate_counts(all_chunks: list[dict]):
    """Print section counts and warn if outside expected ranges."""
    counts: dict[str, int] = {}
    for c in all_chunks:
        counts[c["act"]] = counts.get(c["act"], 0) + 1

    expected = {
        "BNS 2023": (350, 365),
        "BNSS 2023": (525, 540),
        "BSA 2023": (165, 175),
    }

    print("\n-- Section Counts --")
    all_ok = True
    for act, (lo, hi) in expected.items():
        n = counts.get(act, 0)
        status = "OK" if lo <= n <= hi else "WARNING: OUT OF RANGE"
        if not (lo <= n <= hi):
            all_ok = False
        print(f"  {act}: {n}  {status}  (expected {lo}-{hi})")

    if all_ok:
        print("\nAll counts within expected ranges.")
    else:
        print("\nWARNING: Some counts are outside expected ranges - review the "
              "source files and regex patterns.")


def main():
    all_chunks: list[dict] = []

    for filename, act_name in FILES.items():
        filepath = os.path.join(DATA_CLEAN, filename)
        if not os.path.isfile(filepath):
            print(f"ERROR: File not found -> {filepath}")
            sys.exit(1)

        print(f"Parsing {filename} ...")
        chunks = parse_sections(filepath, act_name)
        print(f"  -> {len(chunks)} sections extracted")
        all_chunks.extend(chunks)

    validate_counts(all_chunks)

    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(all_chunks)} chunks -> {OUTPUT_PATH}")

    # Quick sanity print
    print("\n-- Sample Chunks --")
    for c in all_chunks[:2]:
        print(f"\n  [{c['chunk_id']}] {c['source_label']}")
        print(f"  Chapter: {c['chapter']}")
        print(f"  Text preview: {c['text'][:200]}...")


if __name__ == "__main__":
    main()
