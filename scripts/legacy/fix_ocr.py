"""Clean OCR artifacts from the KUHP corpus JSON."""

import json
import re
from pathlib import Path

CORPUS_PATH = Path("D:/Arsip Belajar di Itenas/Semester 7/Tugas Akhir/TA Rainova/File Hukum/data/processed/kuhp_bersih.json")

REPLACEMENTS = [
    # Header/footer OCR noise — "FRESIDEN <garbled> INDONESIA"
    (r'FRESIDEN\s+[^\s]*\.?[A-Z\.]*[^\s]*\s+INDONESIA', 'REPUBLIK INDONESIA'),
    (r'FRESIDEN\s+[^\s]*INDONESIA', 'REPUBLIK INDONESIA'),
    (r'FRESIDEN\s+REPUBUK\s+INDONESIA', 'REPUBLIK INDONESIA'),
    (r'FRESIDEN\s+REI\'UBUK\s+INDONESIA', 'REPUBLIK INDONESIA'),
    (r'FRESIDEN\s+REPUELIK\s+INDONESIA', 'REPUBLIK INDONESIA'),

    (r'PNES!DEN\s+REPTIEIJK\s+INDONESIA', 'REPUBLIK INDONESIA'),
    (r'PNES!DEN\s+REPTIEIJK', 'REPUBLIK'),

    (r'\|K\s+INDONESIA', 'REPUBLIK INDONESIA'),
    (r'\|K\s+REPUELIK\s+INDONESIA', 'REPUBLIK INDONESIA'),
    (r'\|K\s+INOONESIA', 'REPUBLIK INDONESIA'),

    (r'AT\s+REPUBUK\s+INDONESIA', 'AT REPUBLIK INDONESIA'),

    # "REPUBUK" → "REPUBLIK" (any remaining)
    (r'REPUBUK', 'REPUBLIK'),
    (r'REPUEUK', 'REPUBLIK'),
    (r'REPUBL', 'REPUBLIK'),
    (r'INOONESIA', 'INDONESIA'),

    # "rl ffitrEIEtrN" style line-number garbage before REPUBLIK
    (r'rl\s*ffitrEIEtrN\s+REPUBL', ' '),

    # Garbled sequences in Pasal 5, 71, etc.
    (r'\s*\.\.\.\s*[\.\s]*[A-Za-z\[\]\{\}]*trtr[A-Za-z\[\]\{\}]*\s*', ' '),
    (r'\{II\s*l-irfl\{rf:IrfilNlr\'trltrFlltr\s*', ' '),
    (r'l-irfl\{rf:IrfilNlr\'trltrFlltr', ' '),

    # "asal 5O9" / "asal 360" / "asal 496" / "asal 25O" → these are OCR of "Ayat (X)"
    # Pattern: "asal <digit>O<digit>" → remove
    (r'\s*asal\s+\d+O\d+\.{0,2}\s*', ' '),
    (r'\s*asal\s+\d+\.{0,2}\s*', ' '),

    # "Ayat(l)" → "Ayat (1)" (OCR: lowercase l → digit 1)
    (r'Ayat\s*\((l)\)', 'Ayat (1)'),
    (r'Ayat(l)', 'Ayat (1)'),
    # "Ayatl2l" → "Ayat (2)"
    (r'Ayatl(\d+)l', r'Ayat (\1)'),

    # Pasal numbering artifacts
    (r'Pasal(\d+)\s', r'Pasal \1 '),
    (r'Pasel(\d+)', r'Pasal \1'),

    # "SK No l6l357A" → "SK No. 161357-A" (OCR turned digits into l)
    (r'SK\s+No\s+l(\d+)l(\d+)l\s*([A-Z])', r'SK No. \g<1>\g<2>-\g<3>'),

    # "Tahwr" → "Tahun", "Tahw" → "Tahun"
    (r'Tahwr', 'Tahun'),
    (r'Tahw\s', 'Tahun '),

    # "lgtaFi" → likely "dikatai" or noise; these garble patterns — remove standalone
    (r'lgtaFi\s+', ' '),
    (r'daar\s+', ' '),

    # Multiple dots/spaces cleanup
    (r'\.{4,}', '.'),
    (r'  +', ' '),
]


def clean_text(text: str) -> str:
    for pattern, replacement in REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def main() -> None:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed_count = 0
    for key, value in data.items():
        if isinstance(value, str):
            original = value
            cleaned = clean_text(value)
            if cleaned != original:
                data[key] = cleaned
                fixed_count += 1

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Cleaned {fixed_count}/{len(data)} entries in {CORPUS_PATH.name}")


if __name__ == "__main__":
    main()
