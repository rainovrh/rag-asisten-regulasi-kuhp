"""Fix damage from overly-greedy OCR cleanup + remaining artifacts."""

import json
import re
from pathlib import Path

CORPUS_PATH = Path("D:/Arsip Belajar di Itenas/Semester 7/Tugas Akhir/TA Rainova/File Hukum/data/processed/kuhp_bersih.json")

def clean_entry(key: str, text: str) -> str:
    # Fix "REPUBLIKIK" and other double-K artifacts from greedy REPUBL replacement
    text = re.sub(r'REPUBLIKIK', 'REPUBLIK', text)
    text = re.sub(r'INOONESIA', 'INDONESIA', text)

    # Fix "P " prefix: "asal" was stripped from "Pasal" leaving just "P"
    # Reconstruct "Pasal <number>" from the key name
    m = re.match(r'Pasal\s+(\d+)', key)
    if m and text.startswith('P '):
        text = f"Pasal {m.group(1)} {text[2:]}"
    elif m and (text == 'P' or text == 'P ' or text == 'P.'):
        text = f"Pasal {m.group(1)} Cukup jelas."

    # Remove l-irfl{rf... garbage in Pasal 71
    text = re.sub(r'\{II\s*l-irfl\{rf:IrfilNlr\'trltrFlltr\s*', '', text)
    text = re.sub(r'\s*l-irfl\{rf:IrfilNlr\'trltrFlltr\s*', '', text)

    # Remove remaining FRESIDEN garbage (e.g. "PasaJ492.. .  FRESIDEN REI'UELIK INOONESIA")
    text = re.sub(r'\s*FRESIDEN\s+[^\s]*\s+[^\s]*\s*INDONESIA', ' REPUBLIK INDONESIA', text)
    text = re.sub(r'\s*\.\.\.\s*FRESIDEN\s+[^\s]*\s+[^\s]*\s*INDONESIA', ' ', text)

    # Clean up any leftover "REPUBLREPUBLIK" → REPUBLIK (was partially fixed)
    text = re.sub(r'REPUBL\s*REPUBLIK', 'REPUBLIK', text)

    # Remove any dangling "P " if a Pasal number was partially stripped
    # e.g., "P Ketentuan" → "Pasal X Ketentuan"
    if m:
        pasal_num = m.group(1)
        if re.match(r'^P\s+[A-Z]', text):
            text = f"Pasal {pasal_num} {text[3:]}"

    # Clean multiple spaces
    text = re.sub(r'  +', ' ', text)

    return text.strip()


def main() -> None:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed_count = 0
    for key, value in data.items():
        if isinstance(value, str):
            original = value
            cleaned = clean_entry(key, value)
            if cleaned != original:
                data[key] = cleaned
                fixed_count += 1

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Fixed {fixed_count}/{len(data)} entries")


if __name__ == "__main__":
    main()
