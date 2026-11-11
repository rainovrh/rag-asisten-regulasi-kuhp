"""Enrich Golden Dataset with algorithmic Target Chunk ID mapping (Blok 5, Kotak 19)."""

import csv
import json
from pathlib import Path
import re
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import DATASETS_DIR, PROCESSED_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger("INFO")


def map_konteks_to_chunk_id(konteks_str: str, mapping: dict) -> str:
    """Algorithmically map legal article context string to exact chunk ID.

    Examples:
        'Pasal 1 ayat (1)' -> 'kuhp_pasal_1'
        'Pasal 4 huruf a' -> 'kuhp_pasal_4'
        'Pasal 46' -> 'kuhp_pasal_46'
    """
    m = re.search(r"pasal\s*(\d+)", konteks_str, re.IGNORECASE)
    if not m:
        logger.warning(f"Could not extract article number from: '{konteks_str}'")
        return ""

    num = m.group(1)
    key = f"pasal {num}"

    chunk_list = mapping.get(key, [])
    if chunk_list:
        # For single chunk or default article chunk
        return chunk_list[0]

    # Fallback to direct ID convention
    return f"kuhp_pasal_{num}"


def main() -> None:
    """Enrich golden dataset CSV with Target Chunk ID column."""
    csv_path = DATASETS_DIR / "golden_dataset_rag_hukum_indonesia_rev3.csv"
    mapping_path = PROCESSED_DATA_DIR / "pasal_to_chunks.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {csv_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Pasal mapping not found at {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "Target Chunk ID" not in fieldnames:
        fieldnames.append("Target Chunk ID")

    enriched_count = 0
    for row in rows:
        target_chunk_id = map_konteks_to_chunk_id(row["Konteks Pasal"], mapping)
        row["Target Chunk ID"] = target_chunk_id
        if target_chunk_id:
            enriched_count += 1

    # Overwrite CSV with new enriched column
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Successfully enriched {enriched_count}/{len(rows)} rows with 'Target Chunk ID' in {csv_path}")


if __name__ == "__main__":
    main()

