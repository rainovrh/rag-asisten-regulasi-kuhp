#!/usr/bin/env python3
"""Generate human evaluation template from golden dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import PROCESSED_DATA_DIR, LOGS_DIR, DATASETS_DIR
from src.config.settings import settings
from src.utils.logger import setup_logger
from src.evaluation.golden_dataset import GoldenDataset
import pandas as pd


def main() -> None:
    """Generate human evaluation template."""
    setup_logger("INFO")
    
    # Load dataset
    dataset_path = settings.golden_dataset_path
    dataset = GoldenDataset(dataset_path)
    
    # Sample 15-20% of scenarios (7-10 out of 50)
    sample_size = 10  # 20% of 50
    scenarios = dataset.scenarios[:sample_size]
    
    # Create template
    rows = []
    for scenario in scenarios:
        rows.append({
            "ID": scenario.id,
            "Query": scenario.query,
            "Konteks_Pasal": scenario.context,
            "Ground_Truth": scenario.ground_truth,
            "AI_Answer": "",  # To be filled after running evaluation
            "Akurasi_Hukum": "",
            "Kelengkapan_Situsasi": "",
            "Konsistensi_Logika": "",
            "Skor_Total": "",
            "Kesimpulan": "",
            "Catatan": "",
        })
    
    df = pd.DataFrame(rows)
    
    # Save template
    output_path = LOGS_DIR / "human_evaluation_template.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Human evaluation template saved to: {output_path}")
    print(f"Total scenarios: {len(df)}")
    print(f"\nSample queries:")
    for _, row in df.iterrows():
        print(f"  {row['ID']}. {row['Query'][:60]}...")


if __name__ == "__main__":
    main()
