"""Golden dataset management."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.utils.validators import validate_file_path

logger = get_logger(__name__)


@dataclass
class EvaluationScenario:
    """Single evaluation scenario from golden dataset."""
    id: int
    query: str
    context: str
    ground_truth: str


class GoldenDataset:
    """Manage golden dataset for evaluation."""
    
    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        """Initialize golden dataset.
        
        Args:
            dataset_path: Path to CSV dataset file.
        """
        self.scenarios: list[EvaluationScenario] = []
        
        if dataset_path is not None:
            self.load(dataset_path)
    
    def load(self, dataset_path: Path) -> None:
        """Load dataset from CSV.
        
        Args:
            dataset_path: Path to CSV file.
        
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If CSV format is invalid.
        """
        dataset_path = validate_file_path(dataset_path)
        logger.info(f"Loading golden dataset from {dataset_path}")
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Validate required columns
            required_cols = {"ID", "Query", "Konteks Pasal", "Ground Truth"}
            if not required_cols.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"CSV missing required columns: {required_cols - set(reader.fieldnames or [])}"
                )
            
            self.scenarios = []
            for row in reader:
                scenario = EvaluationScenario(
                    id=int(row["ID"]),
                    query=row["Query"],
                    context=row["Konteks Pasal"],
                    ground_truth=row["Ground Truth"],
                )
                self.scenarios.append(scenario)
        
        logger.info(f"Loaded {len(self.scenarios)} scenarios")
    
    def get_queries(self) -> list[str]:
        """Get all queries.
        
        Returns:
            List of query strings.
        """
        return [s.query for s in self.scenarios]
    
    def get_ground_truths(self) -> list[str]:
        """Get all ground truth answers.
        
        Returns:
            List of ground truth strings.
        """
        return [s.ground_truth for s in self.scenarios]
    
    def get_contexts(self) -> list[str]:
        """Get all expected contexts.
        
        Returns:
            List of context strings.
        """
        return [s.context for s in self.scenarios]
    
    def __len__(self) -> int:
        """Number of scenarios."""
        return len(self.scenarios)
    
    def __iter__(self):
        """Iterate over scenarios."""
        return iter(self.scenarios)
