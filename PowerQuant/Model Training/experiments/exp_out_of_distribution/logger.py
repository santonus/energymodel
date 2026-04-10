import json
import os
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.results_dir = Path(__file__).parent.parent.parent.parent / "Results"
        self.results_dir.mkdir(exist_ok=True)
        self.log_file = self.results_dir / f"{experiment_id}_results.jsonl"
    
    def log_result(self, config: dict, metrics: dict):
        """Log experiment results to JSONL file."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "experiment_id": self.experiment_id,
            "config": config,
            "metrics": metrics
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
        
        print(f"✓ Results logged to {self.log_file}")
