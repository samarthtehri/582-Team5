import csv
import json

import numpy as np

from src.path import dataset_dir, dataset_stats_dir
from src.utils.get_performance import get_performance


if __name__ == "__main__":
    for split in ["test", "train"]:
        data_path = dataset_dir / split / f"{split}.csv"

        with open(data_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)

        # label distribution
        labels_list: list[int] = []
        for row in data:
            labels_list.append(int(row[0]))
        
        dataset_stats = {
            "summary": {
                "total_samples": len(labels_list),
            },
            "labels": {
                "count": {"0": labels_list.count(0), "1": labels_list.count(1)},
                "average": np.average(labels_list).item(),
            }
        }
        
        # baseline performance
        majority_baseline_predictions = [0 for _ in range(len(labels_list))]
        simple_baseline_performance = {
            "majority_class": get_performance(y_true=labels_list, y_pred=majority_baseline_predictions)
        }
        
        dataset_stats["baseline_performance"] = simple_baseline_performance
        
        # save
        dataset_stats_dir.mkdir(exist_ok=True, parents=True)
        with open(dataset_stats_dir / f"{split}.json", "w") as f:
            json.dump(dataset_stats, f, indent=4)
