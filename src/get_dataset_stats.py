import json

from src.path import dataset_dir, dataset_stats_dir
from src.utils.get_performance import get_performance
from src.utils.dataset_io import load_csv_file_as_list


def get_dataset_stats(data_path: str) -> dict:
    data = load_csv_file_as_list(data_path)

    # label distribution
    labels_list: list[int] = []
    for row in data:
        labels_list.append(int(row[0]))
    
    dataset_stats = {
        "summary": {
            "total_samples": len(labels_list),
        },
    }
    
    # baseline performance
    majority_baseline_predictions = [0 for _ in range(len(labels_list))]
    simple_baseline_performance = {
        "majority_class": get_performance(y_true=labels_list, y_pred=majority_baseline_predictions)
    }
    
    dataset_stats["baseline_performance"] = simple_baseline_performance
    
    return dataset_stats


if __name__ == "__main__":
    for split in ["test", "train"]:
        data_path = dataset_dir / split / f"{split}.csv"
        dataset_stats = get_dataset_stats(data_path)
        
        # save
        dataset_stats_dir.mkdir(exist_ok=True, parents=True)
        with open(dataset_stats_dir / f"{split}.json", "w") as f:
            json.dump(dataset_stats, f, indent=4)
