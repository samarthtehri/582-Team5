import json

from src.path import dataset_dir, dataset_stats_dir
from src.utils.get_performance import get_performance
from src.utils.dataset_io import load_csv_file_as_list
from data_augmentation.src.data_augmentation import get_user_pair


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
    
    # user pair distribution
    user_pair_list = []
    for row in data:
        user_pair_list.append(get_user_pair(row))
    
    user_pair_distribution = {}
    for user_pair in user_pair_list:
        user_pair_distribution[user_pair] = user_pair_distribution.get(user_pair, 0) + 1
    
    label_distribution_by_user_pair = {}
    for i, user_pair in enumerate(user_pair_list):
        label = labels_list[i]
        label_distribution_by_user_pair.setdefault(user_pair, {"count": {"0": 0, "1": 0}})
        label_distribution_by_user_pair[user_pair]["count"][str(label)] += 1
    
    dataset_stats["user_pair"] = {
        "user_pair_distribution": user_pair_distribution,
        "label_distribution_by_user_pair": label_distribution_by_user_pair,
    }
        
    return dataset_stats


if __name__ == "__main__":
    for split in ["test", "train"]:
        data_path = dataset_dir / split / f"{split}.csv"
        dataset_stats = get_dataset_stats(data_path)
        
        # save
        dataset_stats_dir.mkdir(exist_ok=True, parents=True)
        with open(dataset_stats_dir / f"{split}.json", "w") as f:
            json.dump(dataset_stats, f, indent=4)
        
        # make a table of label_distribution_by_user_pair
        with open(dataset_stats_dir / f"label_distribution_by_user_pair_{split}.txt", "w") as f:
            f.write("user_pair & label_0 & label_1 & Num Data\n")
            for user_pair in ["Alien-Human", "Human-Alien", "Human-Human", "Alien-Alien"]:
                user_pair_num = dataset_stats["user_pair"]["user_pair_distribution"][user_pair]
                label_0_percentage = dataset_stats["user_pair"]["label_distribution_by_user_pair"][user_pair]["count"]["0"] / user_pair_num * 100
                label_1_percentage = dataset_stats["user_pair"]["label_distribution_by_user_pair"][user_pair]["count"]["1"] / user_pair_num * 100
                f.write(f"{user_pair} & {label_0_percentage:.1f} & {label_1_percentage:.1f} & {user_pair_num}\n")
