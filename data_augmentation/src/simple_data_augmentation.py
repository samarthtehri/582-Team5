import json
import random
import ast

from src.path import dataset_dir, augmented_data_dir
from src.utils.dataset_io import load_csv_file_as_list, dump_dataset_to_csv
from src.utils.get_performance import get_label_distrubiton


def save_augmented_data(data: list[list[str]], name: str):
    dump_dataset_to_csv(data, augmented_data_dir / f"{name}_augmentation.csv")
    
    stats = get_label_distrubiton([int(row[0]) for row in data])
    with open(augmented_data_dir / f"{name}_augmentation_stats.json", "w") as f:
        json.dump(stats, f, indent=4)


def get_user_pair(row):
    u1 = ast.literal_eval(row[2])
    u2 = ast.literal_eval(row[3])

    return f"{u1['user']}-{u2['user']}"


if __name__ == "__main__":
    original_train_data = load_csv_file_as_list(dataset_dir / "train" / "train.csv")
    
    original_labels = [int(row[0]) for row in original_train_data]
    original_label_distribution = get_label_distrubiton(original_labels)
    
    augmented_data_dir.mkdir(parents=True, exist_ok=True)
    
    # original
    label_1_data = [row for row in original_train_data if row[0] == "1"]
    num_label_0_and_1_diff = original_label_distribution["count"]["0"] - original_label_distribution["count"]["1"]
    
    # copy label 1 to make the dataset balanced
    copied_data = []
    for i in range(num_label_0_and_1_diff):
        copied_data.append(label_1_data[i % len(label_1_data)])

    copy_augmented = original_train_data + copied_data
    save_augmented_data(copy_augmented, "copy")
    
    # criss-crossing
    userpair_to_data_dict: dict[str, list[str]] = {}
    for row in label_1_data:
        userpair_to_data_dict.setdefault(get_user_pair(row), []).append(row[3])  # utterance 2
    
    crossing_data = []
    for idx in range(num_label_0_and_1_diff):
        row = label_1_data[idx % len(label_1_data)]
        userpair = get_user_pair(row)
        other_data = random.Random(idx).choice(userpair_to_data_dict[userpair])
        new_row = row[:3] + [other_data]
        
        copied_data.append(new_row)
    
    crossing_augmented = original_train_data + copied_data
    save_augmented_data(crossing_augmented, "crossing")
