import json


from src.path import dataset_dir, augmented_data_dir
from src.utils.dataset_io import load_csv_file_as_list, dump_dataset_to_csv
from src.utils.get_performance import get_label_distrubiton


if __name__ == "__main__":
    original_train_data = load_csv_file_as_list(dataset_dir / "train" / "train.csv")
    
    original_labels = [int(row[0]) for row in original_train_data]
    original_label_distribution = get_label_distrubiton(original_labels)
    
    augmented_data_dir.mkdir(parents=True, exist_ok=True)
    
    # copy label 1 to make the dataset balanced
    label_1_data = [row for row in original_train_data if row[0] == "1"]
    label_0_and_1_diff = original_label_distribution["count"]["0"] - original_label_distribution["count"]["1"]
    
    copied_data = []
    for i in range(label_0_and_1_diff):
        copied_data.append(label_1_data[i % len(label_1_data)])

    copy_augmented = original_train_data + copied_data
    dump_dataset_to_csv(copy_augmented, augmented_data_dir / "copy_augmentation.csv")
    
    stats = get_label_distrubiton([int(row[0]) for row in copy_augmented])
    with open(augmented_data_dir / "copy_augmentation_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
