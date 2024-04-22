import csv
from pathlib import Path

import numpy as np


if __name__ == "__main__":
    annotated_data_dir = Path("task_ambiguity_analysis/annotated_data")
    file_names_list = ["task_ambiguity_ryo.csv", "task_ambiguity_vanshaj.csv"]
    
    ambiguity_label = []
    ground_truth_label = []
    for file_name in file_names_list:
        with open(annotated_data_dir / file_name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                ambiguity_label.append(row[0] == "a")
                ground_truth_label.append(row[1])
    
    overall = np.mean(ambiguity_label).item() * 100
    label_0 = np.mean([ambiguity_label[i] for i in range(len(ambiguity_label)) if ground_truth_label[i] == "0"]).item() * 100
    label_1 = np.mean([ambiguity_label[i] for i in range(len(ambiguity_label)) if ground_truth_label[i] == "1"]).item() * 100
    
    output_dir = Path("task_ambiguity_analysis")
    with open(output_dir / "task_ambiguity_analysis_results.txt", "w") as f:
        f.write("label=0 & label=1 & Overall\n")
        f.write(f"{label_0:.1f} & {label_1:.1f} & {overall:.1f}\n")
