import json

from src.config import finetuned_models_list
from src.path import finetuning_results_dir, finetuning_table_dir


positive_labels_list = ["pos_label=0", "pos_label=1", "macro"]
metrics_list = ["precision", "recall", "f1"]

if __name__ == "__main__":
    finetuning_table_dir.mkdir(exist_ok=True, parents=True)

    table_list: list[str] = []
    for model_name_full in finetuned_models_list:
        table_list.append(model_name_full)
        model_name = model_name_full.split("/")[-1]

        first_row = [""] + [f"{positive_label} {metrics}" for positive_label in positive_labels_list for metrics in metrics_list]
        table_list.append(" & ".join(first_row) + " \\\\")
        for input_type in ["['user', 'text']", "['user', 'text', 'intent']", "['user', 'text', 'category']"]:
            row_list: list[str] = [input_type]
            
            with open(finetuning_results_dir / f"model={model_name},train=train.csv,input_format={input_type}/performance.json", "r") as f:
                performance = json.load(f)["eval_performance"]
            
            for positive_label in positive_labels_list:
                for metrics in ["precision", "recall", "f1"]:
                    value = performance[positive_label][metrics] * 100
                    row_list.append(f"{value:.0f}")
            table_list.append(" & ".join(row_list) + " \\\\")
        table_list.append("")
    
    # save
    with open(finetuning_table_dir / "input_type_table.txt", "w") as f:
        for line in table_list:
            f.write(line + "\n")
