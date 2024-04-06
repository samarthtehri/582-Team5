import json

from src.prompts import get_prompt_template
from src.path import performance_dir, table_dir
from src.config import input_formats_list, llms_list


positive_labels_list = ["pos_label=0", "pos_label=1", "macro"]
metrics_list = ["precision", "recall", "f1"]

if __name__ == "__main__":
    with open(performance_dir / "performance.json", "r") as f:
        performance_dict: dict = json.load(f)
    
    table_dir.mkdir(exist_ok=True, parents=True)
    
    for input_format in input_formats_list:
        for model_name_full in llms_list:
            model_name = model_name_full.split("/")[-1]
            table_list: list[str] = []
            
            # first row
            first_row = [""] + [f"{positive_label} {metrics}" for positive_label in positive_labels_list for metrics in metrics_list]
            table_list.append(" & ".join(first_row) + " \\\\")
            for prompt_name in get_prompt_template.keys():
                row_list: list[str] = [prompt_name]
                p = performance_dict[f"prompt={prompt_name}"][f"input_format={input_format}"][model_name]["performance"]
                for positive_label in positive_labels_list:
                    for metrics in ["precision", "recall", "f1"]:
                        value = p[positive_label][metrics] * 100
                        row_list.append(f"{value:.0f}")
                table_list.append(" & ".join(row_list) + " \\\\")
            
            output_path = table_dir / f"table_{input_format}_{model_name}.txt"
            with open(output_path, "w") as f:
                for line in table_list:
                    f.write(line + "\n")
