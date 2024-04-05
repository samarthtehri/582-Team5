import json

from src.config import llms_list
from src.path import llm_outputs_dir, performance_dir
from src.prompts import get_prompt_template
from src.utils.load_csv_dataset import load_csv_dataset
from src.utils.get_performance import get_performance


def postprocess_response(response_dict: dict, prompt_name: str) -> int:
    response = response_dict["response"]
    
    if prompt_name == "zeroshot":
        label = response[0]
        if label in ["0", "1"]:
            label = int(label)
        else:
            label = -1
        return label
    else:
        raise NotImplementedError(f"Postprocessing for prompt_name={prompt_name} is not implemented.")


if __name__ == "__main__":
    dataset = load_csv_dataset("test")
    y_true = [int(data["label"]) for data in dataset]
    
    performance_dict = {}
    for prompt_name in get_prompt_template.keys():
        for input_format in ["user-text"]:
            for model_name_long in llms_list:
                # postprocess
                model_name = model_name_long.split("/")[-1]
                responses_path = llm_outputs_dir / f"prompt={prompt_name}" / f"input_format={input_format}" / "responses" / f"{model_name}.jsonl"
                if not responses_path.exists():
                    continue
                
                with open(responses_path, "r") as f:
                    responses = [json.loads(line) for line in f]
                processed = [postprocess_response(response, prompt_name) for response in responses]
                
                output_path = llm_outputs_dir / f"prompt={prompt_name}" / f"input_format={input_format}" / "processed" / f"{model_name}.txt"
                output_path.parent.mkdir(exist_ok=True, parents=True)
                with open(output_path, "w") as f:
                    for label in processed:
                        f.write(f"{label}\n")

                # calculate performance
                performance_dict.setdefault(f"prompt={prompt_name}", {}).setdefault(f"input_format={input_format}", {})[
                    model_name] = get_performance(y_true=y_true, y_pred=processed)
    
    performance_dir.mkdir(exist_ok=True, parents=True)
    with open(performance_dir / "performance.json", "w") as f:
        json.dump(performance_dict, f, indent=4)
