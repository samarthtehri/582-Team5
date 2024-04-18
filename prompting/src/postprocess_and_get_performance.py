import json

from src.config import llms_list, input_formats_list
from src.path import llm_outputs_dir, performance_dir
from src.prompts import get_prompt_template
from utils.dataset_io import load_original_dataset
from src.utils.get_performance import get_performance


def postprocess_response(response_dict: dict, prompt_name: str) -> int:
    response = response_dict["response"]
    
    if prompt_name in ["zeroshot", "fewshot"]:
        raw_label = response[0]
    elif "fewshot_cot" in prompt_name:
        raw_label = response[-1]
    else:
        raise NotImplementedError(f"Postprocessing for prompt_name={prompt_name} is not implemented.")

    if raw_label in ["0", "1"]:
        label = int(raw_label)
    else:
        label = -1
    return label


if __name__ == "__main__":
    dataset = load_original_dataset("test")
    y_true = [int(data["label"]) for data in dataset]
    
    performance_dict = {}
    for prompt_name in get_prompt_template.keys():
        for input_format in input_formats_list:
            for model_name_long in llms_list:
                # postprocess
                model_name = model_name_long.split("/")[-1]
                responses_path = llm_outputs_dir / f"prompt={prompt_name}" / f"input_format={input_format}" / "responses" / f"{model_name}.jsonl"
                if not responses_path.exists():
                    continue
                
                with open(responses_path, "r") as f:
                    responses = [json.loads(line) for line in f]
                assert len(responses) == len(dataset)
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
