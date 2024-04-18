from tqdm import tqdm
import ast
import json

from tap import Tap
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.path import llm_outputs_dir
from src.prompts import get_prompt_template
from src.utils.load_csv_dataset import load_csv_dataset
from src.utils.preprocess_data import preprocess_utterance


def get_tokenizer(model_name: str) -> str:
    if model_name == "mosaicml/mpt-7b-instruct":
        return "EleutherAI/gpt-neox-20b"
    
    return model_name


def get_prompt(data, input_format: list[str], prompt_type: str):
    utterances = preprocess_utterance(data, input_format)
    
    prompt_template = get_prompt_template[prompt_type]
    prompt = prompt_template.format(input1=utterances["utterance1"], input2=utterances["utterance2"])
    
    if "category" in prompt_type:
        prompt += f"\nCategory: {data['category']}"
    
    return prompt


class LlmTap(Tap):
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    batch_size: int = 4
    
    llm_input_format: list[str] = ["user", "text"]
    prompt_type: str = "zeroshot"


if __name__ == "__main__":
    args = LlmTap().parse_args()
    
    print("load dataset")
    dataset = load_csv_dataset("test")

    print("load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(get_tokenizer(args.model_name), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("load model")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto', trust_remote_code=True)
    
    responses: list[str] = []
    idx_batches = [list(range(idx, min(idx + args.batch_size, len(dataset)))) for idx in range(0, len(dataset), args.batch_size)]
    for idx_batch in tqdm(idx_batches):
        batch = dataset.select(idx_batch)

        prompts = [get_prompt(b, input_format=args.llm_input_format, prompt_type=args.prompt_type) for b in batch]
        template_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False) for p in prompts]
        tokens = tokenizer(template_prompts, return_tensors="pt", padding=True).to(model.device)
        
        generated_ids = model.generate(**tokens, max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(tokens.input_ids, generated_ids)
        ]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        responses.extend([{"response": d} for d in decoded])

    output_dir = llm_outputs_dir / f"prompt={args.prompt_type}" / f"input_format={'-'.join(args.llm_input_format)}" / "responses"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{args.model_name.split('/')[-1]}.jsonl", "w") as f:
        for response in responses:
            f.write(json.dumps(response) + "\n")
