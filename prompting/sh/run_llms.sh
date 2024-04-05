python src/run_llms.py --model mistralai/Mistral-7B-Instruct-v0.1 --llm_input_format user text --prompt_type zeroshot
python src/run_llms.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --llm_input_format user text --prompt_type zeroshot

python src/postprocess_and_get_performance.py
