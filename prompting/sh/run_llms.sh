# cd ./prompting

for prompt in zeroshot fewshot fewshot_cot
do
    for model in mistralai/Mistral-7B-Instruct-v0.1 mistralai/Mixtral-8x7B-Instruct-v0.1
    do
        python src/run_llms.py --model $model --llm_input_format user text --prompt_type $prompt
    done
done

python src/postprocess_and_get_performance.py
python src/generate_table.py
