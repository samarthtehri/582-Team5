# cd ./prompting

for prompt in fewshot fewshot_cot fewshot_cot_intent fewshot_cot_category  # zeroshot
do
    for model in mistralai/Mistral-7B-Instruct-v0.1 mistralai/Mixtral-8x7B-Instruct-v0.1
    do
        python prompting/src/run_llms.py --model $model --llm_input_format user text --prompt_type $prompt
    done
done

python prompting/src/postprocess_and_get_performance.py
python prompting/src/generate_table.py
