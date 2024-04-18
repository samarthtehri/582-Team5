for MODEL in google-bert/bert-base-cased google-bert/bert-large-cased FacebookAI/roberta-base FacebookAI/roberta-large
do
    python finetuning/src/finetune.py --model_name $MODEL
done
