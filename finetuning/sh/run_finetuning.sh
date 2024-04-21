for DATA in ./Data/train/train.csv data_augmentation/augmented_data/copy_augmentation.csv data_augmentation/augmented_data/crossing_augmentation.csv data_augmentation/augmented_data/human_annotation_augmentation.csv
do
    for MODEL in google-bert/bert-base-cased google-bert/bert-large-cased FacebookAI/roberta-base FacebookAI/roberta-large google-t5/t5-base google-t5/t5-large google-t5/t5-3b
    do
        python finetuning/src/finetune.py --model_name $MODEL --train_file $DATA --utterance_format text
    done
done

# intent, category
for DATA in ./Data/train/train.csv
do
    for MODEL in google-bert/bert-base-cased google-bert/bert-large-cased FacebookAI/roberta-base FacebookAI/roberta-large
    do
        python finetuning/src/finetune.py --model_name $MODEL --train_file $DATA --utterance_format user text
        python finetuning/src/finetune.py --model_name $MODEL --train_file $DATA --utterance_format text intent
        python finetuning/src/finetune.py --model_name $MODEL --train_file $DATA --utterance_format text category
    done
done
