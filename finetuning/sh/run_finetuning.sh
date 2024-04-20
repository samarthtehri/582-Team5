for DATA in ./Data/train/train.csv  # ./Data/train/train.csv data_augmentation/augmented_data/copy_augmentation.csv data_augmentation/augmented_data/crossing_augmentation.csv
do
    for MODEL in google-t5/t5-large google-t5/t5-3b # google-t5/t5-base google-bert/bert-base-cased FacebookAI/roberta-base FacebookAI/roberta-large google-bert/bert-large-cased
    do
        python finetuning/src/finetune.py --model_name $MODEL --train_file $DATA
    done
done

# # intent, category
# for DATA in ./Data/train/train.csv
# do
#     for MODEL in google-bert/bert-base-cased FacebookAI/roberta-base FacebookAI/roberta-large google-bert/bert-large-cased
#     do
#         python finetuning/src/finetune.py --model_name $MODEL --train_file $DATA --utterance_format user text intent
#         python finetuning/src/finetune.py --model_name $MODEL --train_file $DATA --utterance_format user text category
#     done
# done
