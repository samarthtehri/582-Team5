for DATA in data_augmentation/augmented_data/crossing_augmentation.csv  # ./Data/train/train.csv data_augmentation/augmented_data/copy_augmentation.csv
do
    for MODEL in google-bert/bert-base-cased FacebookAI/roberta-base FacebookAI/roberta-large google-bert/bert-large-cased
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
