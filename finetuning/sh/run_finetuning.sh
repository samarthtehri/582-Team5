for DATA in ./Data/train/train.csv  # data_augmentation/augmented_data/copy_augmentation.csv
do
    for MODEL in FacebookAI/roberta-large google-bert/bert-large-cased  # google-bert/bert-base-cased FacebookAI/roberta-base
    do
        python finetuning/src/finetune.py --model_name $MODEL --train_file $DATA
    done
done
