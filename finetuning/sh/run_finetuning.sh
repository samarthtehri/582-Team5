for DATA in data_augmentation/augmented_data/copy_augmentation.csv  # './Data/train/train.csv'
do
    for MODEL in google-bert/bert-base-cased google-bert/bert-large-cased FacebookAI/roberta-base FacebookAI/roberta-large
    do
        python finetuning/src/finetune.py --model_name $MODEL --train_file $DATA
    done
done
