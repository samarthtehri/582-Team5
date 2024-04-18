from pathlib import Path


dataset_dir = Path("./Data")

dataset_stats_dir = Path("./dataset_stats")

# prompting
prompting_dir = Path("./prompting")
llm_outputs_dir = prompting_dir / "llm_outputs"
prompting_performance_dir = prompting_dir / "performance"
prompting_table_dir = prompting_performance_dir / "tables"

# data augmentation
data_augmentation_dir = Path("./data_augmentation")
augmented_data_dir = data_augmentation_dir / "augmented_data"

# fine-tuning
finetuning_dir = Path("./finetuning")
finetuning_results_dir = finetuning_dir / "results"
finetuning_table_dir = finetuning_dir / "tables"
