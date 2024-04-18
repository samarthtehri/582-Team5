from pathlib import Path


dataset_dir = Path("./Data")

dataset_stats_dir = Path("./dataset_stats")

prompting_dir = Path("./prompting")
llm_outputs_dir = prompting_dir / "llm_outputs"
performance_dir = prompting_dir / "performance"
table_dir = performance_dir / "tables"
