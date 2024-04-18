# CSE 582 Final Project - Team 5

CSE 582 Final Project. Dataset is from the source linked in canvas

## Environment

`setup.sh` includes the code to set up the environment. Make sure to update `PYTHONPATH`.

```sh
sh setup.sh
```

## Fine-Tuning

The [finetuning](./finetuning) directory includes code for fine-tuning language models on training data.

```sh
sh finetuning/sh/run_finetuning.sh
```

## LLM Prompting

In addition to fine-tuning language models on training data, we evaluate the performance of large language models (LLMs) with prompting, without fine-tuning.

The [prompting](./prompting) directory includes code for the experiments on LLMs.

```sh
sh prompting/sh/run_llms.sh
```

### Prompts

Prompts are defined in `prompting/src/prompts.py`. You can add new prompts and update `get_prompt_template` to use them in `prompting/src/run_llms.py`

Please refer to `sh prompting/sh/run_llms.sh` for how to use `prompting/src/run_llms.py`.

### Performance

`prompting/performance/performance.json` includes all performance.

`prompting/performance/tables` includes all tables.
