cd prompting

conda env create -f environment.yml
conda activate cse582
conda install pytorch=2.2.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

export PYTHONPATH="./"
