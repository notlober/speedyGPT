# speedyGPT
speedrunning gpts for 2.00 loss!

this repo aims to get 2.00 language modeling cross entropy loss fastest way possible.

inspiration:

modded-nanogpt by keller jordan - main inspiration for this project

https://github.com/KellerJordan/modded-nanogpt

### Setup
to run current code:

```python
pip install --pre --upgrade torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install tiktoken datasets tqdm numpy huggingface-hub --no-deps # do not modify torch installation with --no-deps
huggingface-cli login --token hf_xxxx # your token here, to save checkpoint to hf
# modify code line at train_gpt.py: "repo_id="yourname/yourrepo" to your preferred hf repo
python data/cached_fineweb10B.py 8 # downloads only the first 800M training tokens to save time
torchrun --standalone --nproc_per_node=8 train_gpt.py # 8x h100s are required!
wget https://huggingface.co/yourname/yourrepo/blob/main/model_stepxxxx.pt # download model files from checkpoint, modify your name, your repo, model step for evals
python benchmark_hellaswag.py # run with 1 gpu eg. rtx 4090
```

current model size: 110m
current evals:
loss: 3.28
hellaswag: 0.3361

TODO's: merge benchmark and train_gpt model definition