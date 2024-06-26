module load python/3.12/3.12.2

```sh
pip install torch==2.3.0+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib lightning einops tensorboard
pip install ruff ipykernel
```


https://github.com/huggingface/transformers/blob/ab0f050b42d903f34d6eb97f3f8c0c07f0517ad2/src/transformers/models/vilt/convert_vilt_original_to_pytorch.py#L180
