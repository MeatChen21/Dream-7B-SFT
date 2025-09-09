# file: test_collator.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from diffusion_collator import DiffusionSFTCollator

model_id = "Dream-org/Dream-v0-Instruct-7B"
tok = AutoTokenizer.from_pretrained("./dream7b_tok_with_mask", trust_remote_code=True)

ds = load_dataset("open-r1/s1K-1.1")["train"].select(range(4))  # 取4条做测试
collate_fn = DiffusionSFTCollator(tokenizer=tok, max_seq_len=768)

batch = collate_fn([ds[i] for i in range(len(ds))])
for k, v in batch.items():
    print(k, v.shape, v.dtype)

# 解码看一个样本的“被掩码输入”
i = 0
print("\n[INPUT (with masks)]")
print(tok.decode([x for x in batch["input_ids"][i].tolist() if x != tok.pad_token_id]))
print("\n[Non-ignored labels count]:", (batch["labels"][i] != -100).sum().item())