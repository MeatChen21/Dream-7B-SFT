# file: ensure_mask_token.py
import torch, json
from transformers import AutoTokenizer, AutoModel

model_id = "Dream-org/Dream-v0-Instruct-7B"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

added = False
if tok.mask_token_id is None:
    tok.add_special_tokens({"mask_token": "<mask>"})
    added = True
print("mask_token_id:", tok.mask_token_id, "| newly_added:", added)

# 仅在你需要“保存一个带mask的新tokenizer”时使用：
tok.save_pretrained("./dream7b_tok_with_mask")
print("Saved tokenizer to ./dream7b_tok_with_mask")
