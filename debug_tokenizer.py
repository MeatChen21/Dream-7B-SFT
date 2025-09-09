#!/usr/bin/env python3
# 调试tokenizer问题

from transformers import AutoTokenizer
from datasets import load_dataset

# 加载tokenizer
tok = AutoTokenizer.from_pretrained("./dream7b_tok_with_mask", trust_remote_code=True)

# 加载数据集
ds = load_dataset("open-r1/s1K-1.1")["train"]
ex = ds[0]
msgs = ex["messages"]

print("Messages type:", type(msgs))
print("First message type:", type(msgs[0]))
print("First message content:", msgs[0])

# 测试apply_chat_template
try:
    canvas = tok.apply_chat_template(
        msgs[:-1], add_generation_prompt=True, return_tensors=None
    )
    print("Canvas type:", type(canvas))
    print("Canvas content:", repr(canvas))
    
    # 测试修复后的逻辑
    if isinstance(canvas, list):
        # apply_chat_template已经返回了token IDs
        canvas_ids = canvas
        print("Using canvas directly as token IDs")
    else:
        # 如果是字符串，需要tokenize
        canvas_ids = tok(canvas, add_special_tokens=False).input_ids
        print("Tokenizing canvas string")
    
    print("Canvas IDs type:", type(canvas_ids))
    print("Canvas IDs length:", len(canvas_ids))
    
except Exception as e:
    print("Error in apply_chat_template:", e)
    import traceback
    traceback.print_exc()

# 测试target_ids
try:
    target_ids = tok(
        msgs[-1]["content"], add_special_tokens=False
    ).input_ids
    print("Target IDs type:", type(target_ids))
    print("Target IDs length:", len(target_ids))
except Exception as e:
    print("Error in target tokenization:", e)
    import traceback
    traceback.print_exc()