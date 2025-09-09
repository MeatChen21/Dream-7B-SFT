# file: quick_infer.py
import torch
from transformers import AutoTokenizer, AutoModel

model_id = "Dream-org/Dream-v0-Instruct-7B"  # 也可以换成你本地权重目录
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id, dtype=dtype, trust_remote_code=True
).eval().to(device)

# 一个超短的算术题
messages = [{"role":"user","content":"一步得出答案：2+3*4=？只给结果。"}]
input_ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)

with torch.no_grad():
    # Dream 的扩散式解码接口（质量-速度由 steps 控制）
    out = model.diffusion_generate(
        inputs=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.float32),
        steps=256,                # 也可以先 128 更快
        max_new_tokens=64,
        temperature=0.2,
        alg="entropy"             # 生成顺序策略，可试 "maskgit_plus"
    )

print(tok.decode(out[0], skip_special_tokens=True))
