# file: sweep_infer.py
import time, torch
from transformers import AutoTokenizer, AutoModel

CKPT = "./outputs_dream7b_s1k_qlora/epoch-3"  # 换成你选的最佳
BASE = "Dream-org/Dream-v0-Instruct-7B"
PROMPTS = [
    "一步得出答案：27*14=？",
    "一步得出答案：96*37=？",
    "把 144 分成两个质数之和，给一个例子并只输出表达式。"
]

tok  = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
model= AutoModel.from_pretrained(BASE, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
model.load_adapter(CKPT)

def gen(q, steps, alg):
    inp = tok.apply_chat_template([{"role":"user","content":q}], add_generation_prompt=True, return_tensors="pt").cuda()
    t0=time.time()
    with torch.no_grad():
        out = model.diffusion_generate(inp, attention_mask=torch.ones_like(inp),
                                       steps=steps, max_new_tokens=64, alg=alg)
    s = tok.decode(out[0], skip_special_tokens=True)
    return s.strip(), time.time()-t0

GRID = [(256,"entropy"),(384,"entropy"),(512,"entropy"),
        (256,"maskgit_plus"),(384,"maskgit_plus"),(512,"maskgit_plus")]

for q in PROMPTS:
    print(f"\nQ: {q}")
    for steps, alg in GRID:
        s, t = gen(q, steps, alg)
        print(f"[{alg} steps={steps}] {t:.2f}s  ->  {s}")
