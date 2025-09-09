from transformers import AutoTokenizer, AutoModel
import torch

ckpt = "./outputs_dream7b_s1k_qlora/epoch-3"  # 换成你的本次输出目录
base_id = "Dream-org/Dream-v0-Instruct-7B"

tok  = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
base = AutoModel.from_pretrained(base_id, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
base.load_adapter(ckpt)

msgs = [{"role":"user","content":"一步得出答案：96*37=？只给结果。"}]
inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").cuda()
with torch.no_grad():
    out = base.diffusion_generate(inp, attention_mask=torch.ones_like(inp),
                                  steps=256, max_new_tokens=32, alg="entropy")
print(tok.decode(out[0], skip_special_tokens=True))
