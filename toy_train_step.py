# file: toy_train_step.py
import math, torch, torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from diffusion_collator import DiffusionSFTCollator

model_id = "Dream-org/Dream-v0-Instruct-7B"
tok = AutoTokenizer.from_pretrained("./dream7b_tok_with_mask", trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" else torch.float32
qconf = BitsAndBytesConfig(load_in_4bit=True,
                           bnb_4bit_use_double_quant=True,
                           bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16)
# 4bit QLoRA 加载
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=qconf
)
model.resize_token_embeddings(len(tok))          # 若我们新增了 <mask>，需要这句
model = prepare_model_for_kbit_training(model)
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate","up","down"]
)
model = get_peft_model(model, peft_cfg)
model.gradient_checkpointing_enable()
model.train()

ds = load_dataset("open-r1/s1K-1.1")["train"].select(range(128))  # 仅取少量样本做小步检查
collate = DiffusionSFTCollator(tokenizer=tok, max_seq_len=768)
opt = AdamW(model.parameters(), lr=2e-5)

def iter_batches(dataset, bs=1):
    buf = []
    for ex in dataset:
        buf.append(ex)
        if len(buf) == bs:
            yield collate(buf)
            buf = []
    if buf:
        yield collate(buf)

steps = 80
accum = 8
global_step = 0
opt.zero_grad()

for batch in iter_batches(ds, bs=1):
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits  # [B, L, V]
    labels = batch["labels"]
    active = labels != -100
    # 交叉熵只在被监督的位置上计算
    loss = F.cross_entropy(
        logits[active].to(torch.float32),   # CE 要 float32 以稳定
        labels[active],
    ) / accum
    loss.backward()

    if (global_step + 1) % accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()

    if (global_step + 1) % 10 == 0:
        print(f"step {global_step+1:4d} | loss {loss.item()*accum:.4f}")
    global_step += 1
    if global_step >= steps:
        break

# 保存 LoRA（只是验证跑通，Day1 不追求收敛）
model.save_pretrained("./dream7b_s1k_lora_toy")
print("Saved tiny LoRA adapter to ./dream7b_s1k_lora_toy")