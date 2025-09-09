# -*- coding: utf-8 -*-
"""
Dream-7B + S1K-1.1 扩散式 SFT（QLoRA）
- 单卡 32GB 友好
- 断点续训 / 定步评估 / 保存 LoRA
- 只在“被掩码的目标位”计算交叉熵
"""

import os, math, random, argparse, json, time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from datasets import load_dataset

from transformers import (
    AutoTokenizer, AutoModel,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,
    set_seed
)

from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training
)

# ----------------------------
# 训练配置（集中管理超参）
# ----------------------------
@dataclass
class TrainConfig:
    # 基础
    model_id: str = "Dream-org/Dream-v0-Instruct-7B"
    tokenizer_dir: str = "./dream7b_tok_with_mask"  # Day-1 生成过；若无可改为 model_id
    output_dir: str = "./outputs_dream7b_s1k_qlora"

    # 数据
    dataset_name: str = "open-r1/s1K-1.1"
    train_split: str = "train"
    val_ratio: float = 0.05       # 从训练集中切一小块做验证
    max_seq_len: int = 1536

    # 掩码噪声（离散扩散的“时间步”近似）
    mask_low: float = 0.3
    mask_high: float = 0.7

    # 训练
    seed: int = 42
    epochs: int = 3
    per_device_batch_size: int = 1
    grad_accum: int = 16
    lr: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    scheduler_type: str = "cosine"   # or "linear"
    log_every: int = 10
    eval_every: int = 40
    save_every: int = 400
    eval_max_batches: int = 50       # 验证集最多跑这么多 batch（节省时间）

    # QLoRA / 精度
    load_in_4bit: bool = True
    compute_dtype: str = "bfloat16"  # "bfloat16" or "float16"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None  # 默认下方赋值
    gradient_checkpointing: bool = True

    # 断点续训
    resume_from: Optional[str] = None  # e.g. "./outputs_dream7b_s1k_qlora/checkpoint-1200"

    # 推理抽样（评估时打印样例）
    sample_steps: int = 256
    sample_max_new_tokens: int = 64
    sample_alg: str = "entropy"

    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj","k_proj","v_proj","o_proj","gate","up","down"]


# ----------------------------
# 离散掩码式 Collator（核心）
# ----------------------------
class DiffusionSFTCollator:
    def __init__(self, tokenizer, max_seq_len=1536, mask_low=0.3, mask_high=0.7):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_low = mask_low
        self.mask_high = mask_high
        assert self.tok.pad_token_id is not None, "tokenizer需要有pad_token"
        self.mask_id = self.tok.mask_token_id if self.tok.mask_token_id is not None else self.tok.pad_token_id

    def _build_ids(self, msgs):
        # 画布：不含最后一条 assistant；添加生成提示
        canvas = self.tok.apply_chat_template(
            msgs[:-1], add_generation_prompt=True, return_tensors=None
        )
        
        # 检查canvas的类型，如果是token IDs列表则直接使用，否则需要tokenize
        if isinstance(canvas, list):
            # apply_chat_template已经返回了token IDs
            canvas_ids = canvas
        else:
            # 如果是字符串，需要tokenize
            canvas_ids = self.tok(canvas, add_special_tokens=False).input_ids

        # 目标：最后一条 assistant 文本
        target_ids = self.tok(
            msgs[-1]["content"], add_special_tokens=False
        ).input_ids
        return canvas_ids, target_ids

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attn = [], [], []
        for ex in features:
            msgs = ex["messages"]
            canvas_ids, target_ids = self._build_ids(msgs)

            # 采样掩码比例（近似时间步t）
            ratio = random.uniform(self.mask_low, self.mask_high)
            k = max(1, int(len(target_ids) * ratio))
            mask_positions = set(random.sample(range(len(target_ids)), k))

            # 被腐蚀的目标
            noised_target = [ (self.mask_id if i in mask_positions else tid)
                              for i, tid in enumerate(target_ids) ]

            _input = canvas_ids + noised_target
            # _labels = [-100]*len(canvas_ids) + target_ids
            labels_target = [target_ids[i] if i in mask_positions else -100 for i in range(len(target_ids))]
            _labels = [-100]*len(canvas_ids) + labels_target

            # 右截断
            _input  = _input[: self.max_seq_len]
            _labels = _labels[: self.max_seq_len]
            _attn   = [1]*len(_input)

            input_ids.append(_input)
            labels.append(_labels)
            attn.append(_attn)

        pad_id = self.tok.pad_token_id
        L = max(len(x) for x in input_ids)
        def pad(seq, pad_val): return seq + [pad_val]*(L-len(seq))

        input_ids = torch.tensor([pad(x, pad_id) for x in input_ids])
        labels    = torch.tensor([pad(x, -100)  for x in labels])
        attn      = torch.tensor([pad(x, 0)     for x in attn], dtype=torch.float32)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


# ----------------------------
# 评估：只在被监督位计算loss/acc
# ----------------------------
@torch.no_grad()
def evaluate(model, tok, loader, device, max_batches=50):
    model.eval()
    n, tot_loss, tot_correct, tot_count = 0, 0.0, 0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k,v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits      # [B, L, V]
        labels = batch["labels"] # [B, L]
        active = labels != -100

        loss = F.cross_entropy(
            logits[active].to(torch.float32), labels[active]
        )

        # 准确率：只统计被监督位置
        preds = logits.argmax(dim=-1)
        correct = (preds[active] == labels[active]).sum().item()
        count = active.sum().item()

        tot_loss += loss.item()
        tot_correct += correct
        tot_count += count
        n += 1

    model.train()
    avg_loss = tot_loss / max(1, n)
    acc = tot_correct / max(1, tot_count)
    return {"val_loss": avg_loss, "val_acc": acc}


# ----------------------------
# 打印一个解码样例（可选）
# ----------------------------
@torch.no_grad()
def sample_generation(model, tok, device, question_text, steps=256, max_new_tokens=64, alg="entropy"):
    msgs = [{"role":"user","content": question_text}]
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
    out = model.diffusion_generate(
        input_ids=inp,
        attention_mask=torch.ones_like(inp),
        steps=steps,
        max_new_tokens=max_new_tokens,
        alg=alg
    )
    return tok.decode(out[0], skip_special_tokens=True)


# ----------------------------
# 数据切分
# ----------------------------
def split_dataset(ds, val_ratio, seed):
    N = len(ds)
    n_val = max(1, int(N * val_ratio))
    idx = list(range(N))
    random.Random(seed).shuffle(idx)
    val_idx = set(idx[:n_val])
    train = ds.select([i for i in range(N) if i not in val_idx])
    val   = ds.select([i for i in range(N) if i in val_idx])
    return train, val


# ----------------------------
# 主训练函数
# ----------------------------
def main(cfg: TrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 0) 随机种子
    set_seed(cfg.seed)

    # 1) Tokenizer & Dataset
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_dir, trust_remote_code=True)
    if tok.mask_token_id is None:
        # 兜底：没有<mask>也能训练，但建议先补<mask>
        print("[WARN] tokenizer has no mask_token; will use pad_token as pseudo-mask.")
    ds_all = load_dataset(cfg.dataset_name)[cfg.train_split]
    train_ds, val_ds = split_dataset(ds_all, cfg.val_ratio, cfg.seed)

    # 2) Collator / DataLoader
    collate = DiffusionSFTCollator(
        tokenizer=tok, max_seq_len=cfg.max_seq_len,
        mask_low=cfg.mask_low, mask_high=cfg.mask_high
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.per_device_batch_size,
        shuffle=True, collate_fn=collate,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.per_device_batch_size,
        shuffle=False, collate_fn=collate,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

    # 3) QLoRA 加载
    compute_dtype = torch.bfloat16 if cfg.compute_dtype == "bfloat16" else torch.float16
    qconf = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModel.from_pretrained(
        cfg.model_id,
        trust_remote_code=True,
        quantization_config=qconf
    )
    # 若 tokenizer 新增过 <mask>，需要扩展 embedding
    model.resize_token_embeddings(len(tok))
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules
    )
    model = get_peft_model(model, lora_cfg)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False  # 与 checkpointing 兼容
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    # 4) Optimizer & Scheduler
    #   只对可训练参数（LoRA）优化
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = math.ceil(len(train_loader) / cfg.grad_accum) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    if cfg.scheduler_type == "linear":
        sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 5) 断点续训
    global_step = 0
    start_epoch = 0
    if cfg.resume_from:
        # 简化处理：从保存目录加载 LoRA 权重
        print(f"[INFO] Resuming from {cfg.resume_from}")
        model.load_adapter(cfg.resume_from, adapter_name="default")
        # 也可以在保存时额外 dump optim/sched 状态并在此恢复

    # 6) 训练循环
    scaler = GradScaler(device="cuda", enabled=(compute_dtype==torch.float16))
    running_loss = 0.0
    for epoch in range(start_epoch, cfg.epochs):
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k,v in batch.items()}

            with autocast(device_type="cuda", enabled=(compute_dtype==torch.float16)):
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = out.logits
                labels = batch["labels"]
                active = labels != -100
                loss = F.cross_entropy(
                    logits[active].to(torch.float32), labels[active]
                )

            loss = loss / cfg.grad_accum
            running_loss += loss.item()

            # backward
            if compute_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # step
            if (i + 1) % cfg.grad_accum == 0:
                if cfg.max_grad_norm is not None:
                    if compute_dtype == torch.float16:
                        scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                if compute_dtype == torch.float16:
                    scaler.step(optim); scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                sched.step()

                global_step += 1

                # logging
                if global_step % cfg.log_every == 0:
                    avg_loss = running_loss * cfg.grad_accum / cfg.log_every
                    print(f"[epoch {epoch+1}] step {global_step} | lr {sched.get_last_lr()[0]:.2e} | loss {avg_loss:.4f}")
                    running_loss = 0.0

                # eval
                if global_step % cfg.eval_every == 0:
                    metrics = evaluate(model, tok, val_loader, device, max_batches=cfg.eval_max_batches)
                    print(f"[eval] step {global_step} | val_loss {metrics['val_loss']:.4f} | val_acc {metrics['val_acc']:.4f}")
                    # 可选：打印一个样例
                    try:
                        demo_q = "一步得出答案：27*14 = ？只给结果。"
                        ans = sample_generation(model, tok, device, demo_q,
                                                steps=cfg.sample_steps,
                                                max_new_tokens=cfg.sample_max_new_tokens,
                                                alg=cfg.sample_alg)
                        print("[sample]", ans.replace("\n"," ")[:200])
                    except Exception as e:
                        print("[sample skipped]", e)

                # save
                if global_step % cfg.save_every == 0:
                    ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tok.save_pretrained(ckpt_dir)  # 方便推理端加载一致的 tokenizer
                    print(f"[save] saved LoRA adapter to {ckpt_dir}")

        # 每个 epoch 末也保存一次
        ep_dir = os.path.join(cfg.output_dir, f"epoch-{epoch+1}")
        os.makedirs(ep_dir, exist_ok=True)
        model.save_pretrained(ep_dir)
        tok.save_pretrained(ep_dir)
        print(f"[save] epoch {epoch+1} saved to {ep_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 可通过命令行覆盖关键超参
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--scheduler_type", type=str, default=None, choices=["cosine","linear"])
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.output_dir: cfg.output_dir = args.output_dir
    if args.epochs: cfg.epochs = args.epochs
    if args.per_device_batch_size: cfg.per_device_batch_size = args.per_device_batch_size
    if args.grad_accum: cfg.grad_accum = args.grad_accum
    if args.lr: cfg.lr = args.lr
    if args.max_seq_len: cfg.max_seq_len = args.max_seq_len
    if args.scheduler_type: cfg.scheduler_type = args.scheduler_type
    if args.resume_from: cfg.resume_from = args.resume_from

    main(cfg)
