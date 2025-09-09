# file: diffusion_collator.py
import random, torch
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DiffusionSFTCollator:
    tokenizer: Any
    max_seq_len: int = 1536
    mask_low: float = 0.3     # 每条样本随机掩码比例范围（可调）
    mask_high: float = 0.7

    def __post_init__(self):
        assert self.tokenizer.pad_token_id is not None, "tokenizer 需要有 pad_token"
        if self.tokenizer.mask_token_id is None:
            # 兜底：没有 mask_token 就退化用 pad 作为伪掩码（可运行，但建议添加真正的 mask）
            self.mask_id = self.tokenizer.pad_token_id
        else:
            self.mask_id = self.tokenizer.mask_token_id

    def _build_ids(self, msgs: List[Dict[str,str]]):
        # 画布：不含最后一条 assistant，且添加生成提示
        canvas = self.tokenizer.apply_chat_template(
            msgs[:-1], add_generation_prompt=True, return_tensors=None
        )
        # 兼容：有些 tokenizer 的 apply_chat_template 返回字符串，有些直接返回 token id 列表
        if isinstance(canvas, str):
            canvas_ids = self.tokenizer(canvas, add_special_tokens=False).input_ids
        elif isinstance(canvas, list):  # 视为已分好词的 token id 列表
            canvas_ids = canvas
        else:
            # 例如返回 torch.Tensor 的情况
            try:
                import torch as _torch
                if isinstance(canvas, _torch.Tensor):
                    canvas_ids = canvas.tolist()
                else:
                    raise TypeError
            except Exception:
                raise TypeError(f"apply_chat_template 返回了不支持的类型: {type(canvas)}")

        # 目标：最后一条 assistant
        target_text = msgs[-1]["content"]
        target_ids = self.tokenizer(
            target_text, add_special_tokens=False
        ).input_ids

        return canvas_ids, target_ids

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids, batch_labels, batch_attn = [], [], []
        for ex in features:
            msgs = ex["messages"]
            canvas_ids, target_ids = self._build_ids(msgs)

            # 采样掩码比例（近似映射离散时间步）
            ratio = random.uniform(self.mask_low, self.mask_high)
            k = max(1, int(len(target_ids) * ratio))
            # 随机选择 k 个位置掩码（也可按熵/置信度，Day1 先随机）
            mask_positions = set(random.sample(range(len(target_ids)), k))

            # 构造“被腐蚀”的输入序列：画布 + 掩码后的目标
            noised_target = [
                (self.mask_id if i in mask_positions else tid)
                for i, tid in enumerate(target_ids)
            ]
            input_ids = canvas_ids + noised_target

            # 构造 labels：画布区间 -100，目标区间为原 token
            labels = [-100] * len(canvas_ids) + target_ids

            # 截断（右侧截断）
            input_ids = input_ids[: self.max_seq_len]
            labels    = labels[: self.max_seq_len]

            attn_mask = [1] * len(input_ids)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attn.append(attn_mask)

        # 动态 padding
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(x) for x in batch_input_ids)

        def pad(seq, pad_val, L):
            return seq + [pad_val] * (L - len(seq))

        input_ids = torch.tensor([pad(x, pad_id, max_len) for x in batch_input_ids])
        labels    = torch.tensor([pad(x, -100,  max_len) for x in batch_labels])
        # 注意：模型期望 attention_mask 为 bool/float，避免 int64 触发 dtype 报错
        attn_mask = torch.tensor([pad(x, 0,     max_len) for x in batch_attn], dtype=torch.float32)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}
