"""
注意力权重可视化模块。

原理：对已生成的摘要 token 序列再做一次带 output_attentions=True 的前向传播，
取解码器交叉注意力（cross-attention）最后一层，对注意力头与解码步求平均，
得到每个源文本 token 的注意力权重，用于网页热力图展示模型"阅读"原文时
重点关注的字词。
"""
from typing import Dict, Any, List, Optional

import torch
from loguru import logger

from .model_manager import get_model_manager


def _clean_token(tok: str) -> str:
    """清理子词标记（mT5 的 ▁、WordPiece 的 ## 等）。"""
    return tok.replace("▁", "").replace("##", "").strip()


_SPECIAL_TOKENS = {"</s>", "<s>", "<pad>", "<unk>", "</S>", "<PAD>"}


def _force_eager_attention(model) -> None:
    """
    将模型及其全部子模块的注意力实现切换为 eager。

    transformers 5.x 默认用 sdpa 加速，但 sdpa 不返回注意力权重；
    encoder/decoder 子栈与主模型 config 同类，set_attn_implementation
    会跳过它们，因此这里直接对每个模块的 config 赋值（属性 setter 会
    递归设置子配置）。
    """
    seen = set()
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        try:
            if getattr(cfg, "_attn_implementation", None) != "eager":
                cfg._attn_implementation = "eager"
        except Exception:
            pass


def get_cross_attention(
    text: str,
    summary: str,
    model_name: Optional[str] = None,
    max_source_length: int = 256,
) -> Dict[str, Any]:
    """
    计算摘要对源文本的交叉注意力分布。

    Args:
        text: 原始新闻文本。
        summary: 已生成的摘要文本。
        model_name: 模型名称，为空使用默认模型。
        max_source_length: 编码源文本的最大 token 数（热力图不宜过长）。

    Returns:
        {"tokens": [{"token": str, "weight": float}], "model": str}
        weight 已归一化到 0~1。
    """
    manager = get_model_manager()
    model_name = model_name or manager.default_model
    manager._load_model(model_name)
    model = manager._models[model_name]
    tokenizer = manager._tokenizers[model_name]
    device = manager.device

    # 1. 编码源文本（截断到可视化友好长度）
    src = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_source_length,
        truncation=True,
    ).to(device)

    # 2. 编码摘要作为解码器输入（teacher forcing 一次前向即可拿到全部注意力）
    with tokenizer.as_target_tokenizer() if hasattr(tokenizer, "as_target_tokenizer") else _NullCtx():
        tgt = tokenizer(summary, return_tensors="pt", truncation=True, max_length=128)
    tgt = {k: v.to(device) for k, v in tgt.items()}
    # 追加 decoder 起始符
    dec_start = getattr(model.config, "decoder_start_token_id", tokenizer.pad_token_id)
    dec_input_ids = torch.cat(
        [torch.full((1, 1), dec_start, dtype=torch.long, device=device),
         tgt["input_ids"][:, :-1]], dim=1)

    # 3. 前向传播获取交叉注意力（先切换为 eager 注意力，sdpa 不返回权重）
    _force_eager_attention(model)
    with torch.no_grad():
        outputs = model(
            input_ids=src["input_ids"],
            attention_mask=src["attention_mask"],
            decoder_input_ids=dec_input_ids,
            output_attentions=True,
            use_cache=False,
        )

    cross = outputs.cross_attentions  # tuple(layer) of [1, heads, tgt, src]
    if not cross:
        raise RuntimeError("模型未返回交叉注意力权重")

    # 最后一层，对 heads 与解码步求均值 -> [src_len]
    last = cross[-1][0]                      # [heads, tgt, src]
    weights = last.mean(dim=(0, 1))          # [src]
    weights = weights / (weights.max() + 1e-9)

    # 4. 还原 token（过滤特殊符号与空 token）
    ids = src["input_ids"][0].tolist()
    tokens: List[Dict[str, Any]] = []
    for tid, w in zip(ids, weights.tolist()):
        tok = tokenizer.convert_ids_to_tokens([tid])[0]
        tok = _clean_token(str(tok))
        if not tok or tok in _SPECIAL_TOKENS:
            continue
        tokens.append({"token": tok, "weight": round(float(w), 4)})

    # 重新归一化（过滤后权重和可能小于 1）
    total = sum(t["weight"] for t in tokens) or 1.0
    for t in tokens:
        t["weight"] = round(t["weight"] / total, 4)

    logger.info(f"注意力可视化完成 | tokens={len(tokens)} | 模型={model_name}")
    return {"tokens": tokens, "model": model_name}


class _NullCtx:
    """兼容无 as_target_tokenizer 方法的分词器。"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
