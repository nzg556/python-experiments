"""
模型训练脚本：在中文新闻摘要数据集上微调 mT5 / Pegasus 等 Transformer 模型。

基于 HuggingFace Trainer API，支持分布式训练与混合精度。

使用方法：
    python models/train.py \
        --model_name csebuetnlp/mT5_multilingual_XLSum \
        --train_file data/processed/lcsts_clean.json \
        --output_dir models/checkpoints/mt5-finetuned \
        --num_train_epochs 3 \
        --batch_size 4
"""
import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)


def load_data(path: str) -> Dataset:
    """加载 JSON 格式数据并转换为 HuggingFace Dataset。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=128):
    """数据预处理：tokenize 原文与摘要。"""
    inputs = [doc for doc in examples["content"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=examples["summary"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    """计算 ROUGE 指标（训练过程中的评估）。"""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # 将 -100 替换为 pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 简易字符级 ROUGE-L F1
    def rouge_l_f1(pred, ref):
        pred_chars = list(pred)
        ref_chars = list(ref)
        if not pred_chars or not ref_chars:
            return 0.0
        m, n = len(pred_chars), len(ref_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_chars[i - 1] == ref_chars[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        p = lcs / m
        r = lcs / n
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    scores = [rouge_l_f1(p, r) for p, r in zip(decoded_preds, decoded_labels)]
    return {"rougeL_f1": float(np.mean(scores))}


def main():
    parser = argparse.ArgumentParser(description="微调摘要模型")
    parser.add_argument(
        "--model_name",
        default="csebuetnlp/mT5_multilingual_XLSum",
        help="预训练模型名称",
    )
    parser.add_argument("--train_file", required=True, help="训练数据路径")
    parser.add_argument("--val_file", default=None, help="验证数据路径")
    parser.add_argument(
        "--output_dir", default="models/checkpoints/finetuned", help="模型输出目录"
    )
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true", help="使用混合精度训练")
    args = parser.parse_args()

    # 加载 tokenizer 与模型
    print(f"加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # 加载并处理数据
    print(f"加载训练数据: {args.train_file}")
    train_dataset = load_data(args.train_file)
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(
            x, tokenizer, args.max_input_length, args.max_target_length
        ),
        batched=True,
        remove_columns=["content", "summary"],
    )

    eval_dataset = None
    if args.val_file:
        print(f"加载验证数据: {args.val_file}")
        eval_dataset = load_data(args.val_file)
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(
                x, tokenizer, args.max_input_length, args.max_target_length
            ),
            batched=True,
            remove_columns=["content", "summary"],
        )

    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_steps=args.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        predict_with_generate=True,
        fp16=args.fp16,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="rougeL_f1" if eval_dataset else None,
        greater_is_better=True if eval_dataset else None,
        report_to="none",
    )

    # 训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer) if eval_dataset else None,
    )

    # 开始训练
    print("开始训练...")
    trainer.train()

    # 保存最终模型
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"训练完成，模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
