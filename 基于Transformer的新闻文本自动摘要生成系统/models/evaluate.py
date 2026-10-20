"""
模型评估脚本：在测试集上评估摘要模型，计算 ROUGE 分数。

使用方法：
    python models/evaluate.py \
        --model_name models/checkpoints/mt5-finetuned \
        --test_file data/processed/test.json \
        --output results/eval_result.json
"""
import argparse
import json
import os
import sys

# 将脚本所在目录加入 sys.path，确保从任意位置运行都能导入 data_preprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data_preprocess import clean_text


def compute_rouge_scores(hyps, refs):
    """
    计算 ROUGE-1、ROUGE-2、ROUGE-L 分数（字符级，适用于中文）。

    Args:
        hyps: 生成的摘要列表。
        refs: 参考摘要列表。

    Returns:
        包含各指标分数的字典。
    """
    from rouge import Rouge
    import jieba

    def tokenize(text):
        return " ".join(jieba.cut(text.strip()))

    hyps_tokens = [tokenize(h) for h in hyps]
    refs_tokens = [tokenize(r) for r in refs]

    rouge = Rouge()
    scores = rouge.get_scores(hyps_tokens, refs_tokens, avg=True)

    result = {}
    for key, metric in [("rouge-1", "rouge1"), ("rouge-2", "rouge2"), ("rouge-l", "rougeL")]:
        s = scores[key]
        result[metric] = {
            "precision": round(s["p"], 4),
            "recall": round(s["r"], 4),
            "f1": round(s["f"], 4),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="评估摘要模型")
    parser.add_argument("--model_name", required=True, help="模型路径或名称")
    parser.add_argument("--test_file", required=True, help="测试数据路径")
    parser.add_argument("--output", default="results/eval_result.json", help="结果输出路径")
    parser.add_argument("--max_length", type=int, default=128, help="生成摘要最大长度")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam Search 数量")
    parser.add_argument("--batch_size", type=int, default=4, help="推理批次大小")
    parser.add_argument("--max_samples", type=int, default=0, help="评估样本数（0 表示全部）")
    args = parser.parse_args()

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载模型: {args.model_name} (设备: {device})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    model.eval()

    # 加载测试数据
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if args.max_samples > 0:
        test_data = test_data[: args.max_samples]
    print(f"测试数据: {len(test_data)} 条")

    # 生成摘要
    hyps = []
    refs = []
    for i in range(0, len(test_data), args.batch_size):
        batch = test_data[i : i + args.batch_size]
        contents = [clean_text(item["content"]) for item in batch]
        summaries = [clean_text(item["summary"]) for item in batch]

        inputs = tokenizer(
            contents,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=args.max_length,
                num_beams=args.num_beams,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        hyps.extend(preds)
        refs.extend(summaries)

        if (i // args.batch_size) % 10 == 0:
            print(f"进度: {min(i + args.batch_size, len(test_data))}/{len(test_data)}")

    # 计算 ROUGE
    print("计算 ROUGE 分数...")
    scores = compute_rouge_scores(hyps, refs)

    # 输出结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result = {
        "model": args.model_name,
        "num_samples": len(test_data),
        "metrics": scores,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== 评估结果 =====")
    for metric, vals in scores.items():
        print(f"{metric}: P={vals['precision']}, R={vals['recall']}, F1={vals['f1']}")
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
