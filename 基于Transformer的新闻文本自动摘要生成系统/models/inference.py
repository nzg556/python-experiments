"""
命令行推理脚本：加载模型对单条新闻文本生成摘要。

使用方法：
    python models/inference.py \
        --model_name csebuetnlp/mT5_multilingual_XLSum \
        --text "这里是新闻文本..."

    # 从文件读取
    python models/inference.py \
        --model_name models/checkpoints/mt5-finetuned \
        --file news.txt
"""
import argparse
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_summary(text, model_name, max_length=128, min_length=32, num_beams=4, length_penalty=1.0):
    """
    对单条文本生成摘要。

    Args:
        text: 输入文本。
        model_name: 模型名称或路径。
        max_length: 摘要最大长度。
        min_length: 摘要最小长度。
        num_beams: Beam Search 数量。
        length_penalty: 长度惩罚系数。

    Returns:
        生成的摘要文本。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载模型: {model_name} (设备: {device})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)

    print("正在生成摘要...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summary.strip()


def main():
    parser = argparse.ArgumentParser(description="文本摘要推理")
    parser.add_argument("--model_name", required=True, help="模型名称或路径")
    parser.add_argument("--text", type=str, help="输入文本")
    parser.add_argument("--file", type=str, help="从文件读取输入文本")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--min_length", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    args = parser.parse_args()

    # 获取输入文本
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("请通过 --text 或 --file 提供输入文本")
        sys.exit(1)

    if not text.strip():
        print("输入文本为空")
        sys.exit(1)

    # 生成摘要
    summary = generate_summary(
        text,
        args.model_name,
        max_length=args.max_length,
        min_length=args.min_length,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("【原文】")
    print(text)
    print("\n【摘要】")
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
