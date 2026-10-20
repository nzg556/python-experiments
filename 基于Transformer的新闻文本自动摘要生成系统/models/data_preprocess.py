"""
数据预处理脚本：将原始中文新闻摘要数据（如 LCSTS）转换为模型可用格式。

支持的输入格式：
- JSON: [{ "content": "...", "summary": "..." }, ...]
- CSV: 含 content, summary 两列
- 纯文本：每行一篇，用 \t 分隔 content 和 summary

输出格式：JSON 文件，便于 HuggingFace datasets 加载。

使用方法：
    python models/data_preprocess.py \
        --input data/raw/lcsts.json \
        --output data/processed/lcsts_clean.json \
        --format json
"""
import argparse
import json
import os
import re
from typing import List, Dict

import pandas as pd


def clean_text(text: str) -> str:
    """清洗文本：去除多余空白、特殊字符等。"""
    if not text:
        return ""
    # 去除 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 去除多余空白字符
    text = re.sub(r"\s+", " ", text)
    # 去除首尾空白
    text = text.strip()
    return text


def load_json(path: str) -> List[Dict]:
    """加载 JSON 格式数据。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_csv(path: str) -> List[Dict]:
    """加载 CSV 格式数据。"""
    df = pd.read_csv(path)
    # 兼容不同列名
    content_col = "content" if "content" in df.columns else df.columns[0]
    summary_col = "summary" if "summary" in df.columns else df.columns[1]
    return df[[content_col, summary_col]].rename(
        columns={content_col: "content", summary_col: "summary"}
    ).to_dict("records")


def load_text(path: str) -> List[Dict]:
    """加载纯文本格式数据（每行 content\tsummary）。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                data.append({"content": parts[0], "summary": parts[1]})
    return data


def preprocess(data: List[Dict], min_len: int = 10, max_len: int = 2000) -> List[Dict]:
    """
    预处理数据：清洗、过滤过短或过长的样本。

    Args:
        data: 原始数据列表。
        min_len: 原文最小长度。
        max_len: 原文最大长度。

    Returns:
        清洗后的数据列表。
    """
    processed = []
    for item in data:
        content = clean_text(item.get("content", ""))
        summary = clean_text(item.get("summary", ""))
        if not content or not summary:
            continue
        if len(content) < min_len or len(content) > max_len:
            continue
        if len(summary) < 2:
            continue
        processed.append({"content": content, "summary": summary})
    return processed


def main():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("--input", required=True, help="输入数据路径")
    parser.add_argument("--output", required=True, help="输出数据路径")
    parser.add_argument(
        "--format",
        choices=["json", "csv", "txt"],
        default="json",
        help="输入数据格式",
    )
    parser.add_argument("--min_len", type=int, default=10, help="原文最小长度")
    parser.add_argument("--max_len", type=int, default=2000, help="原文最大长度")
    args = parser.parse_args()

    # 加载数据
    loaders = {"json": load_json, "csv": load_csv, "txt": load_text}
    raw_data = loaders[args.format](args.input)
    print(f"加载原始数据: {len(raw_data)} 条")

    # 预处理
    processed = preprocess(raw_data, min_len=args.min_len, max_len=args.max_len)
    print(f"预处理后数据: {len(processed)} 条")

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"已保存到: {args.output}")

    # 打印统计信息
    if processed:
        content_lens = [len(d["content"]) for d in processed]
        summary_lens = [len(d["summary"]) for d in processed]
        print(f"原文长度 - 平均: {sum(content_lens)//len(content_lens)}, "
              f"最大: {max(content_lens)}, 最小: {min(content_lens)}")
        print(f"摘要长度 - 平均: {sum(summary_lens)//len(summary_lens)}, "
              f"最大: {max(summary_lens)}, 最小: {min(summary_lens)}")


if __name__ == "__main__":
    main()
