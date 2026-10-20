"""
数据集下载脚本：下载中文新闻摘要数据集（LCSTS 等）。

由于 LCSTS 数据集较大，本脚本提供下载链接与处理说明。
用户可根据需要选择下载方式。

使用方法：
    python models/download_data.py --dataset lcsts --output data/raw
"""
import argparse
import os
import json


# 数据集信息
DATASETS = {
    "lcsts": {
        "name": "LCSTS (Large Scale Chinese Short Text Summarization)",
        "description": "大规模中文短文本摘要数据集，包含约 240 万条新闻摘要对。",
        "url": "https://huggingface.co/datasets/knkarthick/xlsum",
        "note": "LCSTS 数据集可从 HuggingFace 或各学术平台获取。",
    },
    "xlsum": {
        "name": "XLSum (Multilingual Summarization Dataset)",
        "description": "多语言摘要数据集，包含中文部分。",
        "url": "https://huggingface.co/datasets/csebuetnlp/xlsum",
        "note": "可直接通过 datasets 库加载。",
    },
}


def generate_sample_data(output_dir: str):
    """生成示例数据，方便用户测试数据处理流程。"""
    samples = [
        {
            "content": "近日，我国自主研发的新一代人工智能大模型正式发布。该模型在多项国际基准测试中取得优异成绩，标志着我国在人工智能领域取得重大突破。据介绍，新模型具备更强的自然语言理解、多模态交互和代码生成能力，可广泛应用于教育、医疗、金融等领域。",
            "summary": "我国发布新一代人工智能大模型，在多项测试中取得优异成绩。",
        },
        {
            "content": "随着新能源汽车市场的快速发展，动力电池技术也在不断进步。最新发布的固态电池技术在能量密度、安全性和充电速度方面均有显著提升。业内专家指出，固态电池有望在未来三到五年内实现规模化量产，届时新能源汽车的续航里程将突破1000公里。",
            "summary": "固态电池技术取得突破，新能源汽车续航有望突破1000公里。",
        },
        {
            "content": "全球气候变化问题日益严峻，各国纷纷加大可再生能源投资力度。最新数据显示，2024年全球可再生能源新增装机容量再创新高，其中太阳能和风能占比超过80%。国际能源署预测，到2030年可再生能源将成为全球最大电力来源。",
            "summary": "全球可再生能源投资持续增长，2030年有望成为最大电力来源。",
        },
    ]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"示例数据已生成: {output_path}")
    print(f"共 {len(samples)} 条样本")
    print(f"可使用以下命令预处理:")
    print(f"  python models/data_preprocess.py --input {output_path} --output data/processed/sample_clean.json --format json")


def main():
    parser = argparse.ArgumentParser(description="下载/生成摘要数据集")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["sample"],
        default="sample",
        help="数据集名称",
    )
    parser.add_argument("--output", default="data/raw", help="输出目录")
    args = parser.parse_args()

    if args.dataset == "sample":
        generate_sample_data(args.output)
        return

    info = DATASETS[args.dataset]
    print(f"数据集: {info['name']}")
    print(f"说明: {info['description']}")
    print(f"下载地址: {info['url']}")
    print(f"备注: {info['note']}")
    print(f"\n请手动下载数据到: {args.output}/")
    print(f"下载后使用以下命令预处理:")
    print(f"  python models/data_preprocess.py --input {args.output}/<文件名> --output data/processed/clean.json --format <json/csv/txt>")


if __name__ == "__main__":
    main()
