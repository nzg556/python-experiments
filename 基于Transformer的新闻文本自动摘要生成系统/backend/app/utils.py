"""
工具模块：包含 ROUGE 评估等辅助函数。
"""
from typing import Dict

import jieba
from loguru import logger


def _tokenize_chinese(text: str) -> str:
    """对中文文本进行分词，用空格连接，便于 ROUGE 计算。"""
    return " ".join(jieba.cut(text.strip()))


def compute_rouge(hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
    """
    计算 ROUGE-1、ROUGE-2、ROUGE-L 分数。

    Args:
        hypothesis: 生成的摘要。
        reference: 参考摘要。

    Returns:
        包含 rouge1、rouge2、rougeL 的 precision/recall/f1 字典。
    """
    try:
        from rouge import Rouge

        hypothesis_tokens = _tokenize_chinese(hypothesis)
        reference_tokens = _tokenize_chinese(reference)

        # 如果分词后为空，返回 0 分
        if not hypothesis_tokens.strip() or not reference_tokens.strip():
            zero = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            return {"rouge1": zero, "rouge2": zero, "rougeL": zero}

        rouge = Rouge()
        scores = rouge.get_scores(hypothesis_tokens, reference_tokens, avg=True)

        # rouge 库返回的 key 为 rouge-1/rouge-2/rouge-l，统一映射为 rouge1/rouge2/rougeL
        key_mapping = {"rouge-1": "rouge1", "rouge-2": "rouge2", "rouge-l": "rougeL"}
        result = {}
        for src_key, dst_key in key_mapping.items():
            score = scores[src_key]
            result[dst_key] = {
                "precision": round(score["p"], 4),
                "recall": round(score["r"], 4),
                "f1": round(score["f"], 4),
            }
        return result

    except ImportError:
        logger.warning("rouge 库未安装，使用简易 ROUGE 计算")
        return _simple_rouge(hypothesis, reference)
    except Exception as e:
        logger.error(f"ROUGE 计算失败: {e}")
        zero = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {"rouge1": zero, "rouge2": zero, "rougeL": zero}


def _simple_rouge(hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
    """
    简易 ROUGE 计算（当 rouge 库不可用时的后备方案）。
    基于字符级 n-gram 重叠。
    """

    def _ngrams(text: str, n: int):
        chars = list(text.strip())
        return set(chars[i : i + n] for i in range(len(chars) - n + 1))

    def _lcs_length(a: str, b: str) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    def _score(hyp_ngrams, ref_ngrams):
        if not hyp_ngrams or not ref_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        overlap = hyp_ngrams & ref_ngrams
        p = len(overlap) / len(hyp_ngrams)
        r = len(overlap) / len(ref_ngrams)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    result = {
        "rouge1": _score(_ngrams(hypothesis, 1), _ngrams(reference, 1)),
        "rouge2": _score(_ngrams(hypothesis, 2), _ngrams(reference, 2)),
    }

    # ROUGE-L
    lcs = _lcs_length(hypothesis, reference)
    p = lcs / len(hypothesis) if hypothesis else 0.0
    r = lcs / len(reference) if reference else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    result["rougeL"] = {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }

    return result
