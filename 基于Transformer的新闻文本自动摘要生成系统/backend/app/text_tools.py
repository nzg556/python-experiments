"""
轻量文本处理工具集（不依赖深度学习模型，纯规则与统计实现）。

包含：
- 句子切分
- 摘要润色（语病修正、冗余精简、正式书面风格）
- 抽取式摘要（TF-IDF + 位置权重，用于与生成式摘要对比）
- 关键词提取（jieba TF-IDF）
- 字数统计与压缩比例
- 摘要-原文关键句对齐（用于原文高亮定位）
- 摘要可信度评估（内容词覆盖 + 数字一致性，检测幻觉）
- 新闻主题分类（词典加权，覆盖财经/体育/科技/社会/娱乐/教育/健康/军事/国际/法制）
"""
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import jieba
import jieba.analyse
from loguru import logger

# ---------------------------------------------------------------------------
# 停用词（用于抽取式摘要与关键词过滤）
# ---------------------------------------------------------------------------
STOPWORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 "
    "自己 这 那 他 她 它 们 与 及 或 等 为 以 于 而 之 其 中 对 从 被 把 让 向 但 并 "
    "以及 因此 所以 因为 但是 然而 并且 而且 目前 记者 报道 新闻 消息 时候 问题 方面 "
    "进行 通过 可以 这些 那些 我们 你们 相关 有关 表示 认为 指出 介绍"
    .split()
)

# ---------------------------------------------------------------------------
# 脏数据过滤模式（网页残留标记、多余空白、换行符）
# ---------------------------------------------------------------------------
_DIRTY_PATTERNS: List[Tuple[str, str]] = [
    # 网页图片/序号标记，如 (组图) (1) (2) [组图] [1]
    (r"\s*[（(]\s*组图\s*[）)]\s*", ""),
    (r"\s*[（(]\s*[\d一二三四五六七八九十]+\s*[）)]\s*", ""),
    (r"\s*[（(]\s*来源\s*[:：].*?[）)]\s*", ""),
    (r"\s*[（(]\s*编辑\s*[:：].*?[）)]\s*", ""),
    # HTML 实体与标签残留
    (r"&[a-zA-Z]+;", " "),
    (r"<[^>]+>", " "),
    # 多余空白与换行
    (r"\s+", " "),
]

# 关键词黑名单（jieba 易误分的无意义碎片）
_KEYWORD_BLACKLIST = {
    "组图", "保质", "保量", "表示", "认为", "指出", "介绍", "报道", "记者",
    "新闻", "消息", "时候", "问题", "方面", "进行", "通过", "可以", "这些",
    "那些", "我们", "你们", "相关", "有关", "目前", "因此", "所以", "因为",
    "但是", "然而", "并且", "而且",
}

# ---------------------------------------------------------------------------
# 1. 句子切分
# ---------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")


def split_sentences(text: str) -> List[str]:
    """将中文文本切分为句子列表（保留非空句子，去除首尾空白）。"""
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
    return sents


def clean_news_text(text: str) -> str:
    """过滤网页残留标记、多余空白等脏数据，统一空格。"""
    if not text:
        return text
    cleaned = text.strip()
    for pattern, repl in _DIRTY_PATTERNS:
        cleaned = re.sub(pattern, repl, cleaned)
    return cleaned.strip()


def auto_punctuate(text: str, max_sent_len: int = 40) -> str:
    """对缺少标点的长文本尝试自动断句并插入标点（基于 jieba 词性标注）。"""
    if not text:
        return text
    # 如果已有足够标点，则跳过
    punct_count = len(re.findall(r"[，。！？；]", text))
    avg_len = len(text) / max(punct_count, 1)
    if punct_count > 0 and avg_len < max_sent_len * 1.2:
        return text

    import jieba.posseg as pseg
    words = list(pseg.cut(text))
    result: List[str] = []
    sent_len = 0
    for i, (word, flag) in enumerate(words):
        result.append(word)
        sent_len += len(word)
        if word in "，。！？；":
            sent_len = 0
            continue
        # 句末特征词 + 句子足够长 -> 尝试断句
        if sent_len >= max_sent_len and word in (
            "了", "将", "已", "得到", "缓解", "提升", "完成", "实现",
            "指出", "认为", "表示", "报道", "开启", "启动", "建成",
        ):
            if i < len(words) - 1 and words[i + 1][0] not in "，。！？；":
                next_flag = words[i + 1][1]
                # 下一个是名词/动词/副词/形容词/代词，视为新句开头
                if next_flag.startswith(("n", "v", "d", "a", "r")):
                    result.append("。")
                    sent_len = 0
                    continue
        # 过长且无特征词，在动词/名词后尝试插入逗号
        if sent_len >= max_sent_len * 1.5:
            if flag.startswith(("v", "n")) and i < len(words) - 1:
                next_word, _ = words[i + 1]
                if next_word not in "，。！？；":
                    result.append("，")
                    sent_len = 0
    return "".join(result)


# ---------------------------------------------------------------------------
# 2. 摘要润色（规则引擎）
# ---------------------------------------------------------------------------
# 口语/机器腔 -> 正式书面语 替换表
_FORMAL_REPLACEMENTS: List[Tuple[str, str]] = [
    ("很多", "大量"),
    ("好多", "许多"),
    ("搞", "进行"),
    ("弄", "进行"),
    ("觉得", "认为"),
    ("现在", "目前"),
    ("如今", "当前"),
    ("咱们", "我们"),
    ("跟", "与"),
    ("和…一样", "如同"),
    ("差不多", "几乎"),
    ("特别", "尤为"),
    ("挺", "较为"),
    ("比如说", "例如"),
    ("所以说", "由此可见，"),
    ("还有就是", "此外，"),
    ("也就是说", "即"),
    ("去年", "上年"),
    ("今年", "本年度"),
    ("过一段时间", "后续"),
    ("越来越大", "持续扩大"),
    ("越来越多", "持续增加"),
    ("非常", "十分"),
    ("取得了很大的进展", "取得显著进展"),
    ("有很重要的意义", "具有重要意义"),
    ("起到了很大的作用", "发挥了重要作用"),
    ("引起了很大的关注", "引发广泛关注"),
]

# 句首冗余引导语（生成模型常见的机器腔开头）
_REDUNDANT_OPENERS = [
    "据了解，", "据悉，", "报道称，", "据报道，", "据报道了解，",
    "总的来说，", "总的来看，", "总之，", "综上所述，",
    "在这篇新闻中，", "这篇文章中，", "本文中，", "文章中提到，",
    "新闻中，", "该新闻讲述了", "这篇新闻主要讲述了", "这段文字讲述了",
    "新闻主要报道了", "新闻称，",
]

# 常见语病模式 -> 修正
_GRAMMAR_PATTERNS: List[Tuple[str, str]] = [
    (r"(::{2,}|。{2,}|，{2,}|！{2,}|？{2,})", lambda m: m.group(0)[0]),  # 重复标点
    (r"的的+", "的"),
    (r"了了+", "了"),
    (r"地地+", "地"),
    (r"是是+", "是"),
    (r"在在+", "在"),
    (r"进行进行|研究研究|讨论讨论", None),  # 占位，下面统一处理
    (r"\s{2,}", " "),
    (r"，\s*。", "。"),
    (r"。\s*。", "。"),
    (r"([。！？])\s*，", r"\1"),
    (r"([a-zA-Z0-9])([，。！？；：])", r"\1\2"),  # 去中英文间多余空白由下一步处理
    (r"\s+([，。！？；：、])", r"\1"),
]


def polish_summary(text: str, formal_style: bool = True) -> Dict[str, str]:
    """
    对生成式初稿摘要进行轻量润色。

    Args:
        text: 模型生成的初稿摘要。
        formal_style: 是否启用正式书面风格（适配新闻文体）。

    Returns:
        {"polished": 润色后摘要, "changes": [修改说明列表]}
    """
    if not text or not text.strip():
        return {"polished": text or "", "changes": []}

    polished = text.strip()
    changes: List[str] = []

    # 1) 去除句首冗余引导语
    for opener in _REDUNDANT_OPENERS:
        if polished.startswith(opener):
            polished = polished[len(opener):]
            changes.append(f"删除句首冗余引导语“{opener.rstrip('，,')}”")
            break

    # 2) 语病与格式修正（重复标点/重复字/标点空格）
    before = polished
    for pattern, repl in _GRAMMAR_PATTERNS:
        if repl is None:
            continue
        polished = re.sub(pattern, repl, polished)
    if polished != before:
        changes.append("修正重复用字与标点")

    # 3) 正式书面风格替换
    if formal_style:
        hit_words = []
        for src, dst in _FORMAL_REPLACEMENTS:
            if src in polished:
                polished = polished.replace(src, dst)
                hit_words.append(f"{src}→{dst}")
        if hit_words:
            changes.append("正式书面风格替换：" + "、".join(hit_words[:5]))

    # 4) 精简冗余短语（去修饰性堆叠）
    before = polished
    _REDUNDANT_PHRASES = [
        ("非常地", "十分"), ("特别的", "尤为"), ("进行了进行了", "进行了"),
        ("一个一个", "一个个"), ("自己自己", "自己"),
        ("在……的过程中，", "在…过程中，"),
        ("的过程中，", "时，"),
        ("的发展的", "发展的"),
    ]
    for src, dst in _REDUNDANT_PHRASES:
        polished = polished.replace(src, dst)
    if polished != before:
        changes.append("精简冗余语句")

    # 5) 句子去重：内容几乎相同的相邻句子仅保留一句
    sents = split_sentences(polished)
    if len(sents) > 1:
        kept: List[str] = []
        for s in sents:
            dup = False
            for k in kept:
                if _char_ngram_sim(s, k) > 0.75:
                    dup = True
                    break
            if not dup:
                kept.append(s)
        if len(kept) < len(sents):
            changes.append(f"合并 {len(sents) - len(kept)} 个重复/近重复句子")
        polished = "".join(kept)

    # 6) 收尾规范：末尾句号、首字母（中文无需大写）
    polished = polished.strip()
    if polished and polished[-1] not in "。！？；…":
        polished += "。"

    if not changes:
        changes.append("原文表述已较规范，未做明显修改")
    return {"polished": polished, "changes": changes}


# ---------------------------------------------------------------------------
# 3. 抽取式摘要（TF-IDF + 位置权重）
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    return [w for w in jieba.lcut(text) if w.strip() and w not in STOPWORDS
            and not re.fullmatch(r"[，。！？；：、\s\d%.“”\"'（）()a-zA-Z]", w)]


def _char_ngram_sim(a: str, b: str, n: int = 2) -> float:
    """字符 n-gram 相似度（Dice 系数），用于句对齐与去重。"""
    if not a or not b:
        return 0.0
    ga = {a[i:i + n] for i in range(len(a) - n + 1)}
    gb = {b[i:i + n] for i in range(len(b) - n + 1)}
    if not ga or not gb:
        return 0.0
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def extractive_summary(text: str, max_sentences: int = 3) -> Dict[str, any]:
    """
    抽取式摘要：按 TF-IDF 词频与句间相似度对句子打分，选取 Top-N 句，
    按原文顺序拼接。用于与 Transformer 生成式摘要做对比实验。
    """
    sents = split_sentences(text)
    if not sents:
        return {"summary": "", "sentences": []}
    if len(sents) <= max_sentences:
        return {
            "summary": "".join(sents),
            "sentences": [{"index": i, "sentence": s, "score": 0.0}
                          for i, s in enumerate(sents)],
        }

    tokens_per_sent = [_tokenize(s) for s in sents]
    df = Counter()
    for toks in tokens_per_sent:
        df.update(set(toks))
    n_sents = len(sents)

    scores: List[float] = []
    for i, (s, toks) in enumerate(zip(sents, tokens_per_sent)):
        if not toks:
            scores.append(0.0)
            continue
        tfidf = sum((1 + df[t] / n_sents) for t in set(toks)) / (1 + len(toks))
        # 位置权重：首段与末段（新闻导语/结语）加权
        pos_w = 1.25 if i == 0 else (1.1 if i == n_sents - 1 else 1.0)
        # 长度惩罚：过短句子（如广告句）降权
        len_w = min(len(s) / 20.0, 1.0) if len(s) < 20 else 1.0
        scores.append(tfidf * pos_w * len_w)

    top_idx = sorted(range(n_sents), key=lambda i: scores[i], reverse=True)[:max_sentences]
    top_idx_sorted = sorted(top_idx)  # 按原文顺序
    summary = "".join(sents[i] for i in top_idx_sorted)
    return {
        "summary": summary,
        "sentences": [{"index": i, "sentence": sents[i], "score": round(scores[i], 4)}
                      for i in top_idx_sorted],
    }


# ---------------------------------------------------------------------------
# 4. 关键词提取
# ---------------------------------------------------------------------------
def extract_keywords(text: str, topk: int = 8) -> List[Dict[str, any]]:
    """基于 jieba TF-IDF 的关键词提取（过滤黑名单中的无意义碎片）。"""
    tags = jieba.analyse.extract_tags(text, topK=topk * 3, withWeight=True)
    keywords = [
        {"word": w, "weight": round(float(s), 4)}
        for w, s in tags
        if w not in _KEYWORD_BLACKLIST and not re.fullmatch(r"[，。！？；：、\s\d%.“”\"'（）()a-zA-Z]", w)
    ]
    return keywords[:topk]


# ---------------------------------------------------------------------------
# 5. 字数统计与压缩比
# ---------------------------------------------------------------------------
def count_stats(original: str, summary: str) -> Dict[str, any]:
    """统计原文字数、摘要字数与压缩比例。"""
    o_chars = len(re.sub(r"\s", "", original or ""))
    s_chars = len(re.sub(r"\s", "", summary or ""))
    o_sents = len(split_sentences(original or ""))
    s_sents = len(split_sentences(summary or ""))
    ratio = round(s_chars / o_chars * 100, 1) if o_chars else 0.0
    return {
        "original_chars": o_chars,
        "summary_chars": s_chars,
        "compression_ratio": ratio,          # 摘要占原文百分比
        "compression_percent": round(100 - ratio, 1),  # 压缩掉的百分比
        "original_sentences": o_sents,
        "summary_sentences": s_sents,
    }


# ---------------------------------------------------------------------------
# 6.5 中文可读性指标（Flesch 易读性的中文适配）
# ---------------------------------------------------------------------------
def readability_score(text: str) -> Dict[str, any]:
    """
    中文文本易读性评估（Flesch Reading Ease 的中文适配公式）。

    参考中文可读性研究常用思路：句子越短、平均词长越短，文本越容易阅读。
    score = 100 - 1.2 × 平均句长 - 8 × 平均词长（截断到 0-100）。

    Returns:
        {"score": 0-100（越高越易读）, "level": 容易/适中/偏难,
         "avg_sentence_len": 平均句长（字）, "avg_word_len": 平均词长（字）}
    """
    clean = re.sub(r"\s", "", text or "")
    if not clean:
        return {"score": 0.0, "level": "无内容", "avg_sentence_len": 0.0, "avg_word_len": 0.0}
    sents = split_sentences(text or "")
    n_sents = max(len(sents), 1)
    words = [w for w in _tokenize(clean) if w.strip()]
    n_words = max(len(words), 1)
    avg_sent_len = len(clean) / n_sents
    avg_word_len = len(clean) / n_words
    score = 100.0 - 1.2 * avg_sent_len - 8.0 * avg_word_len
    score = max(0.0, min(100.0, round(score, 1)))
    level = "容易" if score >= 70 else ("适中" if score >= 50 else "偏难")
    return {
        "score": score,
        "level": level,
        "avg_sentence_len": round(avg_sent_len, 1),
        "avg_word_len": round(avg_word_len, 2),
    }


# ---------------------------------------------------------------------------
# 6. 摘要-原文关键句对齐（高亮定位）
# ---------------------------------------------------------------------------
def align_highlights(original: str, summary: str, threshold: float = 0.18) -> Dict[str, any]:
    """
    将摘要句与原文句做字符 n-gram 相似度匹配，标出摘要主要参考的原文关键句。

    Returns:
        {"sentences": [{"index", "sentence", "is_key", "score"}],
         "coverage": 关键句覆盖的原文句比例}
    """
    o_sents = split_sentences(original)
    s_sents = split_sentences(summary)
    flags = [0.0] * len(o_sents)

    for s in s_sents:
        best_i, best_s = -1, 0.0
        for i, o in enumerate(o_sents):
            sim = _char_ngram_sim(s, o, n=2)
            if sim > best_s:
                best_i, best_s = i, sim
        if best_i >= 0 and best_s >= threshold:
            flags[best_i] = max(flags[best_i], best_s)

    result = [
        {"index": i, "sentence": s, "is_key": flags[i] > 0,
         "score": round(flags[i], 4)}
        for i, s in enumerate(o_sents)
    ]
    covered = sum(1 for f in flags if f > 0)
    return {
        "sentences": result,
        "coverage": round(covered / len(o_sents) * 100, 1) if o_sents else 0.0,
    }


# ---------------------------------------------------------------------------
# 7. 摘要可信度评估（幻觉检测）
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"\d+(?:\.\d+)?(?:%|万|亿|千|百|元|美元|人|次|年|月|日|台|辆|家)?")
_ENTITY_RE = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9·]{2,12}(?:公司|大学|研究院|研究所|银行|集团| agency|部|局|委|院|市|省|国)")


def credibility_score(original: str, summary: str) -> Dict[str, any]:
    """
    评估生成摘要的可信度（检测事实幻觉）：
    1) 内容词覆盖率：摘要实词在原文中的出现比例（权重 0.6）
    2) 数字一致性：摘要中的数字（及单位）是否能在原文中找到（权重 0.25）
    3) 实体一致性：摘要中的机构/地名等实体是否在原文出现（权重 0.15）

    Returns:
        {"score": 0-100, "level": 高/中/低, "missing_fragments": [...],
         "details": {...}, "warnings": [...]}
    """
    o_text = re.sub(r"\s", "", original or "")
    s_text = re.sub(r"\s", "", summary or "")
    missing: List[str] = []
    warnings: List[str] = []

    # --- 1) 内容词覆盖率 ---
    o_words = set(_tokenize(original))
    s_words = [w for w in _tokenize(summary)]
    if s_words:
        covered = [w for w in set(s_words) if w in o_words]
        cov = len(covered) / len(set(s_words))
    else:
        cov = 0.0
    missing_words = [w for w in set(s_words) if w not in o_words]
    if missing_words:
        missing.extend(missing_words[:8])

    # --- 2) 数字一致性 ---
    s_nums = _NUM_RE.findall(s_text)
    num_ok = 1.0
    if s_nums:
        num_hit = [n for n in s_nums if n in o_text]
        num_ok = len(num_hit) / len(s_nums)
        bad_nums = [n for n in s_nums if n not in o_text]
        if bad_nums:
            warnings.append(f"摘要中的数字 {', '.join(bad_nums[:5])} 未在原文中出现，请核对")
            missing.extend([f"数字{n}" for n in bad_nums[:5]])

    # --- 3) 实体一致性 ---
    s_ents = [e for e in _ENTITY_RE.findall(s_text)]
    ent_ok = 1.0
    if s_ents:
        ent_hit = [e for e in s_ents if e in o_text]
        ent_ok = len(ent_hit) / len(s_ents)
        bad_ents = [e for e in set(s_ents) if e not in o_text]
        if bad_ents:
            warnings.append(f"摘要中的实体 {', '.join(bad_ents[:5])} 疑似未在原文出现")
            missing.extend([f"实体“{e}”" for e in bad_ents[:5]])

    # --- 综合得分 ---
    score = round((cov * 0.6 + num_ok * 0.25 + ent_ok * 0.15) * 100, 1)
    # 空摘要兜底
    if not s_text:
        score = 0.0
    level = "高" if score >= 80 else ("中" if score >= 60 else "低")
    details = {
        "content_coverage": round(cov * 100, 1),   # 内容词覆盖率 %
        "number_consistency": round(num_ok * 100, 1),
        "entity_consistency": round(ent_ok * 100, 1),
        "summary_nums": list(set(s_nums))[:10],
    }

    # --- 幻觉类型识别 ---
    # 根据各子指标的具体表现细分幻觉类型，便于论文实验与错误归因
    hallucination_types: List[Dict[str, str]] = []
    if s_text and cov < 0.6:
        hallucination_types.append({
            "type": "无中生有",
            "level": "高",
            "description": "摘要中大量内容词未在原文出现，存在编造事实风险",
        })
    if s_nums and num_ok < 1.0:
        hallucination_types.append({
            "type": "数字错误",
            "level": "高" if num_ok < 0.5 else "中",
            "description": "摘要中的数字/比例在原文中无对应，可能存在数值幻觉",
        })
    if s_ents and ent_ok < 1.0:
        hallucination_types.append({
            "type": "实体编造",
            "level": "高" if ent_ok < 0.5 else "中",
            "description": "摘要出现原文未提及的人名/机构/地名等实体",
        })
    if s_text and 0.6 <= cov < 0.85 and num_ok == 1.0 and ent_ok == 1.0:
        hallucination_types.append({
            "type": "事实偏差",
            "level": "低",
            "description": "摘要基本忠于原文，但个别表述与原文存在偏差",
        })
    if s_text and not hallucination_types:
        hallucination_types.append({
            "type": "无明显幻觉",
            "level": "无",
            "description": "摘要内容与原文高度一致，未检测到明显幻觉",
        })

    return {
        "score": score,
        "level": level,
        "missing_fragments": missing[:12],
        "warnings": warnings,
        "details": details,
        "hallucination_types": hallucination_types,
    }


# ---------------------------------------------------------------------------
# 8. 新闻主题分类（词典加权）
# ---------------------------------------------------------------------------
CATEGORY_LEXICON: Dict[str, Dict[str, int]] = {
    "财经": {"经济": 3, "金融": 3, "股市": 4, "股": 1, "基金": 3, "投资": 2, "资本": 2, "银行": 2,
             "利率": 3, "汇率": 3, "货币": 2, "上市公司": 3, "营收": 3, "利润": 2, "财报": 3, "市值": 3,
             "证券": 3, "债券": 3, "融资": 2, "IPO": 3, "GDP": 3, "通胀": 3, "消费": 3, "市场": 1,
             "交易": 1, "股价": 3, "涨": 1, "跌": 1, "涨幅": 2, "创业板": 3, "证监会": 4, "央行": 4,
             "免税": 4, "海关": 3, "购物": 3, "销售额": 4, "增长": 1, "同比": 2, "环比": 2, "退税": 3,
             "关税": 3, "进出口": 3, "贸易": 3, "订单": 2, "同比分别增长": 3, "万人次": 2, "亿元": 1,
             "万人次参与": 3, "离岛": 3, "假期": 1, "旅游收入": 3, "零售额": 3},
    "体育": {"体育": 3, "足球": 4, "篮球": 4, "比赛": 3, "联赛": 3, "球员": 3, "教练": 2, "夺冠": 4,
             "冠军": 3, "世界杯": 4, "奥运": 4, "亚运": 4, "进球": 3, "比分": 3, "球队": 3, "赛季": 3,
             "淘汰赛": 3, "运动员": 3, "锦标赛": 3, "决赛": 3, "主场": 2, "客场": 2, "中超": 4, "NBA": 4},
    "科技": {"科技": 3, "人工智能": 4, "AI": 3, "大模型": 4, "芯片": 4, "半导体": 4, "算法": 3, "互联网": 2,
             "软件": 2, "硬件": 2, "手机": 2, "发布": 1, "技术": 2, "创新": 2, "研发": 2, "机器人": 3,
             "自动驾驶": 4, "量子": 3, "5G": 3, "6G": 3, "云计算": 3, "数据": 1, "数字化": 3, "开源": 2,
             "程序员": 2, "代码": 2, "专利": 2, "操作系统": 3, "新能源": 2, "智能": 1},
    "社会": {"社会": 3, "民生": 3, "居民": 2, "市民": 2, "社区": 2, "养老金": 3, "医疗": 2, "教育": 1,
             "就业": 2, "房价": 3, "交通": 2, "地铁": 2, "供暖": 2, "养老": 2, "学生": 1, "家长": 1,
             "志愿者": 2, "公益": 2, "救助": 2, "事故": 2, "火灾": 3, "地震": 3, "救援": 3, "警方": 2,
             "天气": 2, "台风": 3, "暴雨": 3},
    "娱乐": {"娱乐": 3, "电影": 4, "电视剧": 4, "综艺": 4, "明星": 4, "演员": 3, "导演": 3, "票房": 4,
             "演唱会": 4, "专辑": 3, "歌曲": 3, "音乐": 2, "直播": 2, "粉丝": 3, "网红": 3, "剧组": 3,
             "上映": 3, "开播": 3, "收视率": 3, "热搜": 3, "代言": 2, "选秀": 3},
    "教育": {"教育": 4, "学校": 3, "高考": 4, "中考": 4, "招生": 3, "大学": 3, "中学": 3, "小学": 3,
             "教师": 3, "学生": 2, "考试": 3, "录取": 3, "分数线": 3, "课程": 2, "双减": 4, "考研": 4,
             "留学": 3, "学费": 3, "校园": 2, "教育厅": 4},
    "健康": {"健康": 3, "医疗": 3, "医院": 3, "医生": 3, "患者": 3, "疾病": 3, "病毒": 3, "疫苗": 4,
             "药": 2, "药品": 3, "治疗": 3, "诊断": 3, "临床": 3, "疫情": 4, "感染": 3, "卫生": 2,
             "食品": 1, "手术": 3, "癌症": 3, "保健": 2},
    "军事": {"军事": 4, "军队": 4, "部队": 3, "导弹": 4, "军演": 4, "国防": 4, "武器": 3, "战机": 4,
             "军舰": 4, "航母": 4, "士兵": 3, "演习": 3, "装备": 2, "射击": 2, "空军": 4, "海军": 4,
             "陆军": 4, "武器装备": 4},
    "国际": {"国际": 3, "美国": 2, "中国": 1, "日本": 2, "俄罗斯": 2, "欧盟": 3, "联合国": 4, "总统": 3,
             "首相": 3, "外交": 4, "峰会": 3, "会谈": 3, "条约": 3, "制裁": 3, "出口": 1, "进口": 1,
             "全球": 2, "世界": 1, "访华": 4, "大使": 3},
    "法制": {"法制": 4, "法律": 3, "法院": 4, "法庭": 4, "犯罪": 4, "嫌疑": 3, "警方": 2, "公安": 3,
             "检察": 4, "起诉": 4, "判决": 4, "审理": 4, "立法": 4, "律师": 3, "治安": 3, "诈骗": 3,
             " arrested": 0, "拘留": 3, "刑": 2, "民法典": 4},
}


def classify_news(text: str, topk: int = 3) -> Dict[str, any]:
    """
    基于领域词典加权的新闻主题分类（无需联网与额外模型，毫秒级完成）。

    Returns:
        {"category": 主类别, "confidence": 0-100, "scores": {类别: 分数},
         "hits": {类别: [命中的词]}}
    """
    words = jieba.lcut(text)
    scores: Dict[str, float] = {cat: 0.0 for cat in CATEGORY_LEXICON}
    hits: Dict[str, List[str]] = {cat: [] for cat in CATEGORY_LEXICON}

    # 词级别匹配
    for w in set(words):
        for cat, lex in CATEGORY_LEXICON.items():
            if w in lex:
                scores[cat] += lex[w]
                hits[cat].append(w)

    # 关键词补充（TF-IDF top 词做模糊包含匹配，提升短文本召回）
    for kw in jieba.analyse.extract_tags(text, topK=15):
        for cat, lex in CATEGORY_LEXICON.items():
            for lw, lw_weight in lex.items():
                if len(lw) >= 2 and lw in kw and kw not in hits[cat] and lw != kw:
                    scores[cat] += lw_weight * 0.5
                    hits[cat].append(kw)
                    break

    total = sum(scores.values())
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_score = ranked[0]
    if total <= 0 or top_score <= 0:
        return {
            "category": "综合/其他",
            "confidence": 0.0,
            "scores": {k: 0.0 for k in scores},
            "hits": {},
        }
    confidence = round(top_score / total * 100, 1)
    return {
        "category": top_cat,
        "confidence": confidence,
        "scores": {k: round(v, 1) for k, v in ranked[:topk] if v > 0},
        "hits": {k: v[:8] for k, v in hits.items() if k == top_cat and v},
    }


# ---------------------------------------------------------------------------
# 9. 新闻关联推荐（基于历史新闻库的简易向量检索）
# ---------------------------------------------------------------------------
def recommend_similar(
    text: str,
    library: List[Dict[str, any]],
    topk: int = 3,
    exclude_prefix: Optional[str] = None,
    current_category: str = "",
) -> List[Dict[str, any]]:
    """
    在历史新闻库中检索与当前新闻最相近的记录。

    Args:
        text: 当前新闻文本。
        library: [{"id", "title", "text", "keywords", "category"}, ...]
        topk: 返回条数。
        exclude_prefix: 排除正文以此开头的记录（用于排除当前新闻自身）。

    Returns:
        [{"id", "title", "snippet", "category", "similarity", "matched_keywords"}]
    """
    cur_kws = {t["word"] for t in extract_keywords(text, topk=15)}
    results: List[Dict[str, any]] = []
    for item in library:
        content = item.get("text", "")
        if exclude_prefix and content.startswith(exclude_prefix[:50]):
            continue
        lib_kws = set(item.get("keywords", []))
        if not lib_kws and content:
            # 未预存关键词的记录（如内置示例数据集），现场提取关键词
            lib_kws = {t["word"] for t in extract_keywords(content, topk=15)}
        if not cur_kws or not lib_kws:
            sim = 0.0
        else:
            inter = cur_kws & lib_kws
            sim = len(inter) / (len(cur_kws | lib_kws) ** 0.5)  # 加权 Jaccard
        if sim <= 0:
            continue
        results.append({
            "id": item.get("id", ""),
            "title": item.get("title", content[:30]),
            "snippet": content[:80] + ("..." if len(content) > 80 else ""),
            "category": item.get("category", ""),
            "similarity": round(sim, 4),
            "matched_keywords": sorted(cur_kws & lib_kws)[:6],
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    top = results[:topk]
    # 兜底：关键词无交集时，按同类目推荐（保证始终有相关推荐）
    if not top and current_category:
        for item in library:
            content = item.get("text", "")
            if exclude_prefix and content.startswith(exclude_prefix[:50]):
                continue
            if item.get("category") != current_category:
                continue
            top.append({
                "id": item.get("id", ""),
                "title": item.get("title", content[:30]),
                "snippet": content[:80] + ("..." if len(content) > 80 else ""),
                "category": item.get("category", ""),
                "similarity": 0.01,
                "matched_keywords": [],
            })
            if len(top) >= topk:
                break
    return top
