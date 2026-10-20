"""
摘要生成路由：POST /api/summarize 与全功能端点 POST /api/summarize/full
"""
from fastapi import APIRouter, HTTPException
from loguru import logger

from ..model_manager import get_model_manager
from ..schemas import (
    SummarizeRequest, SummarizeResponse,
    FullSummarizeRequest, FullSummarizeResponse,
)
from .. import text_tools
from .. import translation
from .. import attention as attention_mod
from ..news_library import add_document, load_library

router = APIRouter(prefix="/api", tags=["摘要生成"])


@router.post("/summarize", response_model=SummarizeResponse, summary="生成新闻摘要")
async def summarize(request: SummarizeRequest):
    """
    对输入的新闻文本生成摘要。

    - **text**: 新闻文本内容（必填）
    - **model**: 模型名称，为空使用默认模型
    - **max_length**: 摘要最大长度
    - **min_length**: 摘要最小长度
    - **num_beams**: Beam Search 数量
    - **length_penalty**: 长度惩罚系数
    """
    try:
        manager = get_model_manager()
        result = manager.summarize(
            text=request.text,
            model_name=request.model,
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams,
            length_penalty=request.length_penalty,
        )
        return SummarizeResponse(**result)
    except ValueError as e:
        logger.warning(f"参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"摘要生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("未预期的错误")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


# 多粒度摘要对应的 token 长度（中文字符与 token 约 1:1）
_GRANULARITY_PARAMS = {
    "short": {"max_length": 64, "min_length": 8},    # 约 50 字，一句话梗概
    "medium": {"max_length": 176, "min_length": 64},  # 约 150 字
    "detailed": {"max_length": 336, "min_length": 128},  # 约 300 字
}


@router.post("/summarize/full", response_model=FullSummarizeResponse,
             summary="全功能摘要（润色/对比/多粒度/分类/可信度/高亮/英文/热力图/推荐）")
async def summarize_full(request: FullSummarizeRequest):
    """一次调用返回论文演示所需的全部增强分析结果。"""
    try:
        manager = get_model_manager()
        raw_text = text_tools.clean_news_text(request.text)

        # ---------- 0. 英译中（若输入为英文且开启翻译） ----------
        original_lang = "zh"
        if translation.is_english_text(raw_text):
            original_lang = "en"
            if request.enable_zh_translate:
                translated = translation.translate_to_chinese(raw_text)
                if translated:
                    raw_text = translated
                else:
                    logger.warning("英译中翻译模型不可用，将直接对英文原文生成摘要")

        # ---------- 1. 生成式摘要（初稿） ----------
        # 长文本优化：模型输入长度有限（mT5 512 / Pegasus 1024 token），
        # 超长文本先抽取关键句压缩到模型可接受范围，避免截断导致的幻觉与低可信度
        model_input = raw_text
        model_cfg = manager.list_models().get(request.model or manager.default_model, {})
        max_src = model_cfg.get("max_source_length", 512)
        if len(raw_text) > int(max_src * 1.5):
            key = text_tools.extractive_summary(raw_text, max_sentences=8)
            if key["summary"] and len(key["summary"]) < len(raw_text):
                model_input = key["summary"]
                logger.info(f"长文本预压缩: {len(raw_text)} -> {len(model_input)} 字")

        gen = manager.summarize(
            text=model_input,
            model_name=request.model,
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams,
            length_penalty=request.length_penalty,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        draft = gen["summary"]

        # ---------- 2. 摘要润色 ----------
        polish_result = text_tools.polish_summary(draft, formal_style=True)
        polished = polish_result["polished"] if request.polish else draft
        polished = text_tools.auto_punctuate(polished)
        final_summary = polished or draft

        # ---------- 3. 字数统计 ----------
        stats = text_tools.count_stats(raw_text, final_summary)

        # ---------- 4. 抽取式摘要（对比实验） ----------
        extr = text_tools.extractive_summary(raw_text, max_sentences=3)

        # ---------- 5. 高亮定位 ----------
        hl = text_tools.align_highlights(raw_text, final_summary)

        # ---------- 6. 关键词 ----------
        keywords = text_tools.extract_keywords(raw_text, topk=8)

        # ---------- 7. 可信度评估（基于润色后摘要） ----------
        credibility = text_tools.credibility_score(raw_text, final_summary)

        # ---------- 8. 主题分类 ----------
        classification = text_tools.classify_news(raw_text)

        # ---------- 9. 多粒度摘要（可选） ----------
        granularities: dict = {}
        for g in (request.granularities or []):
            g = g if g in _GRANULARITY_PARAMS else "medium"
            if g in granularities:
                continue
            try:
                p = _GRANULARITY_PARAMS[g]
                r = manager.summarize(
                    text=raw_text, model_name=request.model,
                    max_length=p["max_length"], min_length=p["min_length"],
                    num_beams=max(request.num_beams, 2), length_penalty=1.0)
                granularities[g] = text_tools.auto_punctuate(r["summary"])
            except Exception as e:
                logger.warning(f"多粒度摘要 {g} 生成失败: {e}")

        # ---------- 10. 英文摘要（可选） ----------
        english = None
        if request.enable_english:
            english = translation.translate_to_english(final_summary)
            if english is None:
                english = "（英文摘要服务暂不可用：翻译模型未安装或下载失败）"

        # ---------- 11. 注意力热力图（可选） ----------
        attention_tokens = []
        if request.enable_attention:
            try:
                att = attention_mod.get_cross_attention(
                    raw_text, final_summary, model_name=gen["model"])
                attention_tokens = att["tokens"]
            except Exception as e:
                logger.warning(f"注意力可视化失败: {e}")

        # ---------- 12. 相关新闻推荐（可选） ----------
        recommendations = []
        if request.enable_recommend:
            try:
                doc_id = add_document(
                    raw_text,
                    title=raw_text[:30],
                    keywords=[k["word"] for k in keywords],
                    category=classification["category"])
                # 推荐库 = 历史库 + 内置示例 + 新闻数据查询库（LCSTS，预计算关键词）
                from ..dataset import get_all_samples
                from ..news_query_db import get_db as get_query_db
                query_db = get_query_db()
                # 用查询库的推荐 API（预计算关键词，毫秒级响应）
                recs = query_db.recommend(raw_text, topk=5)
                # 合并内置示例 + 历史库的推荐
                from ..dataset import get_all_samples as _gas
                corpus = load_library() + [
                    {
                        "id": f"sample-{s['id']}",
                        "title": s.get("title", ""),
                        "text": s.get("text", ""),
                        "keywords": [],
                        "category": s.get("category", ""),
                    }
                    for s in _gas()
                ]
                recs2 = text_tools.recommend_similar(
                    raw_text, corpus, topk=3,
                    exclude_prefix=raw_text[:50],
                    current_category=classification["category"])
                recommendations = (recs + recs2)[:5]
            except Exception as e:
                logger.warning(f"推荐生成失败: {e}")

        return FullSummarizeResponse(
            summary=draft,
            polished_summary=final_summary,
            polish_changes=polish_result["changes"] if request.polish else [],
            model=gen["model"],
            model_display_name=gen["model_display_name"],
            params=gen["params"],
            stats=stats,
            extractive_summary=extr["summary"],
            extractive_detail=extr["sentences"],
            highlights=hl["sentences"],
            highlight_coverage=hl["coverage"],
            keywords=keywords,
            credibility=credibility,
            classification=classification,
            granularities=granularities,
            english_summary=english,
            attention=attention_tokens,
            recommendations=recommendations,
            readability=text_tools.readability_score(final_summary),
            original_lang=original_lang,
        )
    except ValueError as e:
        logger.warning(f"参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"摘要生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("全功能摘要未预期的错误")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")
