"""
文本分析路由：主题分类 / 可信度评估 / 摘要润色（独立端点，便于单独调用与测试）
"""
import io
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from .. import text_tools
from ..schemas import (
    ClassifyRequest, ClassificationInfo,
    CredibilityRequest, CredibilityInfo,
    PolishRequest, PolishResponse,
    ReAnalyzeRequest, ReAnalyzeResponse,
    ReportRequest,
)

router = APIRouter(prefix="/api/analysis", tags=["文本分析"])


@router.post("/classify", response_model=ClassificationInfo, summary="新闻主题分类")
async def classify(request: ClassifyRequest):
    """识别新闻类别（财经/体育/科技/社会/娱乐/教育/健康/军事/国际/法制）。"""
    try:
        return text_tools.classify_news(request.text)
    except Exception as e:
        logger.exception("分类失败")
        raise HTTPException(status_code=500, detail=f"分类失败: {e}")


@router.post("/credibility", response_model=CredibilityInfo, summary="摘要可信度评估（幻觉检测）")
async def credibility(request: CredibilityRequest):
    """评估摘要与原文的事实一致性，输出可信度分数与疑点提示。"""
    try:
        return text_tools.credibility_score(request.original, request.summary)
    except Exception as e:
        logger.exception("可信度评估失败")
        raise HTTPException(status_code=500, detail=f"评估失败: {e}")


@router.post("/polish", response_model=PolishResponse, summary="摘要润色")
async def polish(request: PolishRequest):
    """修正语病、精简冗余语句；可选正式书面风格，适配新闻文体。"""
    try:
        return text_tools.polish_summary(request.text, formal_style=request.formal_style)
    except Exception as e:
        logger.exception("润色失败")
        raise HTTPException(status_code=500, detail=f"润色失败: {e}")


@router.post("/re-analyze", response_model=ReAnalyzeResponse, summary="编辑后重新评估")
async def re_analyze(request: ReAnalyzeRequest):
    """编辑修正摘要后，重新计算字数统计、可信度评估与高亮定位。"""
    try:
        stats = text_tools.count_stats(request.original_text, request.summary)
        credibility = text_tools.credibility_score(request.original_text, request.summary)
        hl = text_tools.align_highlights(request.original_text, request.summary)
        return ReAnalyzeResponse(
            stats=stats,
            credibility=credibility,
            highlights=hl["sentences"],
            highlight_coverage=hl["coverage"],
            readability=text_tools.readability_score(request.summary),
        )
    except Exception as e:
        logger.exception("重新评估失败")
        raise HTTPException(status_code=500, detail=f"重新评估失败: {e}")


def _build_report_pdf(request: ReportRequest) -> bytes:
    """生成中文摘要评估报告 PDF（reportlab + 内置 STSong 字体）。"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    font = "STSong-Light"

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x0, y = 20 * mm, height - 20 * mm
    line_h = 6.5 * mm

    def new_page():
        nonlocal y
        c.showPage()
        c.setFont(font, 11)
        y = height - 20 * mm

    def draw(text: str, size: int = 11, gap: float = 1.0):
        nonlocal y
        c.setFont(font, size)
        # 简单按宽度换行（中文每字约 size pt）
        max_chars = int((width - 40 * mm) / (size * 1.0))
        for i in range(0, max(len(text), 1), max_chars):
            seg = text[i:i + max_chars]
            if y < 20 * mm:
                new_page()
                c.setFont(font, size)
            c.drawString(x0, y, seg)
            y -= line_h * size / 11
        y -= line_h * (gap - 1.0)

    # 标题
    draw("新闻文本自动摘要 · 评估报告", 16, 1.5)
    draw(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 10)
    if request.model_display_name:
        draw(f"使用模型：{request.model_display_name}", 10)
    draw("─" * 46, 10)

    # 重新评估（保证报告数据实时准确）
    stats = text_tools.count_stats(request.original_text, request.summary)
    cred = text_tools.credibility_score(request.original_text, request.summary)
    cls = text_tools.classify_news(request.original_text)

    draw("一、原文", 13, 1.2)
    draw(request.original_text, 11, 1.5)
    draw("二、生成摘要", 13, 1.2)
    draw(request.summary, 11, 1.5)

    draw("三、字数统计", 13, 1.2)
    draw(f"原文：{stats['original_chars']} 字 / {stats['original_sentences']} 句", 11)
    draw(f"摘要：{stats['summary_chars']} 字 / {stats['summary_sentences']} 句", 11)
    draw(f"压缩比例：{stats['compression_ratio']}（{stats['compression_percent']}%）", 11, 1.5)

    draw("四、可信度评估（幻觉检测）", 13, 1.2)
    draw(f"综合可信度：{cred['score']} 分（{cred['level']}可信）", 11)
    d = cred["details"]
    draw(f"内容词覆盖率：{d['content_coverage']}%　数字一致性：{d['number_consistency']}%　实体一致性：{d['entity_consistency']}%", 11)
    if cred.get("hallucination_types"):
        types = "、".join(f"{h['type']}（{h['level']}）" for h in cred["hallucination_types"])
        draw(f"幻觉类型识别：{types}", 11)
    if cred.get("warnings"):
        for w in cred["warnings"]:
            draw(f"⚠ {w}", 10)
    draw("", 11, 1.0)

    draw("五、主题分类", 13, 1.2)
    draw(f"分类结果：{cls['category']}（置信度 {round(cls['confidence'] * 100, 1)}%）", 11, 1.5)
    if request.keywords:
        draw("关键词：" + "、".join(request.keywords[:10]), 11, 1.5)

    rd = text_tools.readability_score(request.summary)
    draw("六、摘要可读性", 13, 1.2)
    draw(f"易读性得分：{rd['score']}（{rd['level']}）　平均句长：{rd['avg_sentence_len']} 字　平均词长：{rd['avg_word_len']} 字", 11, 1.5)

    draw("─" * 46, 10)
    draw("本报告由基于 Transformer 的新闻文本自动摘要生成系统自动生成。", 9)

    c.save()
    return buf.getvalue()


@router.post("/export-report", summary="导出摘要评估报告（PDF）")
async def export_report(request: ReportRequest):
    """生成包含原文、摘要、统计、可信度、幻觉类型与分类结果的 PDF 报告。"""
    try:
        pdf_bytes = _build_report_pdf(request)
    except Exception as e:
        logger.exception("PDF 报告生成失败")
        raise HTTPException(status_code=500, detail=f"PDF 报告生成失败: {e}")
    filename = "summary_report.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
