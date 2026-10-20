"""
Pydantic 数据模型：定义 API 请求与响应的数据结构。
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    """摘要生成请求。"""

    text: str = Field(..., description="输入的新闻文本", min_length=1, max_length=5000)
    model: Optional[str] = Field(None, description="使用的模型名称，为空则使用默认模型")
    max_length: int = Field(128, description="摘要最大长度", ge=8, le=512)
    min_length: int = Field(32, description="摘要最小长度", ge=1, le=256)
    num_beams: int = Field(4, description="Beam Search 数量", ge=1, le=10)
    length_penalty: float = Field(1.0, description="长度惩罚系数", ge=0.1, le=10.0)


class SummarizeResponse(BaseModel):
    """摘要生成响应。"""

    summary: str = Field(..., description="生成的摘要文本")
    model: str = Field(..., description="使用的模型名称")
    model_display_name: str = Field(..., description="模型显示名称")
    params: Dict[str, Any] = Field(..., description="实际使用的生成参数")
    input_length: int = Field(..., description="输入文本长度")
    summary_length: int = Field(..., description="摘要长度")


class EvaluateRequest(BaseModel):
    """ROUGE 评估请求。"""

    hypothesis: str = Field(..., description="生成的摘要")
    reference: str = Field(..., description="参考摘要（标准答案）")


class RougeScore(BaseModel):
    """单个 ROUGE 指标分数。"""

    precision: float
    recall: float
    f1: float


class EvaluateResponse(BaseModel):
    """ROUGE 评估响应。"""

    rouge1: RougeScore
    rouge2: RougeScore
    rougeL: RougeScore


class ModelInfo(BaseModel):
    """模型信息。"""

    name: str
    display_name: str
    description: str
    type: str


class ModelsResponse(BaseModel):
    """模型列表响应。"""

    models: List[ModelInfo]
    default_model: str


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    device: str
    default_model: str


# ---------------------------------------------------------------------------
# 以下为增强功能数据模型
# ---------------------------------------------------------------------------
class FullSummarizeRequest(BaseModel):
    """全功能摘要请求（摘要 + 润色 + 对比 + 分类 + 可信度 + 高亮 + 可选项）。"""

    text: str = Field(..., description="输入的新闻文本", min_length=1, max_length=5000)
    model: Optional[str] = Field(None, description="使用的模型名称")
    max_length: int = Field(128, ge=8, le=512)
    min_length: int = Field(32, ge=1, le=256)
    num_beams: int = Field(4, ge=1, le=10)
    length_penalty: float = Field(1.0, ge=0.1, le=10.0)
    polish: bool = Field(True, description="是否对初稿摘要润色")
    granularities: List[str] = Field(
        default_factory=lambda: [], description="额外多粒度摘要：short/medium/detailed")
    enable_english: bool = Field(False, description="是否生成英文简要摘要")
    enable_attention: bool = Field(False, description="是否返回注意力热力图数据")
    enable_recommend: bool = Field(True, description="是否返回相关新闻推荐")
    enable_zh_translate: bool = Field(True, description="英文输入时是否自动翻译为中文")
    do_sample: bool = Field(False, description="是否启用采样生成（每次结果不同）")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="采样温度，越高越随机")
    top_p: float = Field(0.9, ge=0.1, le=1.0, description="核采样阈值")


class StatsInfo(BaseModel):
    """字数统计信息。"""

    original_chars: int
    summary_chars: int
    compression_ratio: float
    compression_percent: float
    original_sentences: int
    summary_sentences: int


class HighlightSentence(BaseModel):
    """原文句子（含是否为摘要参考关键句）。"""

    index: int
    sentence: str
    is_key: bool
    score: float


class CredibilityInfo(BaseModel):
    """可信度评估结果。"""

    score: float
    level: str
    missing_fragments: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any]
    hallucination_types: List[Dict[str, Any]] = []


class ClassificationInfo(BaseModel):
    """新闻主题分类结果。"""

    category: str
    confidence: float
    scores: Dict[str, float]
    hits: Dict[str, List[str]]


class RecommendationItem(BaseModel):
    """相关新闻推荐条目。"""

    id: str
    title: str
    snippet: str
    category: str = ""
    similarity: float
    matched_keywords: List[str] = []


class AttentionToken(BaseModel):
    """注意力热力图 token。"""

    token: str
    weight: float


class ReadabilityInfo(BaseModel):
    """中文可读性指标。"""

    score: float
    level: str
    avg_sentence_len: float
    avg_word_len: float


class FullSummarizeResponse(BaseModel):
    """全功能摘要响应。"""

    summary: str
    polished_summary: str
    polish_changes: List[str] = []
    model: str
    model_display_name: str
    params: Dict[str, Any]
    stats: StatsInfo
    extractive_summary: str
    extractive_detail: List[Dict[str, Any]] = []
    highlights: List[HighlightSentence] = []
    highlight_coverage: float = 0.0
    keywords: List[Dict[str, Any]] = []
    credibility: CredibilityInfo
    classification: ClassificationInfo
    granularities: Dict[str, str] = {}
    english_summary: Optional[str] = None
    attention: List[AttentionToken] = []
    recommendations: List[RecommendationItem] = []
    readability: Optional[ReadabilityInfo] = None
    original_lang: str = "zh"


class ClassifyRequest(BaseModel):
    """主题分类请求。"""

    text: str = Field(..., min_length=1, max_length=5000)


class CredibilityRequest(BaseModel):
    """可信度评估请求。"""

    original: str = Field(..., min_length=1, max_length=5000)
    summary: str = Field(..., min_length=1, max_length=2000)


class PolishRequest(BaseModel):
    """摘要润色请求。"""

    text: str = Field(..., min_length=1, max_length=2000)
    formal_style: bool = Field(True, description="是否启用正式书面风格")


class PolishResponse(BaseModel):
    """摘要润色响应。"""

    polished: str
    changes: List[str]


class ReAnalyzeRequest(BaseModel):
    """编辑后重新评估请求。"""

    original_text: str = Field(..., min_length=1, max_length=5000)
    summary: str = Field(..., min_length=1, max_length=2000)


class ReAnalyzeResponse(BaseModel):
    """编辑后重新评估响应。"""

    stats: StatsInfo
    credibility: CredibilityInfo
    highlights: List[HighlightSentence] = []
    highlight_coverage: float = 0.0
    readability: Optional[ReadabilityInfo] = None


class FeedbackRequest(BaseModel):
    """人工修正反馈（交互式编辑保存）。"""

    original_text: str
    model_summary: str
    edited_summary: str
    model: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class FeedbackResponse(BaseModel):
    """反馈保存响应。"""

    saved: bool
    total_records: int
    message: str


class ReportRequest(BaseModel):
    """摘要评估报告 PDF 导出请求（内容来自最近一次全功能摘要结果）。"""

    original_text: str = Field(..., min_length=1, max_length=5000)
    summary: str = Field(..., min_length=1, max_length=2000)
    model_display_name: str = ""
    category: str = ""
    category_confidence: float = 0.0
    keywords: List[str] = []
