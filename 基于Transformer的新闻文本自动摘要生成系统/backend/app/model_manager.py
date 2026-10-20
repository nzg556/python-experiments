"""
模型管理器：负责加载与管理基于 Transformer 的中文新闻摘要模型。

支持的模型：
- csebuetnlp/mT5_multilingual_XLSum  : 多语言 T5，支持中文摘要
- IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese : 中文 Pegasus 摘要模型

采用懒加载机制，首次调用时才加载模型，避免启动时占用过多内存。
"""
import os
import threading
from typing import Optional, Dict, Any

# 必须在 transformers / huggingface_hub 导入前配置镜像源与缓存目录
from . import runtime_env  # noqa: F401

import torch
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from .pegasus_tokenizer import PegasusTokenizer

# Pegasus 模型仓库名（单独常量，多处复用）
PEGASUS_REPO = "IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"
# 项目内置的 Pegasus 词表（仓库未发布标准 tokenizer，详见 pegasus_tokenizer.py）
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILTIN_PEGASUS_VOCAB = os.path.join(_BACKEND_DIR, "pegasus_resources", "vocab.txt")


# 支持的模型配置
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "csebuetnlp/mT5_multilingual_XLSum": {
        "name": "mT5 多语言摘要模型",
        "description": "基于 mT5 的多语言文本摘要模型，支持中文，参数量约 5.8 亿。",
        "type": "mt5",
        "max_source_length": 512,
    },
    "IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese": {
        "name": "Pegasus 中文摘要模型",
        "description": "基于 Pegasus 的中文新闻摘要模型，参数量约 2.38 亿。",
        "type": "pegasus",
        "max_source_length": 1024,
    },
}


class ModelManager:
    """Transformer 摘要模型管理器，支持多模型懒加载与缓存。"""

    def __init__(self, default_model: Optional[str] = None, device: str = "auto"):
        """
        初始化模型管理器。

        Args:
            default_model: 默认使用的模型名称，为 None 时使用 AVAILABLE_MODELS 的第一个。
            device: 推理设备，可选 "auto"、"cpu"、"cuda"。
        """
        self._device = self._resolve_device(device)
        self._default_model = default_model or list(AVAILABLE_MODELS.keys())[0]
        # 已加载的模型与分词器缓存
        self._models: Dict[str, PreTrainedModel] = {}
        self._tokenizers: Dict[str, PreTrainedTokenizer] = {}
        # 加载锁，保证线程安全
        self._lock = threading.Lock()
        logger.info(
            f"ModelManager 初始化完成，默认模型: {self._default_model}, 设备: {self._device}"
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """解析推理设备。"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("请求使用 CUDA 但不可用，回退到 CPU")
            return "cpu"
        return device

    @property
    def device(self) -> str:
        return self._device

    @property
    def default_model(self) -> str:
        return self._default_model

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """返回所有可用模型及其元信息。"""
        return AVAILABLE_MODELS

    def is_model_loaded(self, model_name: str) -> bool:
        """检查指定模型是否已加载到内存。"""
        return model_name in self._models

    def _resolve_pegasus_vocab(self) -> str:
        """
        获取 Pegasus 词表文件路径。

        该模型仓库只发布了 vocab.txt 与自定义分词器代码，没有标准 tokenizer
        文件，AutoTokenizer 无法加载。优先使用项目内置词表，其次从镜像下载。
        """
        if os.path.isfile(_BUILTIN_PEGASUS_VOCAB):
            return _BUILTIN_PEGASUS_VOCAB
        logger.info("内置词表不存在，尝试从模型仓库下载 vocab.txt ...")
        return hf_hub_download(PEGASUS_REPO, "vocab.txt")

    def _load_pegasus(self) -> None:
        """加载 Randeng-Pegasus：自定义 WordPiece+jieba 分词器 + seq2seq 权重。"""
        # 1. 分词器：使用仓库官方自定义实现（BERT WordPiece + jieba）
        vocab_path = self._resolve_pegasus_vocab()
        tokenizer = PegasusTokenizer(vocab_file=vocab_path)

        # 2. 模型权重：优先从本地缓存快照加载（仓库含 0 字节 safetensors 占位，
        #    直接 from_pretrained(repo_id) 可能误选），缓存缺失时再走在线下载
        model_dir = None
        cached_config = try_to_load_from_cache(PEGASUS_REPO, "config.json")
        if cached_config and os.path.isfile(cached_config):
            candidate = os.path.dirname(cached_config)
            if os.path.isfile(os.path.join(candidate, "pytorch_model.bin")):
                model_dir = candidate

        load_kwargs = {"torch_dtype": torch.float32}  # config 声明 fp16，CPU 推理需 fp32
        if model_dir is not None:
            logger.info(f"从本地缓存加载 Pegasus 权重: {model_dir}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_dir, local_files_only=True, **load_kwargs)
        else:
            logger.info("本地无缓存，从模型仓库下载 Pegasus 权重 ...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                PEGASUS_REPO, **load_kwargs)

        model.to(self._device)
        model.eval()
        self._models[PEGASUS_REPO] = model
        self._tokenizers[PEGASUS_REPO] = tokenizer

    def _load_model(self, model_name: str) -> None:
        """
        加载指定模型与分词器到缓存中（线程安全）。

        Args:
            model_name: HuggingFace 模型名称或本地路径。
        """
        if model_name in self._models:
            return

        with self._lock:
            # 双重检查，避免并发重复加载
            if model_name in self._models:
                return

            if model_name not in AVAILABLE_MODELS:
                raise ValueError(
                    f"不支持的模型: {model_name}，可用模型: {list(AVAILABLE_MODELS.keys())}"
                )

            logger.info(f"开始加载模型: {model_name}")
            try:
                if model_name == PEGASUS_REPO:
                    # Pegasus 仓库没有标准 tokenizer 文件，需要专用加载流程
                    self._load_pegasus()
                else:
                    # 使用经典 sentencepiece 分词器（use_fast=False），
                    # 兼容 transformers 4.x/5.x，避免 tiktoken 后端解析 spiece.model 失败
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name, use_fast=False)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    model.to(self._device)
                    model.eval()
                    self._models[model_name] = model
                    self._tokenizers[model_name] = tokenizer
                logger.info(f"模型加载成功: {model_name}")
            except Exception as e:
                logger.error(f"模型加载失败: {model_name}, 错误: {e}")
                raise RuntimeError(f"加载模型 {model_name} 失败: {e}") from e

    def summarize(
        self,
        text: str,
        model_name: Optional[str] = None,
        max_length: int = 128,
        min_length: int = 32,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        对输入文本生成摘要。

        Args:
            text: 输入的新闻文本。
            model_name: 使用的模型名称，为 None 时使用默认模型。
            max_length: 生成摘要的最大长度（token 数）。
            min_length: 生成摘要的最小长度（token 数）。
            num_beams: Beam Search 的 beam 数量。
            length_penalty: 长度惩罚系数，>1 鼓励更长输出，<1 鼓励更短输出。
            do_sample: 是否启用采样生成（每次结果不同，用于多样化摘要）。
            temperature: 采样温度，越高越随机。
            top_p: 核采样阈值，0.9 表示保留概率累积前 90% 的 token。
            seed: 随机种子，用于复现采样结果。

        Returns:
            包含摘要文本、使用的模型、生成参数等信息的字典。
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        model_name = model_name or self._default_model
        # 确保模型已加载
        self._load_model(model_name)

        model = self._models[model_name]
        tokenizer = self._tokenizers[model_name]
        model_config = AVAILABLE_MODELS[model_name]
        max_source_length = model_config.get("max_source_length", 512)

        try:
            # 文本编码（单条请求无需 padding，避免 CPU 上按最大长度补全导致推理过慢）
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_source_length,
                truncation=True,
            ).to(self._device)

            # 参数校验
            max_length = max(int(max_length), 8)
            min_length = max(int(min_length), 1)
            min_length = min(min_length, max_length - 1)
            num_beams = max(int(num_beams), 1)
            length_penalty = float(length_penalty)

            # 生成摘要（参数优化以减少幻觉与重复）
            generate_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": num_beams,
                "length_penalty": length_penalty,
                "early_stopping": True,
                "no_repeat_ngram_size": 3,       # 防止3-gram重复，减少幻觉
                "repetition_penalty": 1.2,       # 重复惩罚，抑制生成原文中不存在的词
                "num_return_sequences": 1,
            }
            if do_sample:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = float(temperature)
                generate_kwargs["top_p"] = float(top_p)
                if seed is not None:
                    torch.manual_seed(seed)
            with torch.no_grad():
                summary_ids = model.generate(**generate_kwargs)

            # 解码
            summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # 清理摘要中的多余空白
            summary = summary.strip()
            if not summary:
                summary = "（模型未能生成有效摘要，请尝试调整参数或更换模型）"

            logger.info(
                f"摘要生成成功 | 模型: {model_name} | 输入长度: {len(text)} | 摘要长度: {len(summary)}"
            )

            return {
                "summary": summary,
                "model": model_name,
                "model_display_name": model_config["name"],
                "params": {
                    "max_length": max_length,
                    "min_length": min_length,
                    "num_beams": num_beams,
                    "length_penalty": length_penalty,
                },
                "input_length": len(text),
                "summary_length": len(summary),
            }

        except Exception as e:
            logger.error(f"摘要生成失败: {e}")
            raise RuntimeError(f"摘要生成过程中发生错误: {e}") from e


# 全局单例实例
_model_manager: Optional[ModelManager] = None
_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """获取全局 ModelManager 单例（线程安全）。"""
    global _model_manager
    if _model_manager is None:
        with _manager_lock:
            if _model_manager is None:
                default_model = os.getenv("DEFAULT_MODEL", list(AVAILABLE_MODELS.keys())[0])
                device = os.getenv("DEVICE", "auto")
                _model_manager = ModelManager(default_model=default_model, device=device)
    return _model_manager
