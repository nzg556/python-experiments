"""
运行时环境配置（必须在 import transformers / huggingface_hub 之前生效）。

统一处理：
1. 加载 backend/.env
2. HuggingFace 国内镜像源（默认 hf-mirror.com）
3. 模型缓存目录（默认 backend/models_cache）

任何入口（uvicorn、run.py、脚本、测试）只要 import 了 app.model_manager
或本模块，都会自动获得一致的环境配置。
"""
import os

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. 加载 .env（不存在则静默跳过）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_BACKEND_DIR, ".env"))
except ImportError:
    pass

# 2. HuggingFace 镜像源（.env 未配置时使用国内镜像）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 3. 模型缓存目录统一放在 backend/models_cache 下
_cache_dir = os.path.join(_BACKEND_DIR, "models_cache")
os.makedirs(_cache_dir, exist_ok=True)
os.environ.setdefault("HF_HOME", _cache_dir)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(_cache_dir, "hub"))
