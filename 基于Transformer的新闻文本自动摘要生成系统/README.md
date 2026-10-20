# 基于 Transformer 的新闻文本自动摘要生成系统

## 项目简介

本项目是一个基于 **Transformer** 架构的中文新闻文本自动摘要生成系统，采用生成式摘要（Abstractive Summarization）方法，集成了 mT5 和 Pegasus 等预训练模型，提供完整的前后端解决方案。

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    前端 (Vue 3)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │
│  │ 摘要生成  │ │ 历史记录  │ │ 摘要评估  │ │ 关于   │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘  │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP (Axios)
┌───────────────────────▼─────────────────────────────┐
│                 后端 (FastAPI)                        │
│  ┌──────────────────────────────────────────────┐   │
│  │  REST API: /api/summarize, /api/evaluate ...  │   │
│  └──────────────────────┬───────────────────────┘   │
│                         │                            │
│  ┌──────────────────────▼───────────────────────┐   │
│  │         ModelManager (Transformer 推理)        │   │
│  │   mT5 多语言模型 / Pegasus 中文摘要模型         │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 技术栈

### 后端
- **Python 3.9+**
- **FastAPI** - 异步 Web 框架
- **PyTorch** - 深度学习框架
- **Transformers** - HuggingFace 模型库
- **ROUGE** - 摘要评估指标

### 前端
- **Vue 3** - 渐进式前端框架
- **Vite** - 前端构建工具
- **Element Plus** - Vue 3 组件库
- **Pinia** - 状态管理
- **Axios** - HTTP 客户端

## 功能特性

- ✅ 中文新闻文本摘要生成
- ✅ 多模型选择（mT5 / Pegasus）
- ✅ 生成参数可调（长度、Beam Search、长度惩罚）
- ✅ 摘要历史记录（本地存储）
- ✅ ROUGE 摘要质量评估
- ✅ 示例新闻文本快速体验
- ✅ 完整的模型训练、评估、推理脚本

### 增强功能（13 项）

| # | 功能 | 说明 |
|---|------|------|
| 1 | 摘要润色 | 初稿自动修正语病、精简冗余，可选正式书面风格（规则引擎，`text_tools.py`） |
| 2 | 摘要高亮定位 | 原文中高亮生成摘要参考的关键句（字符 n-gram 相似度对齐） |
| 3 | 摘要对比 | 抽取式（TF-IDF 选句）与生成式（Transformer）左右并排对比，可作论文对比实验 |
| 4 | 字数统计 | 原文/摘要字数、压缩比例、句数统计 |
| 5 | 新闻主题分类 | 财经/体育/科技/社会/娱乐/教育/健康/军事/国际/法制 十类词典加权分类 |
| 6 | 多语言摘要 | 一键生成英文简要摘要（Helsinki-NLP/opus-mt-zh-en，懒加载，失败优雅降级） |
| 7 | 摘要可信度评估 | 内容词覆盖 + 数字一致性 + 实体一致性三维打分，提示疑似幻觉片段 |
| 8 | 批量摘要处理 | 多 txt 文件上传、后台逐篇处理、进度轮询、结果导出 Excel（openpyxl） |
| 9 | 摘要交互式编辑 | 网页直接修改摘要并保存，人工修正记录存入 `data/human_feedback.jsonl`（可作微调数据） |
| 10 | 新闻关联推荐 | 关键词检索历史新闻库（`data/news_library.jsonl` 自动积累），推荐主题相近新闻 |
| 11 | 多粒度摘要 | 一键生成短（约50字）/ 中（约150字）/ 详（约300字）三档摘要 |
| 12 | 注意力热力图 | Transformer 交叉注意力（末层均值）可视化，颜色深浅表示模型关注程度 |
| 13 | 桌面启动器 | 双击桌面图标一键启动前后端并自动打开浏览器（`创建桌面图标.bat` + `启动系统.vbs`） |

增强功能 API 一览：

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/summarize/full` | 全功能摘要（一次返回 1-5、7、10-12 项结果，英文/热力图/多粒度按开关） |
| POST | `/api/analysis/classify` | 新闻主题分类 |
| POST | `/api/analysis/credibility` | 可信度评估（幻觉检测） |
| POST | `/api/analysis/polish` | 摘要润色 |
| POST | `/api/batch/upload` | 批量上传 txt 开始处理 |
| GET | `/api/batch/status/{batch_id}` | 批处理进度与结果 |
| GET | `/api/batch/export/{batch_id}` | 导出 Excel |
| POST | `/api/feedback/save` | 保存人工修正摘要 |
| GET | `/api/feedback/list` | 人工反馈统计 |

### 一键启动（功能 13）

1. 双击项目根目录 `创建桌面图标.bat`，会在桌面生成"新闻摘要系统"快捷方式
2. 双击桌面图标：自动启动后端（8000）与前端（5173），服务就绪后自动打开浏览器
3. 如需停止服务，运行项目根目录 `停止系统.bat`

## 环境要求

- Python 3.9+
- Node.js 16+
- 建议 8GB+ 内存（模型加载需要）

## 快速开始

### 1. 克隆项目

```bash
git clone <仓库地址>
cd 基于Transformer的新闻文本自动摘要生成系统
```

### 2. 启动后端

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

后端启动后访问 http://localhost:8000/docs 查看 API 文档。

### 3. 启动前端

```bash
cd frontend
npm install
npm run dev
```

前端启动后访问 http://localhost:5173

## 使用说明

1. 在"摘要生成"页面输入或粘贴新闻文本
2. 选择模型并调整生成参数
3. 点击"生成摘要"按钮等待结果
4. 生成的摘要会自动保存到"历史记录"
5. 在"摘要评估"页面可计算 ROUGE 分数

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/summarize` | 生成新闻摘要 |
| POST | `/api/evaluate` | 评估摘要质量（ROUGE） |
| GET | `/api/models` | 获取可用模型列表 |
| GET | `/api/health` | 健康检查 |

### 摘要生成请求示例

```json
POST /api/summarize
{
  "text": "新闻文本内容...",
  "model": "csebuetnlp/mT5_multilingual_XLSum",
  "max_length": 128,
  "min_length": 32,
  "num_beams": 4,
  "length_penalty": 1.0
}
```

## 模型训练

### 数据准备

```bash
# 生成示例数据
python models/download_data.py --dataset sample --output data/raw

# 数据预处理
python models/data_preprocess.py \
  --input data/raw/sample_data.json \
  --output data/processed/sample_clean.json \
  --format json
```

### 模型微调

```bash
python models/train.py \
  --model_name csebuetnlp/mT5_multilingual_XLSum \
  --train_file data/processed/sample_clean.json \
  --output_dir models/checkpoints/mt5-finetuned \
  --num_train_epochs 3 \
  --batch_size 4
```

### 模型评估

```bash
python models/evaluate.py \
  --model_name models/checkpoints/mt5-finetuned \
  --test_file data/processed/test.json \
  --output results/eval_result.json
```

### 命令行推理

```bash
python models/inference.py \
  --model_name csebuetnlp/mT5_multilingual_XLSum \
  --text "这里是新闻文本..."
```

## 项目目录结构

```
基于Transformer的新闻文本自动摘要生成系统/
├── backend/                    # 后端服务
│   ├── app/
│   │   ├── main.py            # FastAPI 入口
│   │   ├── model_manager.py   # 模型管理器
│   │   ├── schemas.py         # Pydantic 数据模型
│   │   ├── utils.py           # 工具函数（ROUGE）
│   │   └── routers/           # API 路由
│   ├── requirements.txt
│   └── .env.example
├── frontend/                   # 前端应用
│   ├── src/
│   │   ├── views/             # 页面组件
│   │   ├── layouts/           # 布局组件
│   │   ├── router/            # 路由配置
│   │   ├── stores/            # 状态管理
│   │   ├── api/               # API 封装
│   │   └── styles/            # 全局样式
│   └── package.json
├── models/                     # 模型脚本
│   ├── data_preprocess.py     # 数据预处理
│   ├── train.py               # 模型训练
│   ├── evaluate.py            # 模型评估
│   ├── inference.py           # 命令行推理
│   └── download_data.py       # 数据下载
├── data/                       # 数据目录
└── README.md
```

## 许可证

本项目仅用于毕业设计学习用途。
