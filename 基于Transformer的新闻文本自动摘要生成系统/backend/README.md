# 后端服务

基于 FastAPI 的新闻摘要后端服务。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境配置

复制 `.env.example` 为 `.env` 并按需修改：

```bash
cp .env.example .env
```

主要配置项：
- `DEFAULT_MODEL`: 默认使用的摘要模型
- `DEVICE`: 推理设备（auto/cpu/cuda）
- `CORS_ORIGINS`: 允许跨域的前端地址

## 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问 http://localhost:8000/docs 查看交互式 API 文档。

## API 接口

### POST /api/summarize
生成新闻摘要。

请求体：
```json
{
  "text": "新闻文本",
  "model": "csebuetnlp/mT5_multilingual_XLSum",
  "max_length": 128,
  "min_length": 32,
  "num_beams": 4,
  "length_penalty": 1.0
}
```

### POST /api/evaluate
计算 ROUGE 分数。

请求体：
```json
{
  "hypothesis": "生成的摘要",
  "reference": "参考摘要"
}
```

### GET /api/models
获取可用模型列表。

### GET /api/health
健康检查。
