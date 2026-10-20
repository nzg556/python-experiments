# 基于 Transformer 的新闻文本自动摘要生成系统 - 产品需求文档

## Overview
- **Summary**: 构建一个基于 Transformer 架构的中文新闻文本自动摘要生成系统，提供后端 API 服务与前端 Web 界面，支持用户输入新闻文本并自动生成摘要。
- **Purpose**: 完成毕业设计，展示 Transformer 模型在自然语言处理（文本摘要）任务中的应用，提供完整的前后端可运行系统。
- **Target Users**: 毕业设计评审老师、对 NLP 文本摘要感兴趣的开发者与学生。

## Goals
- 实现基于 Transformer（mT5/Pegasus）的中文新闻摘要生成模型。
- 提供 RESTful 后端 API，支持文本摘要、模型选择、参数调节。
- 提供现代化前端 Web 界面，支持文本输入、摘要展示、历史记录。
- 提供模型训练脚本、数据处理脚本与评估脚本，形成完整的工程闭环。
- 编写清晰的 README 与使用文档，便于部署与答辩演示。

## Non-Goals
- 不实现分布式训练与大规模模型微调（仅提供单机可运行的训练脚本）。
- 不实现用户认证与多租户系统。
- 不实现移动端 App（仅 Web 端）。
- 不实现实时流式生成（仅同步返回摘要）。

## Background & Context
- Transformer 架构（Vaswani et al., 2017）是当前 NLP 的主流基础架构。
- 文本摘要分为抽取式（Extractive）与生成式（Abstractive），本项目采用生成式摘要。
- 中文新闻摘要常用数据集：LCSTS、NLPCC、CNN/DailyMail（中文翻译版）。
- 常用预训练模型：mT5（多语言 T5）、Pegasus（专为摘要设计）、BART。

## Functional Requirements
- **FR-1**: 用户可在前端输入或粘贴中文新闻文本，点击生成摘要按钮。
- **FR-2**: 后端接收文本，调用 Transformer 模型生成摘要并返回。
- **FR-3**: 前端展示生成的摘要结果。
- **FR-4**: 用户可调节生成参数（最大长度 max_length、最小长度 min_length、beam search 数量 num_beams、长度惩罚 length_penalty）。
- **FR-5**: 支持多个预训练模型选择（至少 2 个中文摘要模型）。
- **FR-6**: 前端保存摘要历史记录（本地存储），可查看与删除。
- **FR-7**: 提供摘要质量评估功能（ROUGE-1、ROUGE-2、ROUGE-L 分数）。
- **FR-8**: 后端提供模型健康检查与元信息接口。
- **FR-9**: 提供模型训练脚本，支持在 LCSTS 等数据集上微调。
- **FR-10**: 提供数据预处理脚本，将原始数据转换为模型可用格式。
- **FR-11**: 前端提供示例新闻文本，方便快速体验。
- **FR-12**: 系统提供使用说明与文档（README）。

## Non-Functional Requirements
- **NFR-1**: 后端 API 响应时间不超过 30 秒（单条摘要，CPU 环境）。
- **NFR-2**: 前端界面美观、响应式，适配桌面浏览器。
- **NFR-3**: 代码结构清晰，模块划分合理，便于理解与维护。
- **NFR-4**: 依赖项明确，提供 requirements.txt 与 package.json。
- **NFR-5**: 后端使用 FastAPI 框架，前端使用 Vue 3 + Element Plus。
- **NFR-6**: 模型加载与推理有错误处理，不会因异常导致服务崩溃。

## Constraints
- **Technical**: 
  - 后端语言 Python 3.9+，框架 FastAPI。
  - 前端框架 Vue 3 + Vite + Element Plus。
  - 深度学习框架 PyTorch + HuggingFace Transformers。
  - 模型使用 HuggingFace Hub 上的开源中文摘要模型。
- **Business**: 毕业设计项目，需在本地可运行。
- **Dependencies**: 
  - `fastapi`, `uvicorn`, `transformers`, `torch`, `rouge`, `pydantic`
  - `vue`, `element-plus`, `axios`, `pinia`

## Assumptions
- 用户具备基本的 Python 与 Node.js 环境。
- 模型可从 HuggingFace Hub 下载（或提供离线加载方案）。
- 默认使用 CPU 推理，可支持 GPU（如有）。
- 摘要质量评估仅作为辅助功能，不保证达到 SOTA。

## Acceptance Criteria

### AC-1: 后端摘要 API 可用
- **Type**: `rule`
- **Given**: 后端服务已启动且模型已加载
- **When**: 客户端向 `/api/summarize` 发送包含新闻文本的 POST 请求
- **Then**: 服务返回 HTTP 200，响应体包含生成的摘要文本
- **Pass Condition**: 接口返回 200 状态码且 `summary` 字段非空
- **Evidence**: 使用 curl 或 Swagger UI 测试接口，记录响应

### AC-2: 前端文本输入与摘要展示
- **Type**: `rule`
- **Given**: 前端页面已打开
- **When**: 用户在文本框输入新闻并点击"生成摘要"
- **Then**: 页面显示加载状态，完成后展示摘要结果
- **Pass Condition**: 摘要结果区域显示非空文本
- **Evidence**: 浏览器中完成一次摘要生成并截图/记录

### AC-3: 生成参数可调节
- **Type**: `rule`
- **Given**: 前端摘要参数面板可见
- **When**: 用户修改 max_length、num_beams 等参数并生成摘要
- **Then**: 后端使用用户指定的参数进行生成
- **Pass Condition**: 请求 payload 中包含用户设置的参数值
- **Evidence**: 浏览器 Network 面板查看请求参数

### AC-4: 多模型选择
- **Type**: `rule`
- **Given**: 前端模型选择下拉框可见
- **When**: 用户选择不同的模型并生成摘要
- **Then**: 后端切换到对应模型进行推理
- **Pass Condition**: 至少支持 2 个不同的模型
- **Evidence**: 切换模型后接口返回的 `model` 字段不同

### AC-5: 历史记录功能
- **Type**: `rule`
- **Given**: 用户已生成至少一条摘要
- **When**: 用户查看历史记录
- **Then**: 显示之前生成的原文与摘要，可删除
- **Pass Condition**: 历史记录列表非空且可删除单条
- **Evidence**: 前端历史记录页面操作记录

### AC-6: ROUGE 评估功能
- **Type**: `rule`
- **Given**: 用户提供原文摘要与参考摘要
- **When**: 点击评估
- **Then**: 返回 ROUGE-1、ROUGE-2、ROUGE-L 的 F1 分数
- **Pass Condition**: 返回三个分数且值在 [0, 1] 范围内
- **Evidence**: 调用评估接口并记录返回分数

### AC-7: 训练脚本可运行
- **Type**: `rule`
- **Given**: 已准备训练数据与依赖环境
- **When**: 执行训练脚本
- **Then**: 脚本启动训练循环，输出 loss 与保存检查点
- **Pass Condition**: 训练至少运行 1 个 epoch 并保存 checkpoint
- **Evidence**: 训练日志输出与 checkpoint 文件存在

### AC-8: 前端界面质量
- **Type**: `rubric`
- **Dimension**: 前端 UI/UX 质量
- **Scale**: 1-5
- **Anchors**: 1 = 界面粗糙、布局混乱；3 = 基本可用、布局一般；5 = 美观现代、交互流畅、响应式良好
- **Pass Threshold**: >= 4
- **Evidence**: 前端页面实际运行截图与交互体验

### AC-9: 代码结构与可维护性
- **Type**: `rubric`
- **Dimension**: 代码工程质量
- **Scale**: 1-5
- **Anchors**: 1 = 代码混乱、无文档；3 = 基本模块化、有少量注释；5 = 模块清晰、注释完善、易于理解
- **Pass Threshold**: >= 4
- **Evidence**: 代码目录结构与关键文件审查

### AC-10: 系统端到端可运行
- **Type**: `rule`
- **Given**: 前后端均已启动
- **When**: 用户在前端完成一次完整的摘要生成流程
- **Then**: 从输入到展示全流程无报错
- **Pass Condition**: 完整流程执行成功
- **Evidence**: 端到端测试记录

## Open Questions
- [ ] 是否需要提供预训练好的本地模型文件，还是运行时从 HuggingFace 下载？（默认：运行时下载，提供离线加载说明）
- [ ] 训练数据集是否需要随项目提供？（默认：提供下载脚本与说明，不附带原始数据）
