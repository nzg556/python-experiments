# 基于 Transformer 的新闻文本自动摘要生成系统 - 独立审查

## 检查点

- [x] CP-R1: 后端摘要 API 可正常调用并返回非空摘要
  - **Type**: `rule`
  - **Covers**: AC-1, AC-3
  - **Evidence**: summarize.py 路由 + ModelManager.summarize 方法完整实现，空摘要有兜底文案。实测 API 返回非空 summary。

- [x] CP-R2: 后端提供至少 2 个可用模型
  - **Type**: `rule`
  - **Covers**: AC-4
  - **Evidence**: AVAILABLE_MODELS 包含 mT5 与 Pegasus 两个模型，/api/models 接口实测返回 2 个模型。

- [x] CP-R3: ROUGE 评估接口返回三个指标分数且值在 [0,1]
  - **Type**: `rule`
  - **Covers**: AC-6
  - **Evidence**: compute_rouge 返回 rouge1/rouge2/rougeL，precision/recall/f1 均在 [0,1]。实测相同文本返回 F1=1.0。

- [x] CP-R4: 前端包含摘要生成、历史记录、评估、关于四个页面
  - **Type**: `rule`
  - **Covers**: AC-2, AC-5, AC-7
  - **Evidence**: router/index.js 定义 4 条路由，views/ 下 SummarizeView、HistoryView、EvaluateView、AboutView 均存在。

- [x] CP-R5: 模型训练脚本包含训练循环与 checkpoint 保存逻辑
  - **Type**: `rule`
  - **Covers**: AC-7
  - **Evidence**: train.py 使用 Seq2SeqTrainer，配置 save_steps/save_total_limit，训练后 save_model。

- [x] CP-U1: 前端 UI/UX 质量
  - **Type**: `rubric`
  - **Covers**: AC-8
  - **Scale**: 1-5
  - **Anchors**: 1 = 界面粗糙、布局混乱；3 = 基本可用、布局一般；5 = 美观现代、交互流畅、响应式良好
  - **Pass Threshold**: >= 4
  - **Evidence**: 评分 4/5。Element Plus 布局美观，渐变 Header，左右分栏，参数滑块，loading/empty 状态，复制按钮，历史持久化。

- [x] CP-U2: 代码工程质量
  - **Type**: `rubric`
  - **Covers**: AC-9
  - **Scale**: 1-5
  - **Anchors**: 1 = 代码混乱、无文档；3 = 基本模块化、有少量注释；5 = 模块清晰、注释完善、易于理解
  - **Pass Threshold**: >= 4
  - **Evidence**: 评分 4/5。后端分层清晰（routers/model_manager/schemas/utils），懒加载+线程安全，Pydantic 校验，loguru 日志。前端 api/stores/router/views/layouts 分离。已修复单例无锁、请求体限制、ROUGE 归一化等问题。

- [x] CP-U3: 文档完整性
  - **Type**: `rubric`
  - **Covers**: AC-12
  - **Scale**: 1-5
  - **Anchors**: 1 = 缺失；3 = 基本；5 = 详细清晰
  - **Pass Threshold**: >= 4
  - **Evidence**: 评分 4/5。根 README 含架构图、技术栈、安装运行、API 文档、训练指南、目录结构。backend/frontend 各有 README。

## Review History

### Review R1
- **Result**: `pass`
- **Evidence**: 
  - CP-R1 (rule): PASS - summarize 路由与 ModelManager.summarize 方法正确实现，空摘要有兜底文案保证非空
  - CP-R2 (rule): PASS - AVAILABLE_MODELS 包含 mT5 与 Pegasus 两个模型
  - CP-R3 (rule): PASS - compute_rouge 返回 rouge1/rouge2/rougeL，值均在 [0,1]
  - CP-R4 (rule): PASS - router 定义 4 条路由，views/ 下对应 4 个组件存在
  - CP-R5 (rule): PASS - train.py 使用 Seq2SeqTrainer，含 save_steps 与 save_model
  - CP-U1 (rubric): 4/5 - Element Plus 布局美观，状态处理完整
  - CP-U2 (rubric): 4/5 - 后端分层清晰，懒加载+线程安全
  - CP-U3 (rubric): 4/5 - README 详尽
- **Actionable 修复**: 单例加锁、text 最大长度限制、ROUGE key 映射简化、前端 min<max 校验、evaluate.py import 路径
- **Conclusion**: 所有检查点通过，评分项达标，无阻塞性问题
