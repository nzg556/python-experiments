<template>
  <div class="about-page">
    <el-card shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><InfoFilled /></el-icon>
          <span>关于本系统</span>
        </div>
      </template>

      <el-timeline>
        <el-timeline-item
          v-for="(item, index) in timeline"
          :key="index"
          :timestamp="item.time"
          :type="item.type"
          :hollow="item.hollow"
        >
          <h4>{{ item.title }}</h4>
          <p>{{ item.desc }}</p>
        </el-timeline-item>
      </el-timeline>

      <el-divider />

      <h3 class="section-title">技术栈</h3>
      <el-row :gutter="16">
        <el-col :span="8" v-for="stack in techStack" :key="stack.name">
          <el-card shadow="hover" class="tech-card">
            <div class="tech-name">{{ stack.name }}</div>
            <div class="tech-desc">{{ stack.desc }}</div>
          </el-card>
        </el-col>
      </el-row>

      <el-divider />

      <h3 class="section-title">模型介绍</h3>
      <el-descriptions :column="1" border>
        <el-descriptions-item label="mT5 多语言摘要模型">
          基于 Google mT5 的多语言文本摘要模型，在 XLSum 数据集上训练，支持中文等多种语言，参数量约 5.8 亿。
        </el-descriptions-item>
        <el-descriptions-item label="Pegasus 中文摘要模型">
          基于 Pegasus 架构的中文新闻摘要模型，由 IDEA 研究院开源，参数量约 2.38 亿，专为中文摘要优化。
        </el-descriptions-item>
      </el-descriptions>

      <el-divider />

      <el-alert
        title="使用说明"
        type="success"
        :closable="false"
        show-icon
      >
        <ol style="line-height: 2">
          <li>在"摘要生成"页面输入或粘贴新闻文本</li>
          <li>选择模型并调整生成参数</li>
          <li>点击"生成摘要"按钮等待结果</li>
          <li>生成的摘要会自动保存到"历史记录"</li>
          <li>在"摘要评估"页面可计算 ROUGE 分数</li>
        </ol>
      </el-alert>
    </el-card>
  </div>
</template>

<script setup>
const timeline = [
  {
    time: '2024',
    title: '系统开发',
    desc: '基于 Transformer 架构搭建新闻摘要系统，集成 mT5 与 Pegasus 模型。',
    type: 'primary',
  },
  {
    time: '2017',
    title: 'Transformer 架构',
    desc: 'Vaswani 等人提出 Transformer，基于自注意力机制，成为 NLP 领域的里程碑。',
    type: 'success',
  },
  {
    time: '2020',
    title: 'mT5 发布',
    desc: 'Google 发布多语言 T5（mT5），支持 101 种语言。',
    type: 'success',
  },
]

const techStack = [
  { name: 'FastAPI', desc: 'Python 异步 Web 框架' },
  { name: 'PyTorch', desc: '深度学习框架' },
  { name: 'Transformers', desc: 'HuggingFace 模型库' },
  { name: 'Vue 3', desc: '渐进式前端框架' },
  { name: 'Element Plus', desc: 'Vue 3 组件库' },
  { name: 'Vite', desc: '前端构建工具' },
]
</script>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.section-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #303133;
}

.tech-card {
  text-align: center;
  margin-bottom: 16px;
}

.tech-name {
  font-size: 18px;
  font-weight: 600;
  color: #409eff;
  margin-bottom: 8px;
}

.tech-desc {
  color: #909399;
  font-size: 13px;
}
</style>
