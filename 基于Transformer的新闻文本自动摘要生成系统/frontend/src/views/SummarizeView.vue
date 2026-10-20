<template>
  <div class="summarize-page">
    <el-row :gutter="20">
      <!-- 左侧：输入与参数 -->
      <el-col :span="14">
        <el-card shadow="hover" class="input-card">
          <template #header>
            <div class="card-header">
              <el-icon><EditPen /></el-icon>
              <span>新闻文本输入</span>
              <el-tag v-if="result" size="small" type="success" style="margin-left: auto">
                {{ result.classification?.category || '未分类' }}
              </el-tag>
            </div>
          </template>

          <el-input
            v-model="form.text"
            type="textarea"
            :rows="12"
            placeholder="请在此粘贴或输入需要摘要的中文新闻文本..."
            maxlength="5000"
            show-word-limit
          />

          <div class="example-buttons">
            <span class="text-muted">示例新闻：</span>
            <el-button
              v-for="(ex, idx) in examples.slice(0, 6)"
              :key="idx"
              size="small"
              :title="ex.title"
              @click="useExample(idx)"
            >
              {{ ex.title }}
            </el-button>
            <el-button size="small" @click="clearText" type="danger" plain>
              清空
            </el-button>
          </div>
        </el-card>

        <el-card shadow="hover" class="params-card" style="margin-top: 16px">
          <template #header>
            <div class="card-header">
              <el-icon><Setting /></el-icon>
              <span>生成参数与增强选项</span>
            </div>
          </template>

          <el-form label-position="top">
            <el-form-item label="模型选择">
              <el-select v-model="form.model" placeholder="选择摘要模型" style="width: 100%">
                <el-option
                  v-for="m in modelList"
                  :key="m.name"
                  :label="m.display_name"
                  :value="m.name"
                >
                  <div>{{ m.display_name }}</div>
                  <div style="font-size: 12px; color: #909399">{{ m.description }}</div>
                </el-option>
              </el-select>
            </el-form-item>

            <el-row :gutter="16">
              <el-col :span="12">
                <el-form-item :label="`最大长度 (${form.max_length})`">
                  <el-slider v-model="form.max_length" :min="16" :max="256" :step="8" show-input />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="`最小长度 (${form.min_length})`">
                  <el-slider v-model="form.min_length" :min="1" :max="128" :step="1" show-input />
                </el-form-item>
              </el-col>
            </el-row>

            <el-row :gutter="16">
              <el-col :span="12">
                <el-form-item :label="`Beam 数量 (${form.num_beams})`">
                  <el-slider v-model="form.num_beams" :min="1" :max="10" :step="1" show-input />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item :label="`长度惩罚 (${form.length_penalty})`">
                  <el-slider v-model="form.length_penalty" :min="0.1" :max="3" :step="0.1" show-input />
                </el-form-item>
              </el-col>
            </el-row>

            <el-divider content-position="left">增强功能</el-divider>

            <el-row :gutter="16">
              <el-col :span="8">
                <el-form-item label="摘要润色">
                  <el-switch v-model="form.polish" />
                  <span class="opt-tip">正式书面风格</span>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="英文摘要">
                  <el-switch v-model="form.enable_english" />
                  <span class="opt-tip">中译英</span>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="注意力热力图">
                  <el-switch v-model="form.enable_attention" />
                  <span class="opt-tip">可视化</span>
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="16">
              <el-col :span="8">
                <el-form-item label="多样化生成">
                  <el-switch v-model="form.do_sample" />
                  <span class="opt-tip">每次结果不同</span>
                </el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="英译中">
                  <el-switch v-model="form.enable_zh_translate" />
                  <span class="opt-tip">英文新闻自动翻译</span>
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item label="多粒度摘要（额外生成，速度较慢）">
              <el-checkbox-group v-model="form.granularities">
                <el-checkbox label="short">短摘要（约50字）</el-checkbox>
                <el-checkbox label="medium">中等摘要（约150字）</el-checkbox>
                <el-checkbox label="detailed">详细摘要（约300字）</el-checkbox>
              </el-checkbox-group>
            </el-form-item>
          </el-form>
        </el-card>

        <el-button
          type="primary"
          size="large"
          class="generate-btn"
          :loading="loading"
          :disabled="!form.text.trim()"
          @click="handleSummarize"
        >
          <el-icon><MagicStick /></el-icon>
          <span>生成摘要</span>
        </el-button>
      </el-col>

      <!-- 右侧：结果展示 -->
      <el-col :span="10">
        <el-card shadow="hover" class="result-card">
          <template #header>
            <div class="card-header">
              <el-icon><Document /></el-icon>
              <span>摘要结果与分析</span>
              <el-button
                v-if="result"
                size="small"
                type="primary"
                plain
                style="margin-left: auto"
                @click="copyResult"
              >
                复制摘要
              </el-button>
              <el-button
                v-if="result"
                size="small"
                type="success"
                plain
                :loading="exporting"
                @click="exportPdf"
              >
                导出PDF报告
              </el-button>
            </div>
          </template>

          <div v-if="loading" class="loading-box">
            <el-icon class="is-loading" :size="40" color="#409EFF"><Loading /></el-icon>
            <p>模型正在生成摘要，请稍候...</p>
            <p class="text-muted" style="margin-top: 8px">
              多粒度 / 英文 / 热力图等选项会显著增加耗时
            </p>
          </div>

          <div v-else-if="result" class="result-content">
            <el-tabs v-model="activeTab" type="border-card">
              <!-- Tab 1: 摘要结果 -->
              <el-tab-pane label="摘要结果" name="summary">
                <el-alert
                  v-if="result.original_lang === 'en'"
                  title="检测到英文新闻，已自动翻译为中文后生成摘要"
                  type="info"
                  :closable="false"
                  style="margin-bottom: 10px"
                />
                <!-- 字数统计 -->
                <div class="stats-row">
                  <div class="stat-item">
                    <div class="stat-num">{{ result.stats.original_chars }}</div>
                    <div class="stat-label">原文（字）</div>
                  </div>
                  <el-icon class="stat-arrow"><Right /></el-icon>
                  <div class="stat-item">
                    <div class="stat-num">{{ result.stats.summary_chars }}</div>
                    <div class="stat-label">摘要（字）</div>
                  </div>
                  <div class="stat-item">
                    <div class="stat-num" style="color: #e6a23c">{{ result.stats.compression_ratio }}%</div>
                    <div class="stat-label">压缩比例</div>
                  </div>
                  <div class="stat-item">
                    <div class="stat-num" style="color: #67c23a">{{ result.stats.compression_percent }}%</div>
                    <div class="stat-label">压缩掉的内容</div>
                  </div>
                  <div class="stat-item">
                    <div class="stat-num" style="color: #409eff">{{ result.readability?.score ?? '--' }}</div>
                    <div class="stat-label">易读性得分</div>
                  </div>
                </div>

                <!-- 可信度评估 -->
                <div class="credibility-box" :class="credClass">
                  <div class="cred-head">
                    <el-icon><CircleCheck /></el-icon>
                    <span>摘要可信度评估（幻觉检测）</span>
                    <el-tag :type="credTagType" size="small" style="margin-left: auto">
                      {{ result.credibility.level }}可信 · {{ result.credibility.score }} 分
                    </el-tag>
                  </div>
                  <el-progress
                    :percentage="result.credibility.score"
                    :color="credColor"
                    :stroke-width="10"
                  />
                  <div class="cred-details">
                    <el-tooltip content="摘要实词在原文中的出现比例，越高说明摘要越忠实于原文" placement="top">
                      <span class="cred-metric">内容词覆盖 {{ result.credibility.details.content_coverage }}%</span>
                    </el-tooltip>
                    <el-tooltip content="摘要内所有数字均可在原文找到对应" placement="top">
                      <span class="cred-metric">数字一致性 {{ result.credibility.details.number_consistency }}%</span>
                    </el-tooltip>
                    <el-tooltip content="人名/地名/机构实体没有编造" placement="top">
                      <span class="cred-metric">实体一致性 {{ result.credibility.details.entity_consistency }}%</span>
                    </el-tooltip>
                  </div>
                  <div v-if="result.credibility.score < 70" class="cred-warning">
                    <el-icon><Warning /></el-icon>
                    <span>可信度较低，请人工复核摘要内容</span>
                  </div>
                  <el-alert
                    v-for="(w, i) in result.credibility.warnings"
                    :key="i"
                    :title="w"
                    type="warning"
                    :closable="false"
                    style="margin-top: 8px"
                  />
                  <div v-if="result.credibility.missing_fragments.length" class="cred-missing">
                    疑似原文未出现：
                    <el-tag
                      v-for="(m, i) in result.credibility.missing_fragments"
                      :key="i"
                      size="small"
                      type="danger"
                      effect="plain"
                      style="margin: 2px"
                    >{{ m }}</el-tag>
                  </div>
                  <div v-if="result.credibility.hallucination_types?.length" class="hallu-types">
                    幻觉类型识别：
                    <el-tooltip
                      v-for="(h, i) in result.credibility.hallucination_types"
                      :key="i"
                      :content="h.description"
                      placement="top"
                    >
                      <el-tag
                        size="small"
                        :type="h.level === '高' ? 'danger' : h.level === '中' ? 'warning' : h.level === '无' ? 'success' : 'info'"
                        style="margin: 2px"
                      >{{ h.type }}</el-tag>
                    </el-tooltip>
                  </div>
                </div>

                <!-- 主题分类与关键词 -->
                <div class="class-box">
                  <div class="box-title">
                    <el-icon><Collection /></el-icon>
                    <span>新闻主题分类</span>
                  </div>
                  <div class="class-tags">
                    <el-tag type="primary" effect="dark" size="large">
                      {{ result.classification.category }}
                    </el-tag>
                    <el-tag type="info" style="margin-left: 8px">
                      置信度 {{ result.classification.confidence }}%
                    </el-tag>
                    <el-tag
                      v-if="result.classification.confidence < 70"
                      type="warning"
                      effect="plain"
                      style="margin-left: 8px"
                    >
                      <el-icon><Warning /></el-icon> 分类结果不确定性较高，请人工复核
                    </el-tag>
                    <el-tag
                      v-for="(s, cat) in result.classification.scores"
                      :key="cat"
                      type="info"
                      effect="plain"
                      style="margin-left: 6px"
                    >{{ cat }} {{ s }}</el-tag>
                  </div>
                  <div class="keyword-tags">
                    <el-tag
                      v-for="k in result.keywords"
                      :key="k.word"
                      size="small"
                      effect="plain"
                      type="warning"
                    >{{ k.word }}</el-tag>
                  </div>
                </div>

                <!-- 润色后摘要（可编辑） -->
                <div class="summary-box">
                  <div class="box-title">
                    <el-icon><MagicStick /></el-icon>
                    <span>摘要（润色后 · 可编辑）</span>
                    <el-button size="small" text type="primary" style="margin-left: auto" @click="toggleEdit">
                      {{ editing ? '取消编辑' : '编辑修正' }}
                    </el-button>
                  </div>

                  <div v-if="!editing" class="summary-text">
                    <template v-for="(token, i) in summaryTokens" :key="i">
                      <span
                        :class="token.inOriginal ? 'token-match' : 'token-hallucination'"
                        :title="token.inOriginal ? '与原文匹配' : '疑似幻觉：原文未出现'"
                      >{{ token.text }}</span>
                    </template>
                  </div>
                  <div v-else>
                    <el-input
                      v-model="editedSummary"
                      type="textarea"
                      :rows="5"
                      maxlength="2000"
                      show-word-limit
                    />
                    <div class="edit-actions">
                      <el-button size="small" type="primary" :loading="savingFeedback" @click="saveEdited">
                        <el-icon><Check /></el-icon> 保存修正版本
                      </el-button>
                      <span class="opt-tip">修正记录将保存为微调补充数据</span>
                    </div>
                  </div>

                  <div v-if="result.polish_changes.length" class="polish-changes">
                    <el-tag
                      v-for="(c, i) in result.polish_changes"
                      :key="i"
                      size="small"
                      type="success"
                      effect="plain"
                    >{{ c }}</el-tag>
                  </div>
                </div>

                <!-- 模型初稿 -->
                <el-collapse>
                  <el-collapse-item title="查看模型初稿（润色前）">
                    <div class="draft-text">{{ result.summary }}</div>
                  </el-collapse-item>
                </el-collapse>

                <!-- 多粒度摘要 -->
                <div v-if="Object.keys(result.granularities).length" class="granularity-box">
                  <div class="box-title">
                    <el-icon><Histogram /></el-icon>
                    <span>多粒度摘要</span>
                  </div>
                  <el-tabs type="card">
                    <el-tab-pane v-if="result.granularities.short" label="短摘要（50字）">
                      <div class="gran-text">{{ result.granularities.short }}</div>
                    </el-tab-pane>
                    <el-tab-pane v-if="result.granularities.medium" label="中等摘要（150字）">
                      <div class="gran-text">{{ result.granularities.medium }}</div>
                    </el-tab-pane>
                    <el-tab-pane v-if="result.granularities.detailed" label="详细摘要（300字）">
                      <div class="gran-text">{{ result.granularities.detailed }}</div>
                    </el-tab-pane>
                  </el-tabs>
                </div>

                <!-- 英文摘要 -->
                <div v-if="result.english_summary" class="english-box">
                  <div class="box-title">
                    <el-icon><Promotion /></el-icon>
                    <span>English Summary</span>
                  </div>
                  <div class="english-text">{{ result.english_summary }}</div>
                </div>

                <el-descriptions :column="1" border size="small" class="meta-info">
                  <el-descriptions-item label="使用模型">{{ result.model_display_name }}</el-descriptions-item>
                  <el-descriptions-item label="生成参数">
                    max_length={{ result.params.max_length }}, num_beams={{ result.params.num_beams }},
                    penalty={{ result.params.length_penalty }}
                  </el-descriptions-item>
                </el-descriptions>
              </el-tab-pane>

              <!-- Tab 2: 对比实验 -->
                <el-tab-pane label="对比实验" name="compare">
                  <div class="compare-tip">
                    支持同时选用2种模型，对同一篇新闻生成两份摘要并排对比，直接对比幻觉分数、压缩率，适合论文实验。
                  </div>
                  <div class="compare-tip">
                    抽取式（TF-IDF 选句）vs 生成式（Transformer），可作论文对比实验
                  </div>
                <el-row :gutter="12">
                  <el-col :span="12">
                    <div class="compare-card extractive">
                      <div class="compare-head">
                        抽取式摘要
                        <el-tag size="small" type="info">原文句子拼接</el-tag>
                      </div>
                      <div class="compare-text">{{ result.extractive_summary }}</div>
                    </div>
                  </el-col>
                  <el-col :span="12">
                    <div class="compare-card generative">
                      <div class="compare-head">
                        生成式摘要
                        <el-tag size="small" type="primary">Transformer</el-tag>
                      </div>
                      <div class="compare-text">{{ result.polished_summary }}</div>
                    </div>
                  </el-col>
                </el-row>
                <el-divider content-position="left">抽取式选中句子</el-divider>
                <div class="extractive-detail">
                  <div v-for="s in result.extractive_detail" :key="s.index" class="extractive-item">
                    <el-tag size="small" type="info">句 {{ s.index + 1 }} · 得分 {{ s.score }}</el-tag>
                    <span style="margin-left: 8px">{{ s.sentence }}</span>
                  </div>
                </div>
              </el-tab-pane>

              <!-- Tab 3: 高亮定位 -->
              <el-tab-pane label="高亮定位" name="highlight">
                <div class="compare-tip">
                  黄色高亮为生成摘要主要参考的原文关键句（覆盖率 {{ result.highlight_coverage }}%），
                  悬停可查看匹配度
                </div>
                <div class="highlight-text">
                  <template v-for="(s, i) in result.highlights" :key="i">
                    <span
                      :class="{ 'key-sentence': s.is_key }"
                      :title="s.is_key ? `匹配度 ${(s.score * 100).toFixed(1)}%` : ''"
                    >{{ s.sentence }}</span>
                  </template>
                </div>
              </el-tab-pane>

              <!-- Tab 4: 注意力热力图 -->
                <el-tab-pane label="注意力热力图" name="attention">
                  <div v-if="result.attention.length" class="attention-box">
                    <div class="compare-tip">
                      可视化Transformer模型注意力权重，查看模型重点关注哪些原文词语
                    </div>
                    <div class="compare-tip">
                      颜色越深表示模型生成摘要时对该词的关注度越高（交叉注意力，末层均值）
                    </div>
                    <div class="heatmap">
                      <span
                        v-for="(t, i) in result.attention"
                        :key="i"
                        class="heat-token"
                        :style="{ backgroundColor: heatColor(t.weight) }"
                        :title="`注意力权重: ${(t.weight * 100).toFixed(1)}%`"
                      >{{ t.token }}</span>
                    </div>
                  <div class="heat-legend">
                    <span>低</span>
                    <span
                      v-for="i in 8"
                      :key="i"
                      class="legend-cell"
                      :style="{ backgroundColor: heatColor(i / 8) }"
                    ></span>
                    <span>高</span>
                  </div>
                  <div class="heat-top">
                    模型最关注的词：
                    <el-tag
                      v-for="t in topAttentionTokens"
                      :key="t.token"
                      size="small"
                      type="danger"
                      effect="plain"
                      style="margin: 2px"
                    >{{ t.token }} ({{ (t.weight * 100).toFixed(1) }}%)</el-tag>
                  </div>
                </div>
                <el-empty v-else description="本次未启用注意力热力图，请在左侧打开开关后重新生成" />
              </el-tab-pane>

              <!-- Tab 5: 相关推荐 -->
              <el-tab-pane label="相关推荐" name="recommend">
                <div v-if="result.recommendations.length">
                  <div class="compare-tip">基于关键词检索历史新闻库的主题相近新闻</div>
                  <div
                    v-for="r in result.recommendations"
                    :key="r.id"
                    class="rec-item"
                  >
                    <div class="rec-head">
                      <el-tag size="small" type="success">{{ r.category || '未分类' }}</el-tag>
                      <span class="rec-title">{{ r.title }}</span>
                      <el-tag size="small" type="warning" style="margin-left: auto">
                        相似度 {{ (r.similarity * 100).toFixed(0) }}%
                      </el-tag>
                    </div>
                    <div class="rec-snippet">{{ r.snippet }}</div>
                    <div class="rec-keywords">
                      <el-tag v-for="k in r.matched_keywords" :key="k" size="small" effect="plain">
                        {{ k }}
                      </el-tag>
                    </div>
                  </div>
                </div>
                <el-empty v-else description="暂无相似新闻（历史库会随使用自动积累）" />
              </el-tab-pane>
            </el-tabs>
          </div>

          <div v-else class="empty-box">
            <el-empty description="输入文本后点击生成摘要"></el-empty>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Warning } from '@element-plus/icons-vue'
import { summarizeFull, getModels, saveFeedback, getDatasetSamples, reAnalyzeSummary, exportReport } from '../api'
import { useHistoryStore } from '../stores/history'

const historyStore = useHistoryStore()

const loading = ref(false)
const result = ref(null)
const modelList = ref([])
const activeTab = ref('summary')
const editing = ref(false)
const editedSummary = ref('')
const savingFeedback = ref(false)

const form = reactive({
  text: '',
  model: '',
  max_length: 128,
  min_length: 32,
  num_beams: 4,
  length_penalty: 1.0,
  polish: true,
  enable_english: false,
  enable_attention: false,
  do_sample: false,
  enable_zh_translate: true,
  granularities: [],
})

const examples = ref([])

onMounted(async () => {
  try {
    const res = await getModels()
    modelList.value = res.models
    form.model = res.default_model
  } catch (e) {
    console.error('获取模型列表失败', e)
  }
  try {
    const res = await getDatasetSamples()
    // 各取 3 条中文、3 条英文示例（英文样本按 ASCII 字母占比判断）
    const isEn = (s) => {
      const t = (s.text || '').slice(0, 200)
      const alpha = [...t].filter(c => c.charCodeAt(0) < 128 && /[a-zA-Z]/.test(c)).length
      return t.length > 0 && alpha / t.replace(/\s/g, '').length >= 0.6
    }
    const zh = res.filter(s => !isEn(s)).slice(0, 3)
    const en = res.filter(s => isEn(s)).slice(0, 3)
    examples.value = [...zh, ...en]
  } catch (e) {
    console.error('获取示例数据集失败', e)
  }
  // 接收新闻数据查询库传来的新闻
  const pending = sessionStorage.getItem('pendingNewsText')
  if (pending) {
    form.text = pending
    sessionStorage.removeItem('pendingNewsText')
    sessionStorage.removeItem('pendingNewsTitle')
    ElMessage.info('已从数据查询库载入新闻文本')
  }
})

const credColor = computed(() => {
  if (!result.value) return '#409eff'
  const s = result.value.credibility.score
  return s >= 80 ? '#67c23a' : s >= 60 ? '#e6a23c' : '#f56c6c'
})
const credTagType = computed(() => {
  if (!result.value) return 'info'
  const s = result.value.credibility.score
  return s >= 80 ? 'success' : s >= 60 ? 'warning' : 'danger'
})
const credClass = computed(() => {
  if (!result.value) return ''
  const s = result.value.credibility.score
  return s >= 80 ? 'cred-good' : s >= 60 ? 'cred-mid' : 'cred-low'
})

const topAttentionTokens = computed(() => {
  if (!result.value || !result.value.attention.length) return []
  return [...result.value.attention]
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 6)
})

// 幻觉高亮：将摘要按 jieba 分词，标记每个词是否在原文中出现
const summaryTokens = computed(() => {
  if (!result.value) return []
  const original = form.text || ''
  const summary = result.value.polished_summary || ''
  if (!summary) return []

  // 简单按字符切分（前端不做 jieba 分词，用滑动窗口匹配）
  const tokens = []
  let i = 0
  while (i < summary.length) {
    let matched = false
    // 尝试匹配最长词（2-6字）
    for (let len = Math.min(6, summary.length - i); len >= 2; len--) {
      const word = summary.slice(i, i + len)
      if (original.includes(word)) {
        tokens.push({ text: word, inOriginal: true })
        i += len
        matched = true
        break
      }
    }
    if (!matched) {
      // 单字或标点
      const char = summary[i]
      const isPunct = /[，。！？；：、\s\d%.""''（）()a-zA-Z]/.test(char)
      tokens.push({ text: char, inOriginal: isPunct ? true : original.includes(char) })
      i++
    }
  }
  return tokens
})

function heatColor(w) {
  // 从浅蓝到深红的注意力色阶
  const alpha = Math.min(Math.max(w, 0), 1)
  return `rgba(230, 80, 30, ${0.06 + alpha * 0.85})`
}

function useExample(idx) {
  const sample = examples.value[idx]
  if (sample) {
    form.text = sample.text
    ElMessage.info(`已加载示例：${sample.title}`)
  }
}

async function clearText() {
  try {
    await ElMessageBox.confirm('确定清空当前输入的文本吗？', '清空确认', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
    })
    form.text = ''
    result.value = null
    ElMessage.success('已清空')
  } catch {
    // 用户取消
  }
}

function toggleEdit() {
  editing.value = !editing.value
  if (editing.value) {
    editedSummary.value = result.value?.polished_summary || ''
  }
}

async function saveEdited() {
  if (!editedSummary.value.trim()) {
    ElMessage.warning('修改后的摘要不能为空')
    return
  }
  savingFeedback.value = true
  try {
    const res = await saveFeedback({
      original_text: form.text,
      model_summary: result.value.summary,
      edited_summary: editedSummary.value.trim(),
      model: result.value.model,
      params: result.value.params,
    })
    const updatedSummary = editedSummary.value.trim()
    result.value.polished_summary = updatedSummary

    // 重新计算可信度、压缩比等指标
    try {
      const re = await reAnalyzeSummary({
        original_text: form.text,
        summary: updatedSummary,
      })
      result.value.stats = re.stats
      result.value.credibility = re.credibility
      result.value.highlights = re.highlights
      result.value.highlight_coverage = re.highlight_coverage
      result.value.readability = re.readability
      ElMessage.success('编辑已保存，评估指标已重新计算')
    } catch (e) {
      console.error('重新分析失败', e)
      ElMessage.warning('编辑已保存，但重新计算指标失败')
    }

    historyStore.addRecord({
      originalText: form.text,
      summary: updatedSummary,
      model: result.value.model_display_name + '（人工修正）',
      params: result.value.params,
    })
    editing.value = false
    ElMessage.success(`${res.message}，当前累计 ${res.total_records} 条`)
  } catch (e) {
    // 拦截器已提示
  } finally {
    savingFeedback.value = false
  }
}

async function handleSummarize() {
  if (!form.text.trim()) {
    ElMessage.warning('请输入新闻文本')
    return
  }
  if (form.min_length >= form.max_length) {
    ElMessage.warning('最小长度必须小于最大长度')
    return
  }
  loading.value = true
  result.value = null
  activeTab.value = 'summary'
  try {
    const res = await summarizeFull({
      text: form.text,
      model: form.model,
      max_length: form.max_length,
      min_length: form.min_length,
      num_beams: form.num_beams,
      length_penalty: form.length_penalty,
      polish: form.polish,
      granularities: form.granularities,
      enable_english: form.enable_english,
      enable_attention: form.enable_attention,
      enable_recommend: true,
      do_sample: form.do_sample,
      enable_zh_translate: form.enable_zh_translate,
    })
    result.value = res
    historyStore.addRecord({
      originalText: form.text,
      summary: res.polished_summary,
      model: res.model_display_name,
      params: res.params,
      category: res.classification?.category,
    })
    ElMessage.success('摘要生成成功')
  } catch (e) {
    ElMessage.error('摘要生成失败，请重试')
  } finally {
    loading.value = false
  }
}

function copyResult() {
  if (result.value) {
    navigator.clipboard.writeText(result.value.polished_summary)
    ElMessage.success('已复制到剪贴板')
  }
}

const exporting = ref(false)

async function exportPdf() {
  if (!result.value) return
  exporting.value = true
  try {
    const res = await exportReport({
      original_text: form.text,
      summary: result.value.polished_summary,
      model_display_name: result.value.model_display_name,
      category: result.value.classification?.category || '',
      category_confidence: result.value.classification?.confidence || 0,
      keywords: (result.value.keywords || []).map(k => k.word || k),
    })
    const url = window.URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `摘要评估报告_${new Date().toISOString().slice(0, 10)}.pdf`
    a.click()
    window.URL.revokeObjectURL(url)
    ElMessage.success('PDF 报告已导出')
  } catch (e) {
    console.error(e)
    ElMessage.error('导出失败')
  } finally {
    exporting.value = false
  }
}
</script>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.example-buttons {
  margin-top: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.opt-tip {
  margin-left: 8px;
  font-size: 12px;
  color: #909399;
}

.result-card {
  display: flex;
  flex-direction: column;
}

.loading-box {
  text-align: center;
  padding: 60px 0;
  color: #606266;
}

.loading-box .el-icon {
  margin-bottom: 12px;
}

.result-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.empty-box {
  padding: 60px 0;
}

.generate-btn {
  width: 100%;
  margin-top: 16px;
}

/* 字数统计 */
.stats-row {
  display: flex;
  align-items: center;
  justify-content: space-around;
  background: #f5f7fa;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 12px;
}

.stat-item {
  text-align: center;
}

.stat-num {
  font-size: 22px;
  font-weight: 700;
  color: #409eff;
}

.stat-label {
  font-size: 12px;
  color: #909399;
  margin-top: 2px;
}

.stat-arrow {
  color: #c0c4cc;
  font-size: 20px;
}

/* 可信度 */
.credibility-box {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 12px;
}

.cred-good { background: #f0f9eb; }
.cred-mid { background: #fdf6ec; }
.cred-low { background: #fef0f0; }

.cred-head {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  margin-bottom: 8px;
}

.cred-details {
  font-size: 12px;
  color: #606266;
  margin-top: 6px;
}

.cred-missing {
  margin-top: 8px;
  font-size: 12px;
  color: #606266;
}

.hallu-types {
  margin-top: 8px;
  font-size: 12px;
  color: #606266;
}

/* 分类与关键词 */
.class-box {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 12px;
}

.box-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  margin-bottom: 8px;
}

.class-tags {
  margin-bottom: 8px;
}

.keyword-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

/* 摘要 */
.summary-box {
  border: 1px solid #d9ecff;
  background: #f0f9ff;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 12px;
}

.summary-text {
  font-size: 15px;
  line-height: 1.8;
  color: #303133;
}

.token-match {
  background-color: #f0f9eb;
  color: #67c23a;
  padding: 1px 0;
}

.token-hallucination {
  background-color: #fef0f0;
  color: #f56c6c;
  padding: 1px 0;
  border-bottom: 1px dashed #f56c6c;
}

.edit-actions {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.polish-changes {
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.draft-text {
  font-size: 14px;
  line-height: 1.8;
  color: #606266;
  padding: 8px;
  background: #fafafa;
  border-radius: 4px;
}

/* 多粒度 */
.granularity-box {
  margin-bottom: 12px;
}

.gran-text {
  font-size: 14px;
  line-height: 1.8;
  color: #303133;
  padding: 8px;
  background: #f5f7fa;
  border-radius: 4px;
}

/* 英文 */
.english-box {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 12px;
}

.english-text {
  font-size: 14px;
  line-height: 1.8;
  color: #303133;
  font-style: italic;
}

/* 对比 */
.compare-tip {
  font-size: 12px;
  color: #909399;
  margin-bottom: 10px;
}

.compare-card {
  border-radius: 8px;
  padding: 12px;
  min-height: 120px;
}

.compare-card.extractive {
  background: #f4f4f5;
  border-left: 4px solid #909399;
}

.compare-card.generative {
  background: #f0f9ff;
  border-left: 4px solid #409eff;
}

.compare-head {
  font-weight: 600;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.compare-text {
  font-size: 14px;
  line-height: 1.8;
  color: #303133;
}

.extractive-item {
  margin-bottom: 8px;
  font-size: 13px;
  line-height: 1.7;
  color: #606266;
}

/* 高亮 */
.highlight-text {
  font-size: 14px;
  line-height: 2.1;
  color: #303133;
  background: #fafafa;
  padding: 12px;
  border-radius: 6px;
}

.key-sentence {
  background: #ffe28a;
  border-bottom: 2px solid #e6a23c;
  padding: 1px 0;
}

/* 热力图 */
.heatmap {
  line-height: 2.4;
  font-size: 14px;
}

.heat-token {
  padding: 2px 4px;
  margin: 2px;
  border-radius: 3px;
  cursor: default;
}

.heat-legend {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 10px;
  font-size: 12px;
  color: #909399;
}

.legend-cell {
  width: 22px;
  height: 12px;
  border-radius: 2px;
  display: inline-block;
}

.heat-top {
  margin-top: 10px;
  font-size: 13px;
  color: #606266;
}

/* 推荐 */
.rec-item {
  border: 1px solid #ebeef5;
  border-radius: 8px;
  padding: 10px 12px;
  margin-bottom: 10px;
}

.rec-head {
  display: flex;
  align-items: center;
  gap: 8px;
}

.rec-title {
  font-weight: 600;
  font-size: 14px;
}

.rec-snippet {
  font-size: 13px;
  color: #606266;
  line-height: 1.7;
  margin-top: 6px;
}

.rec-keywords {
  margin-top: 6px;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.meta-info {
  margin-top: 8px;
}
</style>
