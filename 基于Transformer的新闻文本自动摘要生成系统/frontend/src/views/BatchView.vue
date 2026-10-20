<template>
  <div class="batch-page">
    <el-row :gutter="20">
      <el-col :span="8">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><UploadFilled /></el-icon>
              <span>上传新闻文件</span>
            </div>
          </template>

          <el-upload
            ref="uploadRef"
            drag
            multiple
            accept=".txt,.csv"
            :auto-upload="false"
            :on-change="onFileChange"
            :on-remove="onFileChange"
            :file-list="fileList"
          >
            <el-icon :size="42" color="#409EFF"><UploadFilled /></el-icon>
            <div class="el-upload__text">将 txt / csv 新闻文件拖到此处，或<em>点击选择</em></div>
            <template #tip>
              <div class="el-upload__tip">支持多选 .txt 或含 text/正文 列的 .csv，单文件不超过 2MB，编码为 UTF-8 / GBK</div>
            </template>
          </el-upload>

          <el-form label-position="top" style="margin-top: 12px">
            <el-form-item label="摘要模型">
              <el-select v-model="model" placeholder="使用默认模型" style="width: 100%" clearable>
                <el-option
                  v-for="m in modelList"
                  :key="m.name"
                  :label="m.display_name"
                  :value="m.name"
                />
              </el-select>
            </el-form-item>
            <el-form-item label="摘要长度">
              <el-slider v-model="max_length" :min="64" :max="256" :step="8" show-input />
            </el-form-item>
          </el-form>

          <el-button
            type="primary"
            style="width: 100%"
            :loading="uploading"
            :disabled="!fileList.length"
            @click="startBatch"
          >
            <el-icon><VideoPlay /></el-icon>
            <span>开始批量生成摘要</span>
          </el-button>
        </el-card>
      </el-col>

      <el-col :span="16">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><DataAnalysis /></el-icon>
              <span>批量处理结果</span>
              <el-button
                v-if="done && results.length"
                size="small"
                type="success"
                plain
                style="margin-left: auto"
                @click="exportFile('xlsx')"
              >
                <el-icon><Download /></el-icon>
                导出 Excel
              </el-button>
              <el-button
                v-if="done && results.length"
                size="small"
                plain
                @click="exportFile('csv')"
              >
                导出 CSV
              </el-button>
            </div>
          </template>

          <div v-if="processing || done" class="progress-row">
            <el-progress
              :percentage="progressPercent"
              :status="done ? 'success' : undefined"
            />
            <span class="progress-text">
              {{ done ? `处理完成，共 ${total} 篇` : `正在处理 ${progress}/${total} 篇...` }}
            </span>
          </div>

          <el-table v-if="results.length" :data="results" stripe size="small" height="520">
            <el-table-column type="expand">
              <template #default="{ row }">
                <div class="expand-summary">
                  <div><b>润色后摘要：</b>{{ row.polished_summary || '—' }}</div>
                  <div style="margin-top: 6px; color: #909399">
                    <b>模型初稿：</b>{{ row.summary || '—' }}
                  </div>
                </div>
              </template>
            </el-table-column>
            <el-table-column prop="filename" label="文件名" min-width="140" show-overflow-tooltip />
            <el-table-column prop="original_chars" label="原文字数" width="90" align="center" />
            <el-table-column prop="summary_chars" label="摘要字数" width="90" align="center" />
            <el-table-column label="压缩比" width="80" align="center">
              <template #default="{ row }">{{ row.compression_ratio }}%</template>
            </el-table-column>
            <el-table-column label="分类" width="120" align="center">
              <template #default="{ row }">
                <el-tag v-if="row.category" size="small">{{ row.category }}</el-tag>
                <span v-if="row.category_confidence" class="conf">{{ row.category_confidence }}%</span>
              </template>
            </el-table-column>
            <el-table-column label="可信度" width="90" align="center">
              <template #default="{ row }">
                <el-tag
                  v-if="row.credibility"
                  size="small"
                  :type="row.credibility >= 80 ? 'success' : row.credibility >= 60 ? 'warning' : 'danger'"
                >{{ row.credibility }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="model_display_name" label="模型" min-width="140" show-overflow-tooltip />
            <el-table-column label="状态" width="90" align="center">
              <template #default="{ row }">
                <el-tag size="small" :type="row.status === '成功' ? 'success' : 'danger'">
                  {{ row.status === '成功' ? '成功' : '失败' }}
                </el-tag>
              </template>
            </el-table-column>
          </el-table>

          <el-empty
            v-else
            description="上传文件并点击“开始批量生成摘要”，结果将在此展示"
          />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, computed, onBeforeUnmount, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { batchUpload, batchStatus, batchExport, getModels } from '../api'

const uploadRef = ref(null)
const fileList = ref([])
const modelList = ref([])
const model = ref('')
const max_length = ref(128)

const uploading = ref(false)
const processing = ref(false)
const done = ref(false)
const batchId = ref('')
const progress = ref(0)
const total = ref(0)
const results = ref([])
let timer = null

const progressPercent = computed(() =>
  total.value ? Math.round((progress.value / total.value) * 100) : 0
)

onMounted(async () => {
  try {
    const res = await getModels()
    modelList.value = res.models
  } catch (e) {
    console.error('获取模型列表失败', e)
  }
})

onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
})

function onFileChange(file, list) {
  fileList.value = list.filter((f) => /\.(txt|csv)$/i.test(f.name))
}

async function startBatch() {
  if (!fileList.value.length) {
    ElMessage.warning('请先选择 txt 或 csv 文件')
    return
  }
  uploading.value = true
  done.value = false
  results.value = []
  try {
    const fd = new FormData()
    for (const f of fileList.value) {
      fd.append('files', f.raw, f.name)
    }
    if (model.value) fd.append('model', model.value)
    fd.append('max_length', max_length.value)
    fd.append('min_length', 32)

    const res = await batchUpload(fd)
    batchId.value = res.batch_id
    total.value = res.total
    progress.value = 0
    processing.value = true
    ElMessage.success(`已提交 ${res.total} 个文件，开始处理`)
    timer = setInterval(poll, 2000)
  } catch (e) {
    // 拦截器已提示
  } finally {
    uploading.value = false
  }
}

async function poll() {
  try {
    const res = await batchStatus(batchId.value)
    progress.value = res.progress
    if (res.status === 'done') {
      clearInterval(timer)
      timer = null
      processing.value = false
      done.value = true
      results.value = res.results
      ElMessage.success('批量处理完成')
    }
  } catch (e) {
    clearInterval(timer)
    timer = null
    processing.value = false
  }
}

async function exportFile(format) {
  try {
    const resp = await batchExport(batchId.value, format)
    const mime = format === 'csv'
      ? 'text/csv;charset=utf-8'
      : 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    const blob = new Blob([resp.data], { type: mime })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `批量摘要结果_${batchId.value}.${format}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
    ElMessage.success(`已导出 ${format.toUpperCase()}`)
  } catch (e) {
    ElMessage.error('导出失败')
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

.progress-row {
  margin-bottom: 12px;
}

.progress-text {
  font-size: 13px;
  color: #606266;
}

.expand-summary {
  padding: 8px 16px;
  font-size: 13px;
  line-height: 1.8;
  color: #303133;
}

.conf {
  margin-left: 4px;
  font-size: 12px;
  color: #909399;
}
</style>
