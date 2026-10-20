<template>
  <div class="evaluate-page">
    <el-row :gutter="20">
      <el-col :span="12">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><EditPen /></el-icon>
              <span>生成摘要</span>
            </div>
          </template>
          <el-input
            v-model="form.hypothesis"
            type="textarea"
            :rows="10"
            placeholder="请输入模型生成的摘要..."
          />
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card shadow="hover">
          <template #header>
            <div class="card-header">
              <el-icon><Collection /></el-icon>
              <span>参考摘要</span>
            </div>
          </template>
          <el-input
            v-model="form.reference"
            type="textarea"
            :rows="10"
            placeholder="请输入参考摘要（标准答案）..."
          />
        </el-card>
      </el-col>
    </el-row>

    <div class="action-bar">
      <el-button
        type="primary"
        :loading="loading"
        :disabled="!form.hypothesis || !form.reference"
        @click="handleEvaluate"
      >
        <el-icon><DataAnalysis /></el-icon>
        <span>开始评估</span>
      </el-button>
    </div>

    <el-card v-if="result" shadow="hover" class="result-card">
      <template #header>
        <div class="card-header">
          <el-icon><DataAnalysis /></el-icon>
          <span>ROUGE 评估结果</span>
        </div>
      </template>

      <el-table :data="tableData" stripe style="width: 100%">
        <el-table-column prop="metric" label="指标" width="140" />
        <el-table-column prop="precision" label="精确率 (Precision)">
          <template #default="{ row }">
            <el-progress :percentage="row.precision * 100" :status="statusColor(row.precision)" />
          </template>
        </el-table-column>
        <el-table-column prop="recall" label="召回率 (Recall)">
          <template #default="{ row }">
            <el-progress :percentage="row.recall * 100" :status="statusColor(row.recall)" />
          </template>
        </el-table-column>
        <el-table-column prop="f1" label="F1 分数">
          <template #default="{ row }">
            <el-progress :percentage="row.f1 * 100" :status="statusColor(row.f1)" />
          </template>
        </el-table-column>
      </el-table>

      <el-alert
        style="margin-top: 16px"
        title="ROUGE 说明"
        type="info"
        :closable="false"
        show-icon
      >
        <p>ROUGE-1: 基于 unigram（1-gram）的重叠度</p>
        <p>ROUGE-2: 基于 bigram（2-gram）的重叠度</p>
        <p>ROUGE-L: 基于最长公共子序列（LCS）的重叠度</p>
      </el-alert>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { evaluate } from '../api'

const loading = ref(false)
const result = ref(null)

const form = reactive({
  hypothesis: '',
  reference: '',
})

const tableData = computed(() => {
  if (!result.value) return []
  return [
    {
      metric: 'ROUGE-1',
      ...result.value.rouge1,
    },
    {
      metric: 'ROUGE-2',
      ...result.value.rouge2,
    },
    {
      metric: 'ROUGE-L',
      ...result.value.rougeL,
    },
  ]
})

function statusColor(value) {
  if (value >= 0.5) return 'success'
  if (value >= 0.3) return 'warning'
  return 'exception'
}

async function handleEvaluate() {
  if (!form.hypothesis || !form.reference) {
    ElMessage.warning('请输入生成摘要和参考摘要')
    return
  }
  loading.value = true
  try {
    const res = await evaluate(form)
    result.value = res
    ElMessage.success('评估完成')
  } catch (e) {
    ElMessage.error('评估失败')
  } finally {
    loading.value = false
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

.action-bar {
  text-align: center;
  margin: 24px 0;
}

.result-card {
  margin-top: 16px;
}
</style>
