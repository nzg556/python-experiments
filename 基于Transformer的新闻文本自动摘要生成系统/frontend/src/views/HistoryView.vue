<template>
  <div class="history-page">
    <el-card shadow="hover">
      <template #header>
        <div class="card-header">
          <el-icon><Clock /></el-icon>
          <span>摘要历史记录</span>
          <el-button
            v-if="historyStore.records.length > 0"
            size="small"
            type="success"
            plain
            style="margin-left: auto"
            @click="exportCsv"
          >
            导出 CSV
          </el-button>
          <el-button
            v-if="historyStore.records.length > 0"
            size="small"
            type="danger"
            plain
            @click="handleClearAll"
          >
            清空全部
          </el-button>
        </div>
      </template>

      <el-empty v-if="historyStore.records.length === 0" description="暂无历史记录"></el-empty>

      <div v-else class="record-list">
        <el-card
          v-for="record in historyStore.records"
          :key="record.id"
          class="record-item"
          shadow="never"
        >
          <div class="record-header">
            <el-tag size="small" type="info">{{ record.model }}</el-tag>
            <span class="record-time">{{ record.createdAt }}</span>
            <el-button
              size="small"
              type="danger"
              text
              @click="handleDelete(record.id)"
            >
              <el-icon><Delete /></el-icon>
            </el-button>
          </div>

          <el-collapse>
            <el-collapse-item title="查看原文" name="original">
              <p class="record-text">{{ record.originalText }}</p>
            </el-collapse-item>
          </el-collapse>

          <div class="record-summary-label">生成摘要：</div>
          <p class="record-summary">{{ record.summary }}</p>

          <div v-if="record.params" class="record-params text-muted">
            参数：max_length={{ record.params.max_length }},
            num_beams={{ record.params.num_beams }},
            penalty={{ record.params.length_penalty }}
          </div>
        </el-card>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ElMessage, ElMessageBox } from 'element-plus'
import { useHistoryStore } from '../stores/history'

const historyStore = useHistoryStore()

function handleDelete(id) {
  historyStore.removeRecord(id)
  ElMessage.success('已删除')
}

async function handleClearAll() {
  try {
    await ElMessageBox.confirm('确定要清空所有历史记录吗？', '提示', {
      type: 'warning',
    })
    historyStore.clearAll()
    ElMessage.success('已清空所有记录')
  } catch {
    // 用户取消
  }
}

function csvEscape(v) {
  const s = String(v ?? '')
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

function exportCsv() {
  const header = ['生成时间', '模型', '原文', '摘要', 'max_length', 'num_beams', 'length_penalty']
  const rows = historyStore.records.map((r) => [
    r.createdAt,
    r.model,
    r.originalText,
    r.summary,
    r.params?.max_length ?? '',
    r.params?.num_beams ?? '',
    r.params?.length_penalty ?? '',
  ])
  const csv = '﻿' + [header, ...rows].map((row) => row.map(csvEscape).join(',')).join('\r\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `摘要历史记录_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  window.URL.revokeObjectURL(url)
  ElMessage.success('已导出 CSV')
}
</script>

<style scoped>
.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
}

.record-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.record-item {
  border: 1px solid #ebeef5;
}

.record-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.record-time {
  color: #909399;
  font-size: 13px;
  margin-left: auto;
}

.record-text {
  color: #606266;
  line-height: 1.6;
  font-size: 13px;
}

.record-summary-label {
  margin-top: 12px;
  font-weight: 600;
  color: #409eff;
}

.record-summary {
  margin-top: 8px;
  line-height: 1.8;
  color: #303133;
  padding: 12px;
  background: #f0f9ff;
  border-left: 3px solid #409eff;
  border-radius: 4px;
}

.record-params {
  margin-top: 8px;
  font-size: 12px;
}
</style>
