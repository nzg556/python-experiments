import axios from 'axios'
import { ElMessage } from 'element-plus'

const request = axios.create({
  baseURL: '/api',
  timeout: 300000, // 多粒度/英文/热力图等叠加时 CPU 推理可能较久
})

// 响应拦截器
request.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || error.message || '请求失败'
    ElMessage.error(message)
    return Promise.reject(error)
  }
)

/**
 * 生成新闻摘要
 * @param {Object} data - { text, model, max_length, min_length, num_beams, length_penalty }
 */
export function summarize(data) {
  return request.post('/summarize', data)
}

/**
 * 全功能摘要：摘要 + 润色 + 抽取式对比 + 多粒度 + 字数统计 +
 * 高亮定位 + 可信度 + 分类 + 关键词 + 英文 + 热力图 + 推荐
 */
export function summarizeFull(data) {
  return request.post('/summarize/full', data)
}

/**
 * 评估摘要质量（ROUGE）
 * @param {Object} data - { hypothesis, reference }
 */
export function evaluate(data) {
  return request.post('/evaluate', data)
}

/**
 * 获取可用模型列表
 */
export function getModels() {
  return request.get('/models')
}

/**
 * 健康检查
 */
export function healthCheck() {
  return request.get('/health')
}

/**
 * 新闻主题分类
 * @param {Object} data - { text }
 */
export function classify(data) {
  return request.post('/analysis/classify', data)
}

/**
 * 摘要可信度评估（幻觉检测）
 * @param {Object} data - { original, summary }
 */
export function credibility(data) {
  return request.post('/analysis/credibility', data)
}

/**
 * 摘要润色
 * @param {Object} data - { text, formal_style }
 */
export function polish(data) {
  return request.post('/analysis/polish', data)
}

/**
 * 保存人工修正摘要（微调补充数据）
 * @param {Object} data - { original_text, model_summary, edited_summary, model, params }
 */
export function saveFeedback(data) {
  return request.post('/feedback/save', data)
}

/**
 * 批量上传 txt 文件开始批量摘要
 * @param {FormData} formData - files: File[], model, max_length, min_length
 */
export function batchUpload(formData) {
  return request.post('/batch/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

/**
 * 查询批量处理进度与结果
 */
export function batchStatus(batchId) {
  return request.get(`/batch/status/${batchId}`)
}

/**
 * 批量结果导出（返回 blob）
 * @param {string} batchId
 * @param {string} format - 'xlsx' | 'csv'
 */
export function batchExport(batchId, format = 'xlsx') {
  return axios.get(`/api/batch/export/${batchId}`, {
    params: { format },
    responseType: 'blob',
  })
}

/**
 * 查看人工反馈记录统计
 */
export function feedbackList() {
  return request.get('/feedback/list')
}

/**
 * 获取内置新闻示例数据集列表
 */
export function getDatasetSamples() {
  return request.get('/dataset/list')
}

/**
 * 按ID加载单条新闻示例
 */
export function loadDatasetSample(id) {
  return request.get(`/dataset/load/${id}`)
}

/**
 * 新闻数据查询库 - 检索
 */
export function newsLibrarySearch(params) {
  return request.get('/news-library/search', { params })
}

/**
 * 新闻数据查询库 - 分类列表
 */
export function newsLibraryCategories() {
  return request.get('/news-library/categories')
}

/**
 * 新闻数据查询库 - 详情
 */
export function newsLibraryDetail(id) {
  return request.get(`/news-library/detail/${id}`)
}

/**
 * 新闻数据查询库 - 互相推荐
 */
export function newsLibraryRecommend(data) {
  return request.post('/news-library/recommend', data)
}

/**
 * 编辑后重新评估摘要指标
 */
export function reAnalyzeSummary(data) {
  return request.post('/analysis/re-analyze', data)
}

/**
 * 导出摘要评估报告（PDF，返回 blob）
 */
export function exportReport(data) {
  return axios.post('/api/analysis/export-report', data, {
    responseType: 'blob',
  })
}

export default request
