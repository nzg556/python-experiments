<template>
  <div class="news-library">
    <el-card shadow="never">
      <template #header>
        <div class="card-header">
          <el-icon :size="20" color="#409eff"><Collection /></el-icon>
          <span class="header-title">新闻文本数据查询库</span>
          <el-tag type="info" size="small">共 {{ total }} 条新闻</el-tag>
          <div style="margin-left: auto; display: flex; gap: 8px">
            <el-input
              v-model="keyword"
              placeholder="搜索关键词、标题、正文"
              clearable
              style="width: 260px"
              @keyup.enter="handleSearch"
              @clear="handleSearch"
            >
              <template #prefix><el-icon><Search /></el-icon></template>
            </el-input>
            <el-select v-model="category" placeholder="分类" style="width: 120px" @change="handleSearch">
              <el-option v-for="c in categories" :key="c" :label="c" :value="c" />
            </el-select>
            <el-button type="primary" @click="handleSearch">查询</el-button>
          </div>
        </div>
      </template>

      <el-table :data="items" v-loading="loading" stripe style="width: 100%" @row-click="showDetail">
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="title" label="标题" min-width="200" show-overflow-tooltip />
        <el-table-column prop="category" label="分类" width="90">
          <template #default="{ row }">
            <el-tag size="small">{{ row.category }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="text" label="正文预览" min-width="280" show-overflow-tooltip />
        <el-table-column label="操作" width="140" align="center">
          <template #default="{ row }">
            <el-button size="small" link type="primary" @click.stop="loadToSummarize(row)">载入摘要</el-button>
            <el-button size="small" link type="success" @click.stop="showDetail(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="pagination-wrap">
        <el-pagination
          v-model:current-page="page"
          v-model:page-size="pageSize"
          :total="total"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next, jumper"
          @current-change="loadData"
          @size-change="handleSizeChange"
        />
      </div>
    </el-card>

    <!-- 详情对话框 -->
    <el-dialog v-model="detailVisible" title="新闻详情" width="70%" top="5vh">
      <div v-if="detailItem">
        <h3 style="margin: 0 0 10px">{{ detailItem.title }}</h3>
        <el-tag size="small" style="margin-bottom: 10px">{{ detailItem.category }}</el-tag>
        <el-divider />
        <h4>正文</h4>
        <p class="news-text">{{ detailItem.text }}</p>
        <el-divider />
        <h4>参考摘要</h4>
        <p class="news-summary">{{ detailItem.summary }}</p>
        <el-divider />
        <div style="text-align: right">
          <el-button type="primary" @click="loadToSummarize(detailItem)">载入到摘要生成</el-button>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { newsLibrarySearch, newsLibraryCategories, newsLibraryDetail } from '../api'

const router = useRouter()
const keyword = ref('')
const category = ref('全部')
const categories = ref(['全部'])
const page = ref(1)
const pageSize = ref(20)
const total = ref(0)
const items = ref([])
const loading = ref(false)
const detailVisible = ref(false)
const detailItem = ref(null)

async function loadData() {
  loading.value = true
  try {
    const res = await newsLibrarySearch({
      keyword: keyword.value,
      category: category.value,
      page: page.value,
      page_size: pageSize.value,
    })
    items.value = res.items
    total.value = res.total
  } catch (e) {
    ElMessage.error('查询失败')
  } finally {
    loading.value = false
  }
}

async function loadCategories() {
  try {
    const res = await newsLibraryCategories()
    categories.value = res.categories
  } catch (e) {
    console.error('分类加载失败', e)
  }
}

function handleSearch() {
  page.value = 1
  loadData()
}

function handleSizeChange() {
  page.value = 1
  loadData()
}

async function showDetail(row) {
  try {
    const res = await newsLibraryDetail(row.id)
    detailItem.value = res
    detailVisible.value = true
  } catch (e) {
    ElMessage.error('加载详情失败')
  }
}

function loadToSummarize(row) {
  // 通过 sessionStorage 把新闻文本传给摘要生成页
  const text = row.full_text || row.text || ''
  sessionStorage.setItem('pendingNewsText', text)
  sessionStorage.setItem('pendingNewsTitle', row.title || '')
  detailVisible.value = false
  router.push('/')
}

onMounted(() => {
  loadCategories()
  loadData()
})
</script>

<style scoped>
.news-library {
  max-width: 1200px;
  margin: 0 auto;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.header-title {
  font-size: 17px;
  font-weight: 600;
}

.pagination-wrap {
  margin-top: 16px;
  display: flex;
  justify-content: center;
}

.news-text {
  white-space: pre-wrap;
  line-height: 1.8;
  color: #303133;
  font-size: 14px;
  max-height: 300px;
  overflow-y: auto;
}

.news-summary {
  color: #409eff;
  font-size: 14px;
  line-height: 1.8;
}
</style>
