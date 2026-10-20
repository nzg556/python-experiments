<template>
  <el-container class="layout-container">
    <el-header class="layout-header">
      <div class="header-inner">
        <div class="logo">
          <el-icon :size="28" color="#409EFF"><Document /></el-icon>
          <span class="title">新闻文本自动摘要生成系统</span>
        </div>
        <el-menu
          :default-active="activeMenu"
          mode="horizontal"
          router
          class="nav-menu"
          background-color="transparent"
          text-color="#ffffff"
          active-text-color="#ffd04b"
        >
          <el-menu-item index="/">
            <el-icon><EditPen /></el-icon>
            <span>摘要生成</span>
          </el-menu-item>
          <el-menu-item index="/history">
            <el-icon><Clock /></el-icon>
            <span>历史记录</span>
          </el-menu-item>
          <el-menu-item index="/batch">
            <el-icon><Files /></el-icon>
            <span>批量处理</span>
          </el-menu-item>
          <el-menu-item index="/news-library">
            <el-icon><Collection /></el-icon>
            <span>新闻数据查询库</span>
          </el-menu-item>
          <el-menu-item index="/evaluate">
            <el-icon><DataAnalysis /></el-icon>
            <span>摘要评估</span>
          </el-menu-item>
          <el-menu-item index="/about">
            <el-icon><InfoFilled /></el-icon>
            <span>关于</span>
          </el-menu-item>
        </el-menu>
      </div>
    </el-header>
    <el-main class="layout-main">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </el-main>
    <el-footer class="layout-footer">
      <span>基于 Transformer 的新闻文本自动摘要生成系统</span>
    </el-footer>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { Document, EditPen, Clock, Files, DataAnalysis, InfoFilled, Collection } from '@element-plus/icons-vue'

const route = useRoute()
const activeMenu = computed(() => route.path)
</script>

<style scoped>
.layout-container {
  min-height: 100vh;
}

.layout-header {
  background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
  color: #fff;
  padding: 0;
  height: 64px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.header-inner {
  max-width: 1400px;
  margin: 0 auto;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
}

.title {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: 1px;
}

.nav-menu {
  border-bottom: none;
}

.layout-main {
  max-width: 1400px;
  width: 100%;
  margin: 0 auto;
  padding: 24px;
}

.layout-footer {
  text-align: center;
  color: #909399;
  font-size: 13px;
  padding: 16px 0;
  background: #fff;
  border-top: 1px solid #ebeef5;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
