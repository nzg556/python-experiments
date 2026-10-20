import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    component: () => import('../layouts/MainLayout.vue'),
    children: [
      {
        path: '',
        name: 'summarize',
        component: () => import('../views/SummarizeView.vue'),
        meta: { title: '摘要生成' },
      },
      {
        path: 'history',
        name: 'history',
        component: () => import('../views/HistoryView.vue'),
        meta: { title: '历史记录' },
      },
      {
        path: 'batch',
        name: 'batch',
        component: () => import('../views/BatchView.vue'),
        meta: { title: '批量处理' },
      },
      {
        path: 'news-library',
        name: 'news-library',
        component: () => import('../views/NewsLibraryView.vue'),
        meta: { title: '新闻数据查询库' },
      },
      {
        path: 'evaluate',
        name: 'evaluate',
        component: () => import('../views/EvaluateView.vue'),
        meta: { title: '摘要评估' },
      },
      {
        path: 'about',
        name: 'about',
        component: () => import('../views/AboutView.vue'),
        meta: { title: '关于' },
      },
    ],
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
