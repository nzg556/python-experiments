# 前端应用

基于 Vue 3 + Element Plus 的新闻摘要前端界面。

## 安装依赖

```bash
npm install
```

## 启动开发服务器

```bash
npm run dev
```

访问 http://localhost:5173

## 构建生产版本

```bash
npm run build
```

## 页面说明

- **摘要生成**: 输入新闻文本，选择模型与参数，生成摘要
- **历史记录**: 查看历史生成的摘要，支持删除
- **摘要评估**: 输入生成摘要与参考摘要，计算 ROUGE 分数
- **关于**: 系统介绍与使用说明

## 目录结构

```
src/
├── api/           # API 请求封装
├── layouts/       # 布局组件
├── router/        # 路由配置
├── stores/        # Pinia 状态管理
├── styles/        # 全局样式
├── views/         # 页面组件
├── App.vue        # 根组件
└── main.js        # 入口文件
```

## 代理配置

`vite.config.js` 中已配置代理，将 `/api` 请求转发到 `http://localhost:8000`。
