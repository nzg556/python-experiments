import { defineStore } from 'pinia'

const STORAGE_KEY = 'news_summary_history'

function loadFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveToStorage(list) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list))
}

export const useHistoryStore = defineStore('history', {
  state: () => ({
    records: loadFromStorage(),
  }),
  actions: {
    /**
     * 添加一条历史记录
     */
    addRecord(record) {
      const item = {
        id: Date.now(),
        ...record,
        createdAt: new Date().toLocaleString('zh-CN'),
      }
      this.records.unshift(item)
      saveToStorage(this.records)
    },
    /**
     * 删除单条记录
     */
    removeRecord(id) {
      this.records = this.records.filter((r) => r.id !== id)
      saveToStorage(this.records)
    },
    /**
     * 清空所有记录
     */
    clearAll() {
      this.records = []
      saveToStorage(this.records)
    },
  },
})
