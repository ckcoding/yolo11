<script setup>
import { computed } from 'vue'

const props = defineProps({
  jobs: {
    type: Array,
    default: () => [],
  },
  selectedJobId: {
    type: String,
    default: null,
  },
  filter: {
    type: String,
    default: 'all',
  },
})

const emit = defineEmits(['update:filter', 'select'])

function formatTime(value) {
  if (!value) return '-'
  return new Date(value).toLocaleString()
}

function getRelativeTime(value) {
  if (!value) return ''
  const diff = Date.now() - new Date(value).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return '刚刚'
  if (mins < 60) return `${mins} 分钟前`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours} 小时前`
  return `${Math.floor(hours / 24)} 天前`
}

function getProgressPercent(job) {
  if (!job) return 0
  if (job.status === 'completed') return 100
  const total = Number(job.progress_total_iter || 0)
  const current = Number(job.progress_iter || 0)
  if (total > 0) return Math.max(3, Math.min(100, Math.round((current / total) * 100)))
  if (job.progress_epoch > 0)
    return Math.min(95, 12 + Number(job.progress_epoch || 0))
  return job.status === 'running' ? 8 : 0
}

function getStatusIcon(status) {
  switch (status) {
    case 'running': return '▶'
    case 'queued': return '◌'
    case 'completed': return '✓'
    case 'failed': return '✗'
    case 'stopped': return '■'
    default: return '·'
  }
}

const filteredJobs = computed(() => {
  if (props.filter === 'all') return props.jobs
  if (props.filter === 'running') {
    return props.jobs.filter((job) => ['queued', 'running'].includes(job.status))
  }
  return props.jobs.filter((job) => job.status === props.filter)
})

const filterOptions = [
  { label: '全部', value: 'all', icon: '◎' },
  { label: '运行中', value: 'running', icon: '▶' },
  { label: '失败', value: 'failed', icon: '✗' },
  { label: '完成', value: 'completed', icon: '✓' },
]
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div class="panel-header">
        <div class="panel-header-left">
          <div class="panel-icon panel-icon--success">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>
          </div>
          <div>
            <h2>任务队列</h2>
            <p class="panel-desc">点击查看详情 · 共 {{ filteredJobs.length }} / {{ jobs.length }}</p>
          </div>
        </div>
      </div>
    </template>

    <!-- Filter tabs (pill style) -->
    <div class="filter-bar">
      <button
        v-for="item in filterOptions"
        :key="item.value"
        class="filter-tab"
        :class="{ active: filter === item.value }"
        @click="emit('update:filter', item.value)"
      >
        <span class="filter-icon">{{ item.icon }}</span>
        {{ item.label }}
      </button>
    </div>

    <!-- Job list -->
    <div class="job-list">
      <template v-if="filteredJobs.length">
        <div
          v-for="job in filteredJobs"
          :key="job.job_id"
          class="job-item"
          :class="{ active: selectedJobId === job.job_id }"
          @click="emit('select', job.job_id)"
        >
          <div class="job-row1">
            <div class="job-name-wrap">
              <span class="status-indicator" :class="'si-' + job.status">{{ getStatusIcon(job.status) }}</span>
              <strong>{{ job.job_name || job.job_id }}</strong>
            </div>
            <el-tag size="small" effect="dark"
              :type="job.status === 'completed' ? 'success' : job.status === 'failed' ? 'danger' : job.status === 'stopped' ? 'warning' : 'info'">
              {{ job.status }}
            </el-tag>
          </div>
          <div class="job-row2">
            <span class="job-time">{{ getRelativeTime(job.updated_at) }}</span>
            <span class="job-msg">{{ job.latest_message || '等待输出...' }}</span>
          </div>
          <div class="job-row3">
            <div class="job-chips">
              <span class="chip chip-arch">{{ job.arch }}</span>
              <span class="chip">E {{ job.progress_epoch || 0 }}</span>
              <span class="chip">I {{ job.progress_iter || 0 }}/{{ job.progress_total_iter || 0 }}</span>
            </div>
            <div class="job-progress">
              <div class="pg-track">
                <div class="pg-fill"
                  :class="{
                    'pg-ok': job.status === 'completed',
                    'pg-fail': job.status === 'failed',
                    'pg-running': job.status === 'running',
                  }"
                  :style="{ width: getProgressPercent(job) + '%' }"
                ></div>
              </div>
              <span class="pg-num">{{ getProgressPercent(job) }}%</span>
            </div>
          </div>
        </div>
      </template>
      <template v-else>
        <div class="job-empty">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="color: var(--muted-soft); margin-bottom: 10px"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>
          <p>暂无任务</p>
          <small>提交一个训练任务开始使用</small>
        </div>
      </template>
    </div>
  </el-card>
</template>

<style scoped>
.panel-header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.panel-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.panel-icon {
  width: 40px;
  height: 40px;
  border-radius: 11px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.panel-icon--success {
  background: var(--success-soft);
  color: var(--success);
  box-shadow: 0 0 12px rgba(34, 197, 94, 0.15);
}

.panel-header h2 { font-size: 16px; font-weight: 700; color: var(--ink); }
.panel-desc { font-size: 12px; color: var(--muted); margin-top: 1px; }

/* filter bar */
.filter-bar {
  display: flex;
  gap: 3px;
  padding: 3px;
  background: rgba(255,255,255,0.03);
  border-radius: 10px;
  margin-bottom: 14px;
  border: 1px solid var(--glass-border);
}

.filter-tab {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  padding: 7px 8px;
  border: none;
  border-radius: 8px;
  background: transparent;
  color: var(--muted);
  font-size: 12px;
  font-weight: 500;
  font-family: inherit;
  cursor: pointer;
  transition: var(--transition);
}

.filter-icon {
  font-size: 10px;
}

.filter-tab:hover { color: var(--ink); background: rgba(255,255,255,0.04); }
.filter-tab.active {
  background: var(--accent);
  color: #fff;
  box-shadow: 0 2px 8px var(--accent-glow);
}

/* job list */
.job-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 580px;
  overflow-y: auto;
}

.job-item {
  padding: 14px 16px;
  border-radius: var(--radius);
  background: var(--panel-soft);
  border: 1px solid var(--glass-border);
  cursor: pointer;
  transition: var(--transition);
}

.job-item:hover {
  border-color: var(--line-strong);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
}

.job-item.active {
  border-color: var(--accent);
  background: var(--accent-soft);
  box-shadow: 0 0 0 2px var(--accent-glow);
}

.job-row1 {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}

.job-name-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.status-indicator {
  width: 20px;
  height: 20px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  flex-shrink: 0;
  font-weight: 700;
}

.si-running { background: var(--success-soft); color: var(--success); }
.si-queued { background: var(--info-soft); color: var(--info); }
.si-completed { background: var(--success-soft); color: var(--success); }
.si-failed { background: var(--danger-soft); color: var(--danger); }
.si-stopped { background: var(--warning-soft); color: var(--warning); }
.si-interrupted { background: var(--warning-soft); color: var(--warning); }

.job-row1 strong {
  font-size: 13px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--ink);
}

.job-row2 {
  display: flex;
  gap: 10px;
  margin-top: 5px;
  font-size: 11px;
  color: var(--muted);
}

.job-time {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--muted-soft);
  flex-shrink: 0;
  padding: 1px 6px;
  border-radius: 4px;
  background: rgba(255,255,255,0.03);
}

.job-msg {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.job-row3 {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 10px;
}

.job-chips {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.chip {
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 10px;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 500;
  background: rgba(255,255,255,0.04);
  color: var(--muted);
  border: 1px solid var(--glass-border);
}

.chip-arch {
  color: var(--accent-hover);
  border-color: rgba(99, 102, 241, 0.15);
  background: var(--accent-soft);
}

/* progress */
.job-progress {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 90px;
}

.pg-track {
  flex: 1;
  height: 4px;
  border-radius: 2px;
  background: rgba(255,255,255,0.06);
  overflow: hidden;
}

.pg-fill {
  height: 100%;
  border-radius: 2px;
  background: var(--accent-gradient);
  transition: width .6s cubic-bezier(.4,0,.2,1);
}
.pg-fill.pg-ok { background: var(--success); }
.pg-fill.pg-fail { background: var(--danger); }
.pg-fill.pg-running {
  background: var(--accent-gradient);
  background-size: 200% 100%;
  animation: gradient-shift 2s ease infinite;
}

.pg-num {
  font-size: 10px;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  color: var(--muted);
  min-width: 28px;
  text-align: right;
}

/* empty */
.job-empty {
  padding: 48px 20px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.job-empty p { font-size: 14px; color: var(--muted); font-weight: 500; }
.job-empty small { font-size: 12px; color: var(--muted-soft); margin-top: 4px; }

@media (max-width: 760px) {
  .job-row3 { flex-direction: column; align-items: flex-start; }
  .job-progress { width: 100%; }
}
</style>
