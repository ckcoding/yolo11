<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({
  job: {
    type: Object,
    default: null,
  },
  logText: {
    type: String,
    default: '等待任务日志...',
  },
  autoRefresh: Boolean,
  loading: Boolean,
})

const emit = defineEmits(['stop', 'reload', 'toggle-auto'])

const logViewerRef = ref(null)

watch(
  () => props.logText,
  async () => {
    await nextTick()
    const el = logViewerRef.value
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 64
    if (nearBottom || props.logText === '暂无日志') {
      el.scrollTop = el.scrollHeight
    }
  },
)

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

function formatTime(value) {
  if (!value) return '-'
  return new Date(value).toLocaleString()
}

function getStatusColor(status) {
  switch (status) {
    case 'running': return 'var(--success)'
    case 'completed': return 'var(--success)'
    case 'failed': return 'var(--danger)'
    case 'stopped': return 'var(--warning)'
    default: return 'var(--info)'
  }
}
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div class="panel-header">
        <div class="panel-header-left">
          <div class="panel-icon panel-icon--info">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
          </div>
          <div>
            <h2>任务详情</h2>
            <p class="panel-desc">实时监控与日志</p>
          </div>
        </div>
        <div class="panel-actions">
          <el-button type="danger" size="small"
            :disabled="!job || !['queued', 'running'].includes(job.status) || loading"
            @click="emit('stop')">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 3px"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>
            停止
          </el-button>
          <el-button size="small" :disabled="!job || loading" @click="emit('reload')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 3px"><path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
            刷新
          </el-button>
          <div class="auto-pill" :class="{ on: autoRefresh }" @click="emit('toggle-auto')">
            <span class="auto-dot"></span>
            {{ autoRefresh ? '自动刷新' : '已暂停' }}
          </div>
        </div>
      </div>
    </template>

    <div class="detail-body" v-loading="loading">
      <template v-if="job">
        <!-- Title row with status indicator line -->
        <div class="d-title-row">
          <div class="d-status-line" :style="{ background: getStatusColor(job.status) }"></div>
          <div class="d-title-content">
            <h3>{{ job.job_name || job.job_id }}</h3>
            <div class="d-tags">
              <el-tag :type="job.status === 'completed' ? 'success' : job.status === 'failed' ? 'danger' : 'info'" effect="dark">
                {{ job.status }}
              </el-tag>
              <el-tag effect="plain" size="small">{{ job.stage }}</el-tag>
              <el-tag effect="plain" size="small">{{ job.arch }}</el-tag>
            </div>
          </div>
        </div>

        <!-- Meta -->
        <div class="d-meta">
          <span>创建 {{ formatTime(job.created_at) }}</span>
          <span class="meta-sep">·</span>
          <span>更新 {{ formatTime(job.updated_at) }}</span>
          <span class="meta-sep">·</span>
          <span>退出码 {{ job.exit_code ?? '-' }}</span>
        </div>

        <!-- Progress -->
        <div class="d-progress">
          <div class="d-progress-head">
            <strong>执行进度</strong>
            <span class="d-progress-info">epoch {{ job.progress_epoch || 0 }} · iter {{ job.progress_iter || 0 }}/{{ job.progress_total_iter || 0 }}</span>
          </div>
          <div class="pg-track-lg">
            <div class="pg-fill-lg"
              :class="{
                'pg-ok': job.status === 'completed',
                'pg-fail': job.status === 'failed',
                'pg-running': job.status === 'running',
              }"
              :style="{ width: getProgressPercent(job) + '%' }"
            ></div>
          </div>
          <div class="d-progress-footer">
            <span class="pg-pct">{{ getProgressPercent(job) }}%</span>
          </div>
        </div>

        <!-- Stats grid -->
        <div class="d-stat-grid">
          <div class="d-stat-card">
            <div class="d-stat-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
            </div>
            <div>
              <small>数据集状态</small>
              <strong :class="job.prepared_dataset_path ? 'val-ok' : 'val-pending'">{{ job.prepared_dataset_path ? 'READY' : 'PENDING' }}</strong>
            </div>
          </div>
          <div class="d-stat-card">
            <div class="d-stat-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/></svg>
            </div>
            <div>
              <small>检查点</small>
              <strong>{{ (job.checkpoints || []).length }}</strong>
            </div>
          </div>
          <div class="d-stat-card d-stat-wide">
            <div class="d-stat-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            </div>
            <div style="min-width:0">
              <small>最新消息</small>
              <strong class="msg-val">{{ job.latest_message || '-' }}</strong>
            </div>
          </div>
        </div>

        <!-- Paths -->
        <section class="d-section">
          <h4>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 5px;vertical-align: -2px"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
            关键路径
          </h4>
          <div class="path-list">
            <div class="path-row" v-for="item in [
              { label: '源数据集', value: job.dataset_path },
              { label: '准备后数据', value: job.prepared_dataset_path },
              { label: '配置文件', value: job.config_path },
              { label: '工作目录', value: job.work_dir },
              { label: '日志文件', value: job.log_path },
            ]" :key="item.label">
              <span class="path-label">{{ item.label }}</span>
              <code>{{ item.value || '-' }}</code>
            </div>
          </div>
        </section>

        <!-- Metrics -->
        <section class="d-section">
          <h4>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 5px;vertical-align: -2px"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
            训练指标
          </h4>
          <div class="metric-row" v-if="job.metrics && Object.keys(job.metrics).length">
            <div v-for="(value, key) in job.metrics" :key="key" class="metric-chip">
              <span class="m-key">{{ key }}</span>
              <span class="m-val">{{ value }}</span>
            </div>
          </div>
          <span v-else class="muted-text">暂无指标</span>
        </section>

        <!-- Checkpoints -->
        <section class="d-section">
          <h4>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 5px;vertical-align: -2px"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/></svg>
            检查点
          </h4>
          <div v-if="job.checkpoints && job.checkpoints.length" class="ckpt-list">
            <span v-for="path in job.checkpoints" :key="path" class="ckpt-chip">
              {{ path.split('/').pop() }}
            </span>
          </div>
          <span v-else class="muted-text">暂无检查点</span>
        </section>
      </template>
      <template v-else>
        <div class="d-empty">
          <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="color: var(--muted-soft); margin-bottom: 12px"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
          <p>选择一个任务查看详情</p>
          <small>从左侧任务队列中点击选择</small>
        </div>
      </template>

      <!-- Log viewer (always visible) -->
      <div class="log-box">
        <div class="log-head">
          <h4>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 5px;vertical-align: -2px"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
            实时日志
          </h4>
          <span class="muted-text">自动滚动到末尾</span>
        </div>
        <pre ref="logViewerRef" class="mmyolo-log-viewer">{{ logText }}</pre>
      </div>
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

.panel-icon--info {
  background: var(--info-soft);
  color: var(--info);
  box-shadow: 0 0 12px rgba(59, 130, 246, 0.15);
}

.panel-header h2 { font-size: 16px; font-weight: 700; color: var(--ink); }
.panel-desc { font-size: 12px; color: var(--muted); margin-top: 1px; }

.panel-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

/* Auto-refresh pill */
.auto-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--glass-border);
  font-size: 12px;
  color: var(--muted);
  cursor: pointer;
  user-select: none;
  transition: var(--transition);
}

.auto-pill:hover { border-color: var(--line-strong); }

.auto-pill.on { border-color: rgba(34, 197, 94, .2); }

.auto-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--muted-soft);
  transition: var(--transition);
}

.auto-pill.on .auto-dot {
  background: var(--success);
  box-shadow: 0 0 6px rgba(34, 197, 94, .4);
}

/* Body */
.detail-body {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

/* Title */
.d-title-row {
  display: flex;
  gap: 14px;
  align-items: stretch;
}

.d-status-line {
  width: 3px;
  border-radius: 2px;
  flex-shrink: 0;
  min-height: 40px;
  transition: var(--transition);
}

.d-title-content {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.d-title-content h3 {
  font-size: 18px;
  font-weight: 700;
  color: var(--ink);
}

.d-tags { display: flex; gap: 6px; flex-wrap: wrap; }

/* Meta */
.d-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  font-size: 11px;
  color: var(--muted);
  font-family: 'JetBrains Mono', monospace;
  padding: 8px 14px;
  border-radius: var(--radius);
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--glass-border);
}

.meta-sep { color: var(--muted-soft); }

/* Progress */
.d-progress {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.d-progress-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}

.d-progress-head strong { font-size: 13px; color: var(--ink); }
.d-progress-info { font-size: 11px; color: var(--muted); font-family: 'JetBrains Mono', monospace; }

.pg-track-lg {
  height: 8px;
  border-radius: 4px;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--glass-border);
  overflow: hidden;
}

.pg-fill-lg {
  height: 100%;
  border-radius: 4px;
  background: var(--accent-gradient);
  transition: width .8s cubic-bezier(.4,0,.2,1);
}

.pg-fill-lg.pg-ok { background: var(--success); }
.pg-fill-lg.pg-fail { background: var(--danger); }
.pg-fill-lg.pg-running {
  background: var(--accent-gradient-vivid);
  background-size: 200% 100%;
  animation: gradient-shift 2s ease infinite;
}

.d-progress-footer {
  display: flex;
  justify-content: flex-end;
}

.pg-pct {
  font-size: 12px;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
  color: var(--accent-hover);
}

/* Stats grid */
.d-stat-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.d-stat-card {
  padding: 12px 14px;
  border-radius: var(--radius);
  background: var(--panel-soft);
  border: 1px solid var(--glass-border);
  display: flex;
  align-items: center;
  gap: 12px;
  transition: var(--transition);
}

.d-stat-card:hover {
  border-color: var(--line-strong);
}

.d-stat-icon {
  width: 34px;
  height: 34px;
  border-radius: 9px;
  background: rgba(255,255,255,0.04);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  color: var(--muted);
}

.d-stat-card small {
  display: block;
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .04em;
  font-weight: 500;
}

.d-stat-card strong {
  display: block;
  font-size: 14px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  margin-top: 2px;
  color: var(--ink);
}

.val-ok { color: var(--success) !important; }
.val-pending { color: var(--warning) !important; }

.d-stat-wide { grid-column: span 2; }
.msg-val {
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: block;
}

/* Sections */
.d-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.d-section h4 {
  font-size: 13px;
  font-weight: 600;
  color: var(--ink-soft);
  display: flex;
  align-items: center;
}

/* Paths */
.path-list {
  display: flex;
  flex-direction: column;
  gap: 1px;
  border-radius: var(--radius);
  overflow: hidden;
}

.path-row {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 14px;
  background: var(--panel-soft);
  border-left: 3px solid rgba(255,255,255,0.05);
  transition: var(--transition);
}

.path-row:hover {
  border-left-color: var(--accent);
  background: var(--panel-hover);
}

.path-label {
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .04em;
  font-weight: 500;
}

.path-row code {
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--ink-soft);
  word-break: break-all;
}

/* Metrics */
.metric-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.metric-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 8px;
  background: var(--panel-soft);
  border: 1px solid var(--glass-border);
  font-size: 12px;
  transition: var(--transition);
}

.metric-chip:hover {
  border-color: var(--line-strong);
  transform: translateY(-1px);
}

.m-key { color: var(--muted); font-weight: 500; }
.m-val { font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--accent-hover); }

/* Checkpoints */
.ckpt-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.ckpt-chip {
  padding: 4px 12px;
  border-radius: 8px;
  background: var(--panel-soft);
  border: 1px solid var(--glass-border);
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--ink-soft);
  transition: var(--transition);
}

.ckpt-chip:hover {
  border-color: var(--accent);
  color: var(--accent-hover);
  transform: translateY(-1px);
}

/* Log */
.log-box {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.log-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.log-head h4 {
  font-size: 13px;
  font-weight: 600;
  color: var(--ink-soft);
  display: flex;
  align-items: center;
}

/* Empty */
.d-empty {
  padding: 40px 20px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.d-empty p { font-size: 14px; color: var(--muted); font-weight: 500; }
.d-empty small { font-size: 12px; color: var(--muted-soft); margin-top: 4px; }

.muted-text { font-size: 12px; color: var(--muted); }

@media (max-width: 760px) {
  .d-stat-grid { grid-template-columns: 1fr; }
  .d-stat-wide { grid-column: span 1; }
}
</style>
