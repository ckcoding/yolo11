<script setup>
const props = defineProps({
  datasets: {
    type: Array,
    default: () => [],
  },
  scanning: Boolean,
  scanStatus: {
    type: String,
    default: '',
  },
})

const emit = defineEmits(['scan', 'use-dataset'])

function classSummary(ds) {
  if (!ds.classes || !ds.classes.length) return '-'
  return `${ds.num_classes || ds.classes.length} 类`
}

function splitSummary(ds) {
  const stats = ds.split_stats || {}
  return Object.keys(stats).join(' / ') || '-'
}
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div class="panel-header">
        <div class="panel-header-left">
          <div class="panel-icon panel-icon--warning">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
          </div>
          <div>
            <h2>数据集发现</h2>
            <p class="panel-desc">自动扫描挂载目录</p>
          </div>
        </div>
        <el-button type="primary" size="small" :loading="scanning" @click="emit('scan')">
          <svg v-if="!scanning" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
          扫描数据集
        </el-button>
      </div>
    </template>

    <!-- Scan status -->
    <div class="scan-bar" :class="{ 'is-scanning': scanning }">
      <span class="scan-dot" :class="scanning ? 'dot-pulse' : 'dot-ok'"></span>
      <span>{{ scanStatus || '正在加载数据集...' }}</span>
    </div>

    <!-- Dataset cards -->
    <div class="ds-list">
      <template v-if="datasets && datasets.length">
        <div v-for="ds in datasets" :key="ds.path" class="ds-card">
          <div class="ds-card-top">
            <div class="ds-name">
              <div class="ds-type-dot" :class="ds.dataset_type === 'coco' ? 'dot-coco' : 'dot-yolo'"></div>
              <h3>{{ ds.name }}</h3>
              <el-tag size="small" :type="ds.dataset_type === 'yolo' ? 'info' : 'warning'" effect="dark">
                {{ ds.dataset_type?.toUpperCase() }}
              </el-tag>
            </div>
            <el-button type="primary" size="small" @click.stop="emit('use-dataset', ds.path)">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 3px"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
              使用
            </el-button>
          </div>
          <div class="ds-path">{{ ds.path }}</div>
          <div class="ds-stats">
            <div class="ds-stat">
              <small>类别</small>
              <strong>{{ classSummary(ds) }}</strong>
            </div>
            <div class="ds-stat">
              <small>图片</small>
              <strong>{{ ds.total_images?.toLocaleString() ?? '-' }}</strong>
            </div>
            <div class="ds-stat">
              <small>标注</small>
              <strong>{{ ds.total_labels?.toLocaleString() ?? '-' }}</strong>
            </div>
            <div class="ds-stat">
              <small>Splits</small>
              <strong>{{ splitSummary(ds) }}</strong>
            </div>
          </div>
        </div>
      </template>
      <template v-else>
        <div class="ds-empty">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="color: var(--muted-soft); margin-bottom: 12px"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
          <p>未发现数据集</p>
          <small>请确认挂载目录下有 YOLO / COCO 数据集</small>
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

.panel-icon--warning {
  background: var(--warning-soft);
  color: var(--warning);
  box-shadow: 0 0 12px rgba(245, 158, 11, 0.15);
}

.panel-header h2 { font-size: 16px; font-weight: 700; color: var(--ink); }
.panel-desc { font-size: 12px; color: var(--muted); margin-top: 1px; }

/* scan bar */
.scan-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 14px;
  border-radius: var(--radius);
  background: var(--panel-soft);
  border: 1px solid var(--glass-border);
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 14px;
  transition: var(--transition);
}

.scan-bar.is-scanning {
  border-color: rgba(59, 130, 246, .2);
  background: var(--info-soft);
  color: var(--info);
}

.scan-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}

.dot-ok { background: var(--success); box-shadow: 0 0 6px rgba(34, 197, 94, 0.4); }
.dot-pulse {
  background: var(--info);
  animation: pulse-dot 1.5s ease-in-out infinite;
}

/* dataset list */
.ds-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 600px;
  overflow-y: auto;
}

.ds-card {
  padding: 14px 16px;
  border-radius: var(--radius);
  background: var(--panel-soft);
  border: 1px solid var(--glass-border);
  transition: var(--transition);
}

.ds-card:hover {
  border-color: var(--line-strong);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
}

.ds-card-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.ds-name {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.ds-type-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.dot-coco { background: var(--warning); box-shadow: 0 0 6px rgba(245, 158, 11, 0.3); }
.dot-yolo { background: var(--info); box-shadow: 0 0 6px rgba(59, 130, 246, 0.3); }

.ds-name h3 {
  font-size: 14px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--ink);
}

.ds-path {
  margin-top: 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--muted-soft);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ds-stats {
  display: flex;
  gap: 20px;
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--glass-border);
}

.ds-stat {
  display: flex;
  flex-direction: column;
}

.ds-stat small {
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .04em;
  font-weight: 500;
}

.ds-stat strong {
  font-size: 14px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  color: var(--ink);
}

.ds-empty {
  padding: 48px 20px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.ds-empty p { font-size: 14px; color: var(--muted); font-weight: 500; }
.ds-empty small { font-size: 12px; color: var(--muted-soft); margin-top: 4px; }
</style>
