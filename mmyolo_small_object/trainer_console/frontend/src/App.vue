<script setup>
import { reactive, ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { ElMessage } from 'element-plus'
import CommandPanel from './components/CommandPanel.vue'
import DatasetPanel from './components/DatasetPanel.vue'
import JobsPanel from './components/JobsPanel.vue'
import JobDetailPanel from './components/JobDetailPanel.vue'
import { apiClient } from './api'

const form = reactive({
  job_name: '',
  dataset_path: '',
  prepared_dataset_path: '',
  device_type: 'cuda',
  arch: 'rtmdet',
  img_scale: 1024,
  max_epochs: 200,
  save_epoch_intervals: 10,
  tile_size: 1280,
  tile_overlap: 320,
  min_intersection_ratio: 0.4,
  min_bbox_side: 2.0,
  max_empty_tiles: 2,
  train_batch_size: 8,
  val_batch_size: 4,
  train_workers: 4,
  val_workers: 2,
  base_lr: 0.004,
  load_from: '',
  prepare_dataset: true,
  overwrite_prepared: true,
  launcher: 'none',
  device_visible_ids: '0',
  extra_train_args: '',
})

const overview = reactive({
  jobs: 0,
  running: 0,
  datasets: 0,
})

const jobs = ref([])
const selectedJobId = ref(null)
const selectedJob = ref(null)
const jobFilter = ref('all')
const datasets = ref([])
const scanning = ref(false)
const scanStatus = ref('正在加载数据集...')
const logText = ref('等待任务日志...')
const autoRefresh = ref(true)
const presets = ref({})
const loadingJobActions = ref(false)

let pollTimer = null

const runningCount = computed(
  () => jobs.value.filter((job) => ['queued', 'running'].includes(job.status)).length,
)

const currentTime = ref(new Date().toLocaleTimeString())
let clockTimer = null

function updateOverview() {
  overview.jobs = jobs.value.length
  overview.running = runningCount.value
  overview.datasets = datasets.value.length
}

async function loadPresets() {
  try {
    const data = await apiClient.getPresets()
    presets.value = data.recommended || {}
    Object.entries(presets.value).forEach(([key, value]) => {
      if (key in form) {
        form[key] = value
      }
    })
  } catch (error) {
    ElMessage.error(error.message)
  }
}

async function loadJobs() {
  const data = await apiClient.getJobs()
  jobs.value = data
  updateOverview()
}

async function loadJobDetail() {
  if (!selectedJobId.value) return
  try {
    const job = await apiClient.getJob(selectedJobId.value)
    selectedJob.value = job
    const log = await apiClient.getJobLogs(selectedJobId.value, 30000)
    logText.value = log.text || '暂无日志'
  } catch (error) {
    ElMessage.error(error.message)
  }
}

async function handleSubmitJob() {
  loadingJobActions.value = true
  try {
    const job = await apiClient.createJob({ ...form })
    selectedJobId.value = job.job_id
    ElMessage.success(`任务已提交: ${job.job_name || job.job_id}`)
    await loadJobs()
    await loadJobDetail()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loadingJobActions.value = false
  }
}

async function handleStopJob() {
  if (!selectedJobId.value) return
  loadingJobActions.value = true
  try {
    await apiClient.stopJob(selectedJobId.value)
    await loadJobs()
    await loadJobDetail()
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    loadingJobActions.value = false
  }
}

async function handleInspectDataset() {
  if (!form.dataset_path) {
    ElMessage.warning('请先填写数据集路径')
    return
  }
  try {
    const result = await apiClient.inspectDataset(form.dataset_path)
    const splitNames = Object.keys(result.splits || {})
    const splitInfo = splitNames.length ? splitNames.join('/') : '无标准 split'
    ElMessage.success(`${result.dataset_type.toUpperCase()} 数据集 · ${splitInfo}`)
  } catch (error) {
    ElMessage.error(error.message)
  }
}

async function handleScanDatasets() {
  if (scanning.value) return
  scanning.value = true
  scanStatus.value = '正在扫描挂载目录中的数据集...'
  try {
    const data = await apiClient.scanDatasets()
    datasets.value = data.datasets || []
    scanStatus.value = data.scan_root_exists
      ? `扫描完成，共发现 ${data.total} 个数据集 (${data.scan_root})`
      : `扫描根目录 ${data.scan_root} 不存在，请确认路径已正确挂载`
    updateOverview()
  } catch (error) {
    scanStatus.value = `扫描失败: ${error.message}`
    datasets.value = []
  } finally {
    scanning.value = false
  }
}

function handleUseDataset(path) {
  form.dataset_path = path
  ElMessage.success(`已选择数据集: ${path}`)
}

function handleSelectJob(jobId) {
  selectedJobId.value = jobId
  loadJobs()
  loadJobDetail()
}

function toggleAutoRefresh() {
  autoRefresh.value = !autoRefresh.value
}

onMounted(async () => {
  await loadPresets()
  await loadJobs()
  await handleScanDatasets()

  pollTimer = setInterval(async () => {
    if (!autoRefresh.value) return
    await loadJobs()
    await loadJobDetail()
  }, 8000)

  clockTimer = setInterval(() => {
    currentTime.value = new Date().toLocaleTimeString()
  }, 1000)
})

onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer)
  if (clockTimer) clearInterval(clockTimer)
})
</script>

<template>
  <main class="mmyolo-console-shell">
    <!-- Hero header -->
    <section class="hero-section">
      <div class="hero-glow"></div>
      <div class="hero-content">
        <div class="hero-left">
          <div class="hero-badge">
            <span class="hero-dot"></span>
            MMYOLO Console
          </div>
          <h1>小目标训练控制台</h1>
          <p class="hero-sub">数据发现 · 参数配置 · 任务调度 · 实时监控</p>
        </div>
        <div class="hero-right">
          <div class="hero-stats">
            <div class="stat-card">
              <div class="stat-icon-wrap stat-icon--info">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18M3 9h18"/></svg>
              </div>
              <div class="stat-body">
                <span class="stat-num">{{ overview.jobs }}</span>
                <span class="stat-label">总任务</span>
              </div>
            </div>
            <div class="stat-card">
              <div class="stat-icon-wrap stat-icon--success">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              </div>
              <div class="stat-body">
                <span class="stat-num">{{ overview.running }}</span>
                <span class="stat-label">运行中</span>
              </div>
            </div>
            <div class="stat-card">
              <div class="stat-icon-wrap stat-icon--warning">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
              </div>
              <div class="stat-body">
                <span class="stat-num">{{ overview.datasets }}</span>
                <span class="stat-label">数据集</span>
              </div>
            </div>
            <div class="stat-card stat-card--clickable" @click="toggleAutoRefresh">
              <div class="stat-icon-wrap" :class="autoRefresh ? 'stat-icon--accent' : 'stat-icon--muted'">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
              </div>
              <div class="stat-body">
                <span class="stat-num" :class="{ 'text-accent': autoRefresh }">{{ autoRefresh ? 'ON' : 'OFF' }}</span>
                <span class="stat-label">自动刷新</span>
              </div>
            </div>
          </div>
          <div class="hero-clock">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            <span>{{ currentTime }}</span>
          </div>
        </div>
      </div>
    </section>

    <!-- Workspace: Config + Datasets -->
    <section class="mmyolo-workspace-grid">
      <CommandPanel
        v-model="form"
        :presets="presets"
        :loading="loadingJobActions"
        @submit="handleSubmitJob"
        @inspect-dataset="handleInspectDataset"
        @apply-presets="loadPresets"
      />
      <DatasetPanel
        :datasets="datasets"
        :scanning="scanning"
        :scan-status="scanStatus"
        @scan="handleScanDatasets"
        @use-dataset="handleUseDataset"
      />
    </section>

    <!-- Monitor: Jobs + Detail -->
    <section class="mmyolo-monitor-grid">
      <JobsPanel
        :jobs="jobs"
        v-model:filter="jobFilter"
        :selected-job-id="selectedJobId || ''"
        @select="handleSelectJob"
      />
      <JobDetailPanel
        :job="selectedJob"
        :log-text="logText"
        :auto-refresh="autoRefresh"
        :loading="loadingJobActions"
        @stop="handleStopJob"
        @reload="loadJobDetail"
        @toggle-auto="toggleAutoRefresh"
      />
    </section>
  </main>
</template>

<style scoped>
/* Hero */
.hero-section {
  position: relative;
  padding: 28px 32px;
  border-radius: var(--radius-2xl);
  background: var(--panel);
  border: 1px solid var(--glass-border);
  box-shadow: var(--shadow-card);
  backdrop-filter: blur(24px) saturate(1.5);
  -webkit-backdrop-filter: blur(24px) saturate(1.5);
  overflow: hidden;
}

.hero-glow {
  position: absolute;
  top: -60%;
  left: -10%;
  width: 60%;
  height: 200%;
  background: radial-gradient(ellipse at center, rgba(99,102,241,0.08) 0%, transparent 70%);
  pointer-events: none;
  animation: float 8s ease-in-out infinite;
}

.hero-content {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px 14px 4px 8px;
  background: var(--accent-soft);
  border: 1px solid rgba(99, 102, 241, 0.15);
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .05em;
  color: var(--accent-hover);
  text-transform: uppercase;
}

.hero-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
  animation: pulse-dot 2.5s ease-in-out infinite;
}

.hero-section h1 {
  font-size: 28px;
  font-weight: 800;
  letter-spacing: -.03em;
  margin-top: 10px;
  background: linear-gradient(135deg, var(--ink) 30%, var(--accent-hover) 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-sub {
  font-size: 13px;
  color: var(--muted);
  margin-top: 4px;
  letter-spacing: .02em;
}

.hero-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 12px;
}

/* Stats */
.hero-stats {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px 10px 10px;
  border-radius: var(--radius);
  border: 1px solid var(--glass-border);
  background: var(--panel-soft);
  transition: var(--transition);
  backdrop-filter: blur(10px);
}

.stat-card:hover {
  border-color: var(--line-strong);
  box-shadow: var(--shadow-sm);
  transform: translateY(-1px);
}

.stat-card--clickable {
  cursor: pointer;
  user-select: none;
}

.stat-icon-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 38px;
  height: 38px;
  border-radius: 10px;
  flex-shrink: 0;
  transition: var(--transition);
}

.stat-icon--info    { background: var(--info-soft); color: var(--info); }
.stat-icon--success { background: var(--success-soft); color: var(--success); }
.stat-icon--warning { background: var(--warning-soft); color: var(--warning); }
.stat-icon--accent  { background: var(--accent-soft); color: var(--accent); }
.stat-icon--muted   { background: rgba(255,255,255,0.04); color: var(--muted); }

.stat-body {
  display: flex;
  flex-direction: column;
  line-height: 1.2;
}

.stat-num {
  font-size: 18px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  color: var(--ink);
}

.stat-label {
  font-size: 10px;
  color: var(--muted);
  font-weight: 500;
  letter-spacing: .03em;
}

.text-accent { color: var(--accent) !important; }

.hero-clock {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--muted);
  padding: 5px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--glass-border);
}
</style>
