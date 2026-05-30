const state = {
  selectedJobId: null,
  autoRefresh: true,
  datasets: [],
  scanning: false,
  jobs: [],
  jobFilter: 'all',
  jobsSignature: '',
  selectedJobSignature: '',
  lastLogText: '',
};

function qs(selector) {
  return document.querySelector(selector);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function formatTime(value) {
  if (!value) return '-';
  return new Date(value).toLocaleString();
}

function formatNumber(value) {
  if (!Number.isFinite(Number(value))) return '-';
  return Number(value).toLocaleString();
}

function statusClass(status) {
  return ['running', 'failed', 'completed'].includes(status) ? status : '';
}

function getProgressPercent(job) {
  if (!job) return 0;
  if (job.status === 'completed') return 100;
  const total = Number(job.progress_total_iter || 0);
  const current = Number(job.progress_iter || 0);
  if (total > 0) return Math.max(3, Math.min(100, Math.round((current / total) * 100)));
  if (job.progress_epoch > 0) return Math.min(95, 12 + Number(job.progress_epoch || 0));
  return job.status === 'running' ? 8 : 0;
}

function updateOverview() {
  const jobs = state.jobs || [];
  const running = jobs.filter((job) => ['queued', 'running'].includes(job.status)).length;

  qs('#overviewJobCount').textContent = formatNumber(jobs.length);
  qs('#overviewRunningCount').textContent = formatNumber(running);
  qs('#overviewDatasetCount').textContent = formatNumber(state.datasets.length);
  qs('#refreshState').textContent = state.autoRefresh ? 'ON' : 'OFF';
}

function createJobsSignature(jobs) {
  return jobs.map((job) => [
    job.job_id,
    job.status,
    job.stage,
    job.progress_epoch,
    job.progress_iter,
    job.progress_total_iter,
    job.updated_at,
    job.latest_message,
  ].join('|')).join('||');
}

function createJobDetailSignature(job) {
  if (!job) return '';
  return [
    job.job_id,
    job.status,
    job.stage,
    job.updated_at,
    job.exit_code,
    job.latest_message,
    job.progress_epoch,
    job.progress_iter,
    job.progress_total_iter,
    JSON.stringify(job.metrics || {}),
    JSON.stringify(job.checkpoints || []),
  ].join('|');
}

function getFilteredJobs(jobs) {
  if (state.jobFilter === 'all') return jobs;
  if (state.jobFilter === 'running') {
    return jobs.filter((job) => ['queued', 'running'].includes(job.status));
  }
  return jobs.filter((job) => job.status === state.jobFilter);
}

function updateJobFilterButtons() {
  document.querySelectorAll('.job-filter-btn').forEach((button) => {
    button.classList.toggle('active', button.dataset.filter === state.jobFilter);
  });
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch (error) {
      const text = await response.text();
      if (text) detail = text;
    }
    throw new Error(detail);
  }
  return response.json();
}

function showInspect(message, isError = false) {
  const el = qs('#inspectResult');
  el.textContent = message || '';
  el.className = 'inspect-inline ' + (isError ? 'error' : 'muted');
}

function formToPayload(form) {
  const formData = new FormData(form);
  return {
    job_name: formData.get('job_name') || '',
    dataset_path: formData.get('dataset_path') || '',
    prepared_dataset_path: formData.get('prepared_dataset_path') || '',
    device_type: formData.get('device_type') || 'cuda',
    arch: formData.get('arch') || 'rtmdet',
    img_scale: Number(formData.get('img_scale') || 1024),
    max_epochs: Number(formData.get('max_epochs') || 200),
    tile_size: Number(formData.get('tile_size') || 1280),
    tile_overlap: Number(formData.get('tile_overlap') || 320),
    min_intersection_ratio: Number(formData.get('min_intersection_ratio') || 0.4),
    min_bbox_side: Number(formData.get('min_bbox_side') || 2.0),
    max_empty_tiles: Number(formData.get('max_empty_tiles') || 2),
    train_batch_size: Number(formData.get('train_batch_size') || 8),
    val_batch_size: Number(formData.get('val_batch_size') || 4),
    train_workers: Number(formData.get('train_workers') || 4),
    val_workers: Number(formData.get('val_workers') || 2),
    base_lr: Number(formData.get('base_lr') || 0.004),
    load_from: formData.get('load_from') || '',
    prepare_dataset: form.elements.prepare_dataset.checked,
    overwrite_prepared: form.elements.overwrite_prepared.checked,
    launcher: formData.get('launcher') || 'none',
    device_visible_ids: formData.get('device_visible_ids') || '0',
    extra_train_args: formData.get('extra_train_args') || '',
  };
}

function renderJobs(jobs) {
  const container = qs('#jobsList');
  state.jobs = jobs;
  updateOverview();
  updateJobFilterButtons();

  const filteredJobs = getFilteredJobs(jobs);
  qs('#jobsToolbarMeta').textContent = `显示 ${formatNumber(filteredJobs.length)} / ${formatNumber(jobs.length)} 个任务`;

  if (!filteredJobs.length) {
    container.innerHTML = '<div class="job-card muted">还没有任务，先从上面的编排区提交一个训练任务。</div>';
    return;
  }

  container.innerHTML = filteredJobs.map((job) => {
    const metrics = Object.entries(job.metrics || {})
      .slice(0, 3)
      .map(([key, value]) => `<span class="chip">${escapeHtml(key)}: ${escapeHtml(value)}</span>`)
      .join('');
    const progress = getProgressPercent(job);

    return `
      <div class="job-card ${state.selectedJobId === job.job_id ? 'active' : ''}" data-job-id="${escapeHtml(job.job_id)}">
        <div class="job-card-main">
          <div class="job-card-head">
            <div class="job-card-title">
              <strong>${escapeHtml(job.job_name || job.job_id)}</strong>
              <p>${escapeHtml(job.latest_message || '等待任务输出')}</p>
            </div>
            <div class="job-card-side">
              <span class="chip ${statusClass(job.status)}">${escapeHtml(job.status)}</span>
              <span class="job-card-time">${escapeHtml(formatTime(job.created_at))}</span>
            </div>
          </div>
          <div class="job-card-meta">
            <span class="chip">${escapeHtml(job.stage)}</span>
            <span class="chip">${escapeHtml(job.arch)}</span>
            <span class="chip">E ${escapeHtml(job.progress_epoch || 0)}</span>
            <span class="chip">I ${escapeHtml(job.progress_iter || 0)}/${escapeHtml(job.progress_total_iter || 0)}</span>
          </div>
          <div class="metric-row">${metrics || '<span class="muted">暂无指标</span>'}</div>
        </div>
        <div class="job-card-progress">
          <div class="progress-track"><div class="progress-fill" style="width: ${progress}%"></div></div>
          <small class="muted">${progress}%</small>
        </div>
      </div>
    `;
  }).join('');

  container.querySelectorAll('.job-card').forEach((card) => {
    card.addEventListener('click', () => {
      state.selectedJobId = card.dataset.jobId;
      loadJobs();
      loadJobDetail();
    });
  });
}

function renderJobDetail(job) {
  const container = qs('#jobDetail');
  qs('#stopJobBtn').disabled = !job || !['queued', 'running'].includes(job.status);
  qs('#reloadDetailBtn').disabled = !job;

  if (!job) {
    container.innerHTML = '还没有选中任务。';
    return;
  }

  const metrics = Object.entries(job.metrics || {})
    .map(([key, value]) => `<span class="chip">${escapeHtml(key)}: ${escapeHtml(value)}</span>`)
    .join('');
  const checkpoints = (job.checkpoints || []).length
    ? `<div class="checkpoint-list">${job.checkpoints.map((item) => `<a href="#" onclick="navigator.clipboard.writeText('${item.replace(/'/g, "\\'")}'); return false;">${escapeHtml(item.split('/').pop())}</a>`).join('')}</div>`
    : '<span class="muted">暂无检查点</span>';
  const progress = getProgressPercent(job);

  container.innerHTML = `
    <section class="detail-summary">
      <div class="detail-title-row">
        <h3>${escapeHtml(job.job_name || job.job_id)}</h3>
        <span class="chip ${statusClass(job.status)}">${escapeHtml(job.status)}</span>
        <span class="chip">${escapeHtml(job.stage)}</span>
        <span class="chip">${escapeHtml(job.arch)}</span>
      </div>
      <div class="detail-meta">
        <span class="chip">创建于 ${escapeHtml(formatTime(job.created_at))}</span>
        <span class="chip">更新于 ${escapeHtml(formatTime(job.updated_at))}</span>
        <span class="chip">退出码 ${escapeHtml(job.exit_code ?? '-')}</span>
      </div>
      <div class="detail-progress">
        <div class="detail-progress-meta">
          <strong>执行进度</strong>
          <span class="muted">epoch ${escapeHtml(job.progress_epoch || 0)} / iter ${escapeHtml(job.progress_iter || 0)}/${escapeHtml(job.progress_total_iter || 0)}</span>
        </div>
        <div class="detail-progress-bar"><div class="progress-fill" style="width: ${progress}%"></div></div>
      </div>
      <div class="detail-stats-grid">
        <div class="detail-stat">
          <div class="k">Prepared Dataset</div>
          <div class="v">${escapeHtml(job.prepared_dataset_path ? 'READY' : 'PENDING')}</div>
        </div>
        <div class="detail-stat">
          <div class="k">Checkpoint Count</div>
          <div class="v">${escapeHtml((job.checkpoints || []).length)}</div>
        </div>
        <div class="detail-stat">
          <div class="k">Latest Message</div>
          <div class="v">${escapeHtml(job.latest_message || '-')}</div>
        </div>
      </div>
    </section>

    <section class="detail-section">
      <h4>关键路径</h4>
      <div class="detail-path-list">
        <div class="detail-path-item">
          <span>源数据集</span>
          <code class="detail-path">${escapeHtml(job.dataset_path)}</code>
        </div>
        <div class="detail-path-item">
          <span>准备后数据</span>
          <code class="detail-path">${escapeHtml(job.prepared_dataset_path)}</code>
        </div>
        <div class="detail-path-item">
          <span>配置文件</span>
          <code class="detail-path">${escapeHtml(job.config_path)}</code>
        </div>
        <div class="detail-path-item">
          <span>工作目录</span>
          <code class="detail-path">${escapeHtml(job.work_dir)}</code>
        </div>
        <div class="detail-path-item">
          <span>日志文件</span>
          <code class="detail-path">${escapeHtml(job.log_path)}</code>
        </div>
      </div>
    </section>

    <section class="detail-section">
      <h4>训练指标</h4>
      <div class="metric-row">${metrics || '<span class="muted">暂无指标</span>'}</div>
    </section>

    <section class="detail-section">
      <h4>检查点</h4>
      ${checkpoints}
    </section>
  `;
}

async function loadJobs() {
  const jobs = await fetchJson('/api/jobs');
  const signature = createJobsSignature(jobs);
  if (signature === state.jobsSignature) {
    state.jobs = jobs;
    updateOverview();
    return;
  }
  state.jobsSignature = signature;
  renderJobs(jobs);
}

async function loadJobDetail() {
  if (!state.selectedJobId) return;
  const job = await fetchJson(`/api/jobs/${state.selectedJobId}`);
  const detailSignature = createJobDetailSignature(job);
  if (detailSignature !== state.selectedJobSignature) {
    state.selectedJobSignature = detailSignature;
    renderJobDetail(job);
  }
  const log = await fetchJson(`/api/jobs/${state.selectedJobId}/logs?max_bytes=30000`);
  const viewer = qs('#logViewer');
  const nextLogText = log.text || '暂无日志';
  if (nextLogText !== state.lastLogText) {
    const nearBottom = viewer.scrollHeight - viewer.scrollTop - viewer.clientHeight < 64;
    state.lastLogText = nextLogText;
    viewer.textContent = nextLogText;
    if (nearBottom || nextLogText === '暂无日志') {
      viewer.scrollTop = viewer.scrollHeight;
    }
  }
}

async function stopJob() {
  if (!state.selectedJobId) return;
  await fetchJson(`/api/jobs/${state.selectedJobId}/stop`, { method: 'POST' });
  await loadJobs();
  await loadJobDetail();
}

async function inspectDataset() {
  const path = qs('input[name="dataset_path"]').value.trim();
  if (!path) {
    showInspect('⚠ 请先填写数据集路径', true);
    return;
  }
  const btn = qs('#inspectDatasetBtn');
  btn.disabled = true;
  btn.textContent = '检查中...';
  showInspect('正在验证路径...');
  try {
    const result = await fetchJson('/api/dataset/inspect', {
      method: 'POST',
      body: JSON.stringify({ path }),
    });
    const splitNames = Object.keys(result.splits || {});
    const splitInfo = splitNames.length ? splitNames.join('/') : '无标准 split';
    showInspect(`✅ ${result.dataset_type.toUpperCase()} 数据集  ·  ${splitInfo}`);
  } catch (error) {
    showInspect(`❌ ${error.message}`, true);
  } finally {
    btn.disabled = false;
    btn.textContent = '检查路径';
  }
}

function setScanStatus(message, cls = 'muted') {
  const el = qs('#scanStatus');
  el.textContent = message;
  el.className = 'scan-status ' + cls;
}

function renderDatasets(datasets) {
  const container = qs('#datasetGrid');
  state.datasets = datasets;
  updateOverview();

  if (!datasets.length) {
    container.innerHTML = '<div class="dataset-empty">未发现数据集。请确认挂载目录下存放有 YOLO 或 COCO 格式的数据集。</div>';
    return;
  }

  container.innerHTML = datasets.map((ds) => {
    const classes = (ds.classes || []).slice(0, 8);
    const moreClasses = (ds.classes || []).length > 8 ? `+${ds.classes.length - 8}` : '';
    const splitStats = ds.split_stats || {};
    const train = splitStats.train || {};
    const val = splitStats.val || {};
    const classSummary = (ds.classes || []).length ? `${ds.num_classes} 类` : '-';

    return `
      <div class="dataset-card">
        <div class="dataset-card-main">
          <div class="dataset-card-top">
            <div class="dataset-card-head">
              <h3>${escapeHtml(ds.name)}</h3>
            </div>
            <div class="dataset-card-actions">
              <span class="dataset-type-badge ${escapeHtml(ds.dataset_type)}">${escapeHtml(ds.dataset_type.toUpperCase())}</span>
              <button data-fill-ds="${escapeHtml(ds.path)}" class="primary-btn">使用此数据集</button>
            </div>
          </div>
          <div class="dataset-card-path">${escapeHtml(ds.path)}</div>
          <div class="dataset-card-stats compact">
            <span class="dataset-mini-stat inline"><span class="k">类别</span><span class="v">${escapeHtml(classSummary)}</span></span>
            <span class="dataset-mini-stat inline"><span class="k">图片</span><span class="v">${ds.total_images ? formatNumber(ds.total_images) : '-'}</span></span>
            <span class="dataset-mini-stat inline"><span class="k">标注</span><span class="v">${ds.total_labels ? formatNumber(ds.total_labels) : '-'}</span></span>
            <span class="dataset-mini-stat inline"><span class="k">训练</span><span class="v">${train.images ? formatNumber(train.images) : '-'}</span></span>
            <span class="dataset-mini-stat inline"><span class="k">验证</span><span class="v">${val.images ? formatNumber(val.images) : '-'}</span></span>
          </div>
          ${classes.length ? `
            <div class="dataset-classes-row">
              ${classes.map((c) => `<span class="dataset-class-chip">${escapeHtml(c)}</span>`).join('')}
              ${moreClasses ? `<span class="dataset-class-chip">${escapeHtml(moreClasses)}</span>` : ''}
            </div>
          ` : ''}
        </div>
      </div>
    `;
  }).join('');

  container.querySelectorAll('[data-fill-ds]').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      event.stopPropagation();
      qs('input[name="dataset_path"]').value = btn.dataset.fillDs;
      setScanStatus(`已选择数据集: ${btn.dataset.fillDs}`, 'success');
    });
  });

  container.querySelectorAll('[data-inspect-ds]').forEach((btn) => {
    btn.addEventListener('click', async (event) => {
      event.stopPropagation();
      qs('input[name="dataset_path"]').value = btn.dataset.inspectDs;
      try {
        const result = await fetchJson('/api/dataset/inspect', {
          method: 'POST',
          body: JSON.stringify({ path: btn.dataset.inspectDs }),
        });
        const splitsText = Object.entries(result.splits || {})
          .map(([name, value]) => `${name}: ${JSON.stringify(value)}`)
          .join('\n');
        showInspect(`${result.message}\n类型: ${result.dataset_type}\n${splitsText || '未发现标准 split'}`);
      } catch (error) {
        showInspect(error.message, true);
      }
    });
  });
}

async function scanDatasets() {
  if (state.scanning) return;
  state.scanning = true;
  setScanStatus('正在扫描挂载目录中的数据集...', 'scanning');
  qs('#scanDatasetsBtn').disabled = true;

  try {
    const data = await fetchJson('/api/datasets/scan');
    state.datasets = data.datasets || [];
    if (!data.scan_root_exists) {
      setScanStatus(`扫描根目录 ${data.scan_root} 不存在，请确认路径已正确挂载`, 'error');
    } else {
      setScanStatus(`扫描完成，共发现 ${data.total} 个数据集 (${data.scan_root})`, 'success');
    }
    renderDatasets(state.datasets);
  } catch (error) {
    setScanStatus(`扫描失败: ${error.message}`, 'error');
    qs('#datasetGrid').innerHTML = '';
  } finally {
    state.scanning = false;
    qs('#scanDatasetsBtn').disabled = false;
  }
}

async function loadPresets() {
  const data = await fetchJson('/api/presets');
  const preset = data.recommended;
  Object.entries(preset).forEach(([key, value]) => {
    const field = qs(`[name="${key}"]`);
    if (field) field.value = value;
  });
}

async function submitJob(event) {
  event.preventDefault();
  const form = event.target;
  const payload = formToPayload(form);
  try {
    const job = await fetchJson('/api/jobs', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    state.selectedJobId = job.job_id;
    showInspect(`任务已提交: ${job.job_name || job.job_id}`);
    await loadJobs();
    await loadJobDetail();
  } catch (error) {
    showInspect(error.message, true);
  }
}

function bindEvents() {
  qs('#jobForm').addEventListener('submit', submitJob);
  qs('#inspectDatasetBtn').addEventListener('click', inspectDataset);
  qs('#refreshJobsBtn').addEventListener('click', async () => {
    await loadJobs();
    await loadJobDetail();
  });
  qs('#stopJobBtn').addEventListener('click', stopJob);
  qs('#reloadDetailBtn').addEventListener('click', loadJobDetail);
  qs('#loadPresetBtn').addEventListener('click', loadPresets);
  qs('#scanDatasetsBtn').addEventListener('click', scanDatasets);
  qs('#scanRefreshBtn').addEventListener('click', scanDatasets);
  qs('#jobFilters').addEventListener('click', (event) => {
    const button = event.target.closest('.job-filter-btn');
    if (!button) return;
    state.jobFilter = button.dataset.filter;
    renderJobs(state.jobs);
  });
  qs('#toggleAutoRefreshBtn').addEventListener('click', () => {
    state.autoRefresh = !state.autoRefresh;
    qs('#toggleAutoRefreshBtn').textContent = state.autoRefresh ? '暂停自动刷新' : '恢复自动刷新';
    updateOverview();
  });
}

async function init() {
  bindEvents();
  await loadPresets();
  showInspect('数据集检查结果会显示在这里。');
  updateOverview();
  await loadJobs();
  scanDatasets();
  setInterval(async () => {
    if (!state.autoRefresh) return;
    await loadJobs();
    await loadJobDetail();
  }, 8000);
}

init().catch((error) => {
  showInspect(error.message, true);
});
