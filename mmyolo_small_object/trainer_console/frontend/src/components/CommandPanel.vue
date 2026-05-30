<script setup>
import { computed } from 'vue'
import { ElMessage } from 'element-plus'

const props = defineProps({
  modelValue: {
    type: Object,
    required: true,
  },
  presets: {
    type: Object,
    default: () => ({}),
  },
  loading: Boolean,
})

const emit = defineEmits(['update:modelValue', 'submit', 'inspect-dataset', 'apply-presets'])

const form = computed({
  get() {
    return props.modelValue
  },
  set(value) {
    emit('update:modelValue', value)
  },
})

function updateField(key, value) {
  form.value = { ...form.value, [key]: value }
}

function handleSubmit() {
  if (!form.value.dataset_path && !form.value.prepared_dataset_path) {
    ElMessage.error('dataset_path 或 prepared_dataset_path 至少要填一个')
    return
  }
  emit('submit')
}
</script>

<template>
  <el-card shadow="never">
    <template #header>
      <div class="panel-header">
        <div class="panel-header-left">
          <div class="panel-icon panel-icon--accent">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
          </div>
          <div>
            <h2>新建训练任务</h2>
            <p class="panel-desc">配置参数并提交训练</p>
          </div>
        </div>
        <el-button size="small" @click="emit('apply-presets')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
          加载推荐参数
        </el-button>
      </div>
    </template>

    <el-form label-position="top" @submit.prevent>
      <!-- 01 基础信息 -->
      <section class="form-block">
        <div class="form-block-head">
          <span class="block-num">01</span>
          <div>
            <h3>基础信息</h3>
            <p>任务名称、基线模型和运行设备</p>
          </div>
        </div>
        <div class="form-grid">
          <el-form-item label="任务名称">
            <el-input v-model="form.job_name" placeholder="例如 drone_rtmdet_v2"
              @update:model-value="val => updateField('job_name', val)" />
          </el-form-item>
          <el-form-item label="设备类型">
            <el-select v-model="form.device_type"
              @update:model-value="val => updateField('device_type', val)">
              <el-option label="CUDA (GPU)" value="cuda" />
            </el-select>
          </el-form-item>
          <el-form-item label="模型基线">
            <el-select v-model="form.arch"
              @update:model-value="val => updateField('arch', val)">
              <el-option label="RTMDet-Tiny" value="rtmdet" />
              <el-option label="YOLOv8-S" value="yolov8" />
            </el-select>
          </el-form-item>
          <el-form-item label="Launcher">
            <el-select v-model="form.launcher"
              @update:model-value="val => updateField('launcher', val)">
              <el-option label="none (单卡)" value="none" />
              <el-option label="pytorch (DDP)" value="pytorch" />
            </el-select>
          </el-form-item>
        </div>
      </section>

      <!-- 02 数据入口 -->
      <section class="form-block">
        <div class="form-block-head">
          <span class="block-num">02</span>
          <div>
            <h3>数据入口</h3>
            <p>源数据集和预处理后数据都可以指定</p>
          </div>
        </div>
        <div class="form-grid">
          <div class="span-2">
            <el-form-item label="源数据集路径">
              <div class="inline-field">
                <el-input v-model="form.dataset_path" placeholder="/home/dataList/your_dataset"
                  @update:model-value="val => updateField('dataset_path', val)" />
                <el-button @click="emit('inspect-dataset')">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 3px"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                  检查
                </el-button>
              </div>
            </el-form-item>
          </div>
          <div class="span-2">
            <el-form-item label="已准备好的数据集路径">
              <el-input v-model="form.prepared_dataset_path" placeholder="可留空，留空则自动准备数据"
                @update:model-value="val => updateField('prepared_dataset_path', val)" />
            </el-form-item>
          </div>
          <div class="span-2">
            <el-form-item label="预训练权重">
              <el-input v-model="form.load_from" placeholder="/path/to/pretrain.pth"
                @update:model-value="val => updateField('load_from', val)" />
            </el-form-item>
          </div>
        </div>
      </section>

      <!-- 03 训练参数 -->
      <section class="form-block">
        <div class="form-block-head">
          <span class="block-num">03</span>
          <div>
            <h3>训练参数</h3>
            <p>切片策略、batch、epoch 和学习率</p>
          </div>
        </div>
        <div class="form-grid">
          <el-form-item label="图像尺寸">
            <el-input-number v-model="form.img_scale" :min="1"
              @change="val => updateField('img_scale', val)" />
          </el-form-item>
          <el-form-item label="训练轮数">
            <el-input-number v-model="form.max_epochs" :min="1"
              @change="val => updateField('max_epochs', val)" />
          </el-form-item>
          <el-form-item label="保存间隔 (epoch)">
            <el-input-number v-model="form.save_epoch_intervals" :min="1"
              @change="val => updateField('save_epoch_intervals', val)" />
          </el-form-item>
          <el-form-item label="设备编号">
            <el-input v-model="form.device_visible_ids"
              @update:model-value="val => updateField('device_visible_ids', val)" />
          </el-form-item>
          <el-form-item label="切片尺寸">
            <el-input-number v-model="form.tile_size" :min="1"
              @change="val => updateField('tile_size', val)" />
          </el-form-item>
          <el-form-item label="切片重叠">
            <el-input-number v-model="form.tile_overlap" :min="0"
              @change="val => updateField('tile_overlap', val)" />
          </el-form-item>
          <el-form-item label="最小相交比例">
            <el-input-number v-model="form.min_intersection_ratio" :step="0.05" :min="0" :max="1"
              @change="val => updateField('min_intersection_ratio', val)" />
          </el-form-item>
          <el-form-item label="最小框边长">
            <el-input-number v-model="form.min_bbox_side" :step="0.5" :min="0"
              @change="val => updateField('min_bbox_side', val)" />
          </el-form-item>
          <el-form-item label="空切片上限">
            <el-input-number v-model="form.max_empty_tiles" :min="0"
              @change="val => updateField('max_empty_tiles', val)" />
          </el-form-item>
          <el-form-item label="训练 Batch">
            <el-input-number v-model="form.train_batch_size" :min="1"
              @change="val => updateField('train_batch_size', val)" />
          </el-form-item>
          <el-form-item label="验证 Batch">
            <el-input-number v-model="form.val_batch_size" :min="1"
              @change="val => updateField('val_batch_size', val)" />
          </el-form-item>
          <el-form-item label="训练 Workers">
            <el-input-number v-model="form.train_workers" :min="0"
              @change="val => updateField('train_workers', val)" />
          </el-form-item>
          <el-form-item label="验证 Workers">
            <el-input-number v-model="form.val_workers" :min="0"
              @change="val => updateField('val_workers', val)" />
          </el-form-item>
          <el-form-item label="基础学习率">
            <el-input-number v-model="form.base_lr" :step="0.0001" :min="0" :precision="6"
              @change="val => updateField('base_lr', val)" />
          </el-form-item>
          <div class="span-2">
            <el-form-item label="额外训练参数">
              <el-input v-model="form.extra_train_args" placeholder="例如 --amp --resume"
                @update:model-value="val => updateField('extra_train_args', val)" />
            </el-form-item>
          </div>
        </div>
      </section>

      <!-- toggle 开关 -->
      <section class="toggle-row">
        <div class="toggle-card">
          <el-switch v-model="form.prepare_dataset"
            @update:model-value="val => updateField('prepare_dataset', val)" />
          <div class="toggle-copy">
            <strong>自动准备数据集</strong>
            <small>自动执行切片、转换和 COCO 整理</small>
          </div>
        </div>
        <div class="toggle-card">
          <el-switch v-model="form.overwrite_prepared"
            @update:model-value="val => updateField('overwrite_prepared', val)" />
          <div class="toggle-copy">
            <strong>覆盖已有输出</strong>
            <small>允许复用目标目录并覆盖旧产物</small>
          </div>
        </div>
      </section>

      <!-- Submit -->
      <div class="submit-row">
        <el-button type="primary" class="submit-btn" size="large" :loading="loading" @click="handleSubmit">
          <svg v-if="!loading" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 6px"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          启动训练任务
        </el-button>
      </div>
    </el-form>
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

.panel-icon--accent {
  background: var(--accent-soft);
  color: var(--accent);
  box-shadow: 0 0 12px var(--accent-glow);
}

.panel-header h2 {
  font-size: 16px;
  font-weight: 700;
  color: var(--ink);
}

.panel-desc {
  font-size: 12px;
  color: var(--muted);
  margin-top: 1px;
}

/* form blocks */
.form-block {
  margin-top: 18px;
  padding: 18px;
  border-radius: var(--radius);
  background: var(--panel-soft);
  border: 1px solid var(--glass-border);
  transition: var(--transition);
}

.form-block:hover {
  border-color: var(--line-strong);
}

.form-block-head {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 16px;
}

.block-num {
  width: 28px;
  height: 28px;
  border-radius: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--accent-gradient-vivid);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  flex-shrink: 0;
  box-shadow: 0 2px 8px var(--accent-glow);
}

.form-block-head h3 {
  font-size: 14px;
  font-weight: 600;
  color: var(--ink);
}

.form-block-head p {
  font-size: 12px;
  color: var(--muted);
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

.span-2 { grid-column: span 2; }

.inline-field {
  display: flex;
  gap: 8px;
}
.inline-field :deep(.el-input) { flex: 1; }

/* toggles */
.toggle-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-top: 18px;
}

.toggle-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  border-radius: var(--radius);
  background: var(--panel-soft);
  border: 1px solid var(--glass-border);
  transition: var(--transition);
}

.toggle-card:hover {
  border-color: var(--line-strong);
}

.toggle-copy { display: flex; flex-direction: column; gap: 2px; }
.toggle-copy strong { font-size: 13px; color: var(--ink); }
.toggle-copy small { font-size: 11px; color: var(--muted); }

/* submit */
.submit-row { margin-top: 20px; }

.submit-btn {
  width: 100%;
  height: 46px !important;
  font-size: 15px !important;
  font-weight: 600 !important;
  border-radius: var(--radius) !important;
  background: var(--accent-gradient-vivid) !important;
  border: none !important;
  letter-spacing: .02em;
  position: relative;
  overflow: hidden;
}

.submit-btn::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.1) 50%, transparent 100%);
  background-size: 200% 100%;
  animation: shimmer 3s ease-in-out infinite;
  pointer-events: none;
}

.submit-btn:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 24px var(--accent-glow) !important;
}

@media (max-width: 760px) {
  .form-grid, .toggle-row { grid-template-columns: 1fr; }
  .span-2 { grid-column: span 1; }
}
</style>
