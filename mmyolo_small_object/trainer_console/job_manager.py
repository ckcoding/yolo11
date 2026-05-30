from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trainer_console.config import CONFIGS_ROOT, JOBS_ROOT, LOGS_ROOT, MMYOLO_ROOT, WORKDIR_ROOT, DATA_ROOT
from trainer_console.schemas import TrainingRequest


ANSI_RE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
EPOCH_RE = re.compile(r'Epoch\((train|val)\)\s*\[(\d+)\](?:\[(\d+)/(\d+)\])?')
KV_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_\/]*)\s*:\s*([^\s,]+)')


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub('', text)


@dataclass
class RunningProcess:
    popen: subprocess.Popen
    stop_requested: bool = False


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._processes: dict[str, RunningProcess] = {}
        self._lock = threading.RLock()
        self._load_jobs()

    def _load_jobs(self) -> None:
        for path in sorted(JOBS_ROOT.glob('*.json')):
            try:
                with path.open('r', encoding='utf-8') as f:
                    payload = json.load(f)
                if payload.get('status') == 'running':
                    payload['status'] = 'interrupted'
                    payload['latest_message'] = '服务重启后发现上次任务未正常结束'
                    payload['updated_at'] = now_iso()
                    self._jobs[payload['job_id']] = payload
                    self._save_job(payload)
                else:
                    self._jobs[payload['job_id']] = payload
            except Exception:
                continue

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda item: item['created_at'], reverse=True)
        for job in jobs:
            job['checkpoints'] = self._collect_checkpoints(job.get('work_dir', ''))
        return jobs

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            result = json.loads(json.dumps(job))
        result['checkpoints'] = self._collect_checkpoints(result.get('work_dir', ''))
        return result

    def create_job(self, request: TrainingRequest) -> dict:
        job_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
        job_name = request.job_name or f'{request.arch}_{job_id}'

        prepared_root = str((DATA_ROOT / job_id).resolve())
        config_path = str((CONFIGS_ROOT / f'{job_id}.py').resolve())
        work_dir = str((WORKDIR_ROOT / job_id).resolve())
        log_path = str((LOGS_ROOT / f'{job_id}.log').resolve())

        now = now_iso()
        payload = {
            'job_id': job_id,
            'job_name': job_name,
            'status': 'queued',
            'stage': 'queued',
            'arch': request.arch,
            'created_at': now,
            'updated_at': now,
            'dataset_path': request.dataset_path,
            'prepared_dataset_path': request.prepared_dataset_path or prepared_root,
            'prepared_dataset_source': request.prepared_dataset_path or '',
            'work_dir': work_dir,
            'config_path': config_path,
            'log_path': log_path,
            'metrics': {},
            'progress_epoch': 0,
            'progress_iter': 0,
            'progress_total_iter': 0,
            'latest_message': '等待启动',
            'exit_code': None,
            'command_history': [],
            'request': request.model_dump(),
            'stopped_by_user': False,
        }
        with self._lock:
            self._jobs[job_id] = payload
            self._save_job(payload)
            thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True, name=f'job-{job_id}')
            self._threads[job_id] = thread
            thread.start()
        return payload

    def stop_job(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            running = self._processes.get(job_id)
            job['stopped_by_user'] = True
            job['latest_message'] = '收到停止请求'
            job['updated_at'] = now_iso()
            self._save_job(job)

        if running:
            running.stop_requested = True
            proc = running.popen
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        return self.get_job(job_id)

    def read_log(self, job_id: str, max_bytes: int = 40000) -> dict | None:
        job = self.get_job(job_id)
        if not job:
            return None
        path = Path(job['log_path'])
        if not path.exists():
            return {'job_id': job_id, 'text': '', 'size': 0}
        data = path.read_bytes()
        return {
            'job_id': job_id,
            'text': data[-max_bytes:].decode('utf-8', errors='replace'),
            'size': len(data),
        }

    def _save_job(self, payload: dict) -> None:
        path = JOBS_ROOT / f"{payload['job_id']}.json"
        with path.open('w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _append_log(self, job: dict, text: str) -> None:
        path = Path(job['log_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f:
            f.write(text)

    def _update_job(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields)
            job['updated_at'] = now_iso()
            self._save_job(job)

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        request = TrainingRequest(**job['request'])
        Path(job['work_dir']).mkdir(parents=True, exist_ok=True)
        self._append_log(job, f'[{datetime.utcnow().isoformat()}] job={job_id} created\n')

        try:
            dataset_root = self._prepare_dataset(job_id, request)
            self._generate_config(job_id, request, dataset_root)
            self._train(job_id, request)

            final_job = self.get_job(job_id)
            if final_job and final_job.get('stopped_by_user'):
                self._update_job(job_id, status='stopped', stage='stopped', latest_message='任务已停止')
            else:
                self._update_job(job_id, status='completed', stage='completed', latest_message='训练完成', exit_code=0)
        except subprocess.CalledProcessError as exc:
            final_status = 'stopped' if self.get_job(job_id).get('stopped_by_user') else 'failed'
            self._update_job(
                job_id,
                status=final_status,
                stage='failed' if final_status == 'failed' else 'stopped',
                latest_message=f'命令失败，退出码 {exc.returncode}',
                exit_code=exc.returncode,
            )
        except Exception as exc:
            final_status = 'stopped' if self.get_job(job_id).get('stopped_by_user') else 'failed'
            self._append_log(self.get_job(job_id), f'\n[ERROR] {exc}\n')
            self._update_job(
                job_id,
                status=final_status,
                stage='failed' if final_status == 'failed' else 'stopped',
                latest_message=str(exc),
            )
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def _prepare_dataset(self, job_id: str, request: TrainingRequest) -> str:
        if not request.prepare_dataset and request.prepared_dataset_path:
            self._update_job(
                job_id,
                status='running',
                stage='prepare_skipped',
                prepared_dataset_path=request.prepared_dataset_path,
                latest_message='跳过数据准备，直接使用现成数据集',
            )
            return request.prepared_dataset_path

        self._update_job(job_id, status='running', stage='preparing', latest_message='开始切片和数据整理')
        dst_root = Path(self.get_job(job_id)['prepared_dataset_path'])
        cmd = [
            sys.executable,
            str(MMYOLO_ROOT / 'tools' / 'prepare_small_object_dataset.py'),
            '--src',
            request.dataset_path,
            '--dst',
            str(dst_root),
            '--tile-size',
            str(request.tile_size),
            '--tile-overlap',
            str(request.tile_overlap),
            '--min-intersection-ratio',
            str(request.min_intersection_ratio),
            '--min-bbox-side',
            str(request.min_bbox_side),
            '--max-empty-tiles',
            str(request.max_empty_tiles),
        ]
        if request.overwrite_prepared:
            cmd.append('--overwrite')
        self._run_command(job_id, cmd, stage='preparing')
        return str(dst_root)

    def _generate_config(self, job_id: str, request: TrainingRequest, dataset_root: str) -> None:
        self._update_job(job_id, stage='configuring', latest_message='生成训练配置')
        job = self.get_job(job_id)
        ann_path = Path(dataset_root) / 'annotations' / 'instances_train.json'
        cmd = [
            sys.executable,
            str(MMYOLO_ROOT / 'tools' / 'generate_config.py'),
            '--ann',
            str(ann_path),
            '--data-root',
            dataset_root,
            '--out',
            job['config_path'],
            '--arch',
            request.arch,
            '--img-scale',
            str(request.img_scale),
            '--max-epochs',
            str(request.max_epochs),
            '--save-epoch-intervals',
            str(request.save_epoch_intervals),
            '--train-batch-size',
            str(request.train_batch_size),
            '--val-batch-size',
            str(request.val_batch_size),
            '--train-workers',
            str(request.train_workers),
            '--val-workers',
            str(request.val_workers),
            '--base-lr',
            str(request.base_lr),
        ]
        if request.load_from:
            cmd.extend(['--load-from', request.load_from])
        self._run_command(job_id, cmd, stage='configuring')

    def _train(self, job_id: str, request: TrainingRequest) -> None:
        self._update_job(job_id, stage='training', latest_message='开始训练')
        job = self.get_job(job_id)
        cmd = [
            sys.executable,
            '-m',
            'mim',
            'train',
            'mmyolo',
            job['config_path'],
            '--work-dir',
            job['work_dir'],
        ]
        if request.launcher != 'none':
            cmd.extend(['--launcher', request.launcher])
        extra = shlex.split(request.extra_train_args) if request.extra_train_args else []
        cmd.extend(extra)

        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        if request.device_visible_ids:
            env['CUDA_VISIBLE_DEVICES'] = request.device_visible_ids
        self._run_command(job_id, cmd, stage='training', env=env)

    def _run_command(self, job_id: str, cmd: list[str], stage: str, env: dict | None = None) -> None:
        job = self.get_job(job_id)
        history = job.get('command_history', [])
        history.append(cmd)
        self._update_job(job_id, command_history=history)
        self._append_log(job, f'\n$ {" ".join(shlex.quote(part) for part in cmd)}\n')

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(MMYOLO_ROOT),
        )
        with self._lock:
            self._processes[job_id] = RunningProcess(proc)

        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = strip_ansi(raw_line)
            self._append_log(job, line)
            self._parse_progress(job_id, line, stage)

        return_code = proc.wait()
        stopped = self.get_job(job_id).get('stopped_by_user')
        if return_code != 0 and not stopped:
            raise subprocess.CalledProcessError(return_code, cmd)

    def _parse_progress(self, job_id: str, line: str, stage: str) -> None:
        line = line.rstrip()
        if not line:
            return

        updates = {'latest_message': line, 'stage': stage}
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            updates['progress_epoch'] = int(epoch_match.group(2))
            if epoch_match.group(3):
                updates['progress_iter'] = int(epoch_match.group(3))
            if epoch_match.group(4):
                updates['progress_total_iter'] = int(epoch_match.group(4))

        metrics = {}
        for key, value in KV_RE.findall(line):
            if key in {'lr', 'loss', 'loss_cls', 'loss_bbox', 'loss_obj', 'loss_dfl', 'acc'}:
                metrics[key] = value

        if metrics:
            current = self.get_job(job_id).get('metrics', {})
            current.update(metrics)
            updates['metrics'] = current

        self._update_job(job_id, **updates)

    def _collect_checkpoints(self, work_dir: str) -> list[str]:
        path = Path(work_dir)
        if not work_dir or not path.exists():
            return []
        files = sorted(path.glob('*.pth'), key=lambda item: item.stat().st_mtime, reverse=True)
        return [str(item.resolve()) for item in files[:20]]


job_manager = JobManager()
