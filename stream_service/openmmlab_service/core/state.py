import threading


class ThreadSafeDict:
    """线程安全的任务字典容器，内部使用可重入锁保护所有增删改查操作"""
    def __init__(self):
        self._data = {}
        self._lock = threading.RLock()

    def __getitem__(self, key):
        with self._lock:
            return self._data[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._data[key] = value

    def __contains__(self, key):
        with self._lock:
            return key in self._data

    def __delitem__(self, key):
        with self._lock:
            del self._data[key]

    def get(self, key, default=None):
        with self._lock:
            value = self._data.get(key, default)
            if isinstance(value, dict):
                return value.copy()
            return value

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def items(self):
        with self._lock:
            result = []
            for key, value in self._data.items():
                if isinstance(value, dict):
                    result.append((key, value.copy()))
                else:
                    result.append((key, value))
            return result

    def values(self):
        with self._lock:
            result = []
            for value in self._data.values():
                if isinstance(value, dict):
                    result.append(value.copy())
                else:
                    result.append(value)
            return result

    def pop(self, key, *args):
        with self._lock:
            return self._data.pop(key, *args)

    def get_field(self, key, field, default=None):
        with self._lock:
            value = self._data.get(key)
            if not isinstance(value, dict):
                return default
            return value.get(field, default)

    def set_field(self, key, field, value):
        with self._lock:
            target = self._data.get(key)
            if not isinstance(target, dict):
                return False
            target[field] = value
            return True

    def update_fields(self, key, **fields):
        with self._lock:
            target = self._data.get(key)
            if not isinstance(target, dict):
                return False
            target.update(fields)
            return True


# active_tasks 管理着当前系统正在跑的所有推流或推理进程
active_tasks = ThreadSafeDict()
