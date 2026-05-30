#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys


def main() -> int:
    print(f'Python: {sys.version.split()[0]}')

    if (3, 8) <= sys.version_info[:2] <= (3, 11):
        print('[OK] Python 版本在 MMYOLO 常用兼容区间内。')
    else:
        print('[WARN] Python 版本不在推荐区间内，建议使用 3.10 或 3.11。')

    modules = ['mmyolo', 'mmengine', 'mmcv', 'mmdet', 'pycocotools', 'PIL', 'yaml']
    failed = False

    for name in modules:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, '__version__', 'unknown')
            print(f'[OK] {name}: {version}')
        except Exception as exc:
            failed = True
            print(f'[MISS] {name}: {exc}')

    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
