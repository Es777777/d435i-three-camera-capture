"""Run each Qt check in a separate process with a consistent import path."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    environment = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8='1', QT_QPA_PLATFORM='offscreen')
    tests = sorted((ROOT/'tests').glob('check_*.py'))
    if not tests:
        raise RuntimeError('No checks found')
    failures = []
    for path in tests:
        print(f'Running {path.name}', flush=True)
        result = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=environment, timeout=180)
        if result.returncode:
            failures.append(path.name)
    print(f'{len(tests)-len(failures)}/{len(tests)} checks passed', flush=True)
    if failures:
        print('Failed: '+', '.join(failures))
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
