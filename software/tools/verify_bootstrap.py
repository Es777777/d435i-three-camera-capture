"""Exercise a genuine first launch without a Python on PATH, using offline inputs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from build_release import ROOT, prepare_offline, source_files


def main():
    verification = ROOT/'.runtime'/'verification'
    verification.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix='首次试用 空格_', dir=verification)).resolve()
    assert workspace.parent == verification.resolve()
    print(f'Verification workspace: {workspace}', flush=True)
    summary = {}
    for source in source_files():
        target = workspace/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for source in prepare_offline():
        target = workspace/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    powershell = str(Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe')
    environment = dict(os.environ)
    environment.update(PATH=str(Path(os.environ['SystemRoot'])/'System32'),
                       PYTHONHOME='Z:/intentionally-missing-python',
                       PYTHONPATH='Z:/intentionally-missing-modules',
                       QT_PLUGIN_PATH='Z:/intentionally-missing-plugins',
                       QT_QPA_PLATFORM_PLUGIN_PATH='Z:/intentionally-missing-plugins')

    def launch(extra=(), folder=workspace, expected=0):
        started = time.monotonic()
        result = subprocess.run([powershell, '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                 '-File', str(folder/'tools/bootstrap.ps1'), '-SetupOnly', '-Offline', *extra],
                                env=environment, capture_output=True, text=True, encoding='utf8',
                                errors='replace', timeout=600)
        print(result.stdout, flush=True)
        assert result.returncode == expected, result.stdout+'\n'+result.stderr
        return round(time.monotonic()-started, 2)

    # Only a fresh extraction and the packaged wheels are present in this fixture.
    summary['offline_first_launch_seconds'] = launch()
    stamp = workspace/'.venv/.requirements.sha256'
    mtime = stamp.stat().st_mtime_ns
    sentinel = workspace/'sessions/keep_existing_data.txt'
    sentinel.write_text('existing capture data', encoding='utf8')
    summary['repeat_launch_seconds'] = launch()
    assert stamp.stat().st_mtime_ns == mtime, 'Repeated launch reinstalled the environment'
    with (workspace/'.runtime/setup.lock').open('rb'):
        launch(expected=1)
    summary['concurrent_setup_rejected'] = True
    with (workspace/'.runtime/in_use.lock').open('rb'):
        launch(['-Repair'], expected=1)
    summary['repair_while_in_use_rejected'] = True
    damaged = workspace/'damaged archive'
    shutil.copytree(workspace/'tools', damaged/'tools')
    (damaged/'offline').mkdir()
    manifest = json.loads((workspace/'tools/runtime.json').read_text(encoding='utf8'))
    (damaged/'offline'/manifest['filename']).write_bytes(b'damaged archive')
    launch(folder=damaged, expected=1)
    summary['damaged_runtime_rejected'] = True
    assert sentinel.read_text(encoding='utf8') == 'existing capture data'
    summary.update(existing_data_preserved=True, no_system_python=True,
                   unicode_and_space_path=True, offline_only=True)
    log_root = ROOT/'logs'
    log_root.mkdir(exist_ok=True)
    (log_root/'bootstrap_verification.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    shutil.copyfile(workspace/'logs/bootstrap.log', log_root/'bootstrap_verification.log')
    print(json.dumps(summary, indent=2), flush=True)
    # Verify the absolute recursive-deletion target before removing our own fixture.
    if (workspace.parent == verification.resolve() and workspace.name.startswith('首次试用 空格_')
            and not workspace.is_symlink()):
        print(f'Removing verified temporary fixture: {workspace}', flush=True)
        shutil.rmtree(workspace)
    else:
        raise RuntimeError(f'Refusing cleanup outside verification directory: {workspace}')


if __name__ == '__main__':
    main()
