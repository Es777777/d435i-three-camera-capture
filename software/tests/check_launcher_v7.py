import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt5 import QtCore, QtWidgets

import capture_app
from launcher import LaunchCenter, create_window, notify_existing, serve_window

ROOT = Path(__file__).resolve().parents[1]


def wait_until(app, predicate, timeout=30):
    deadline = time.monotonic()+timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    app.processEvents()
    assert predicate(), 'Timed out'


def run():
    app = QtWidgets.QApplication([])
    result = subprocess.run([sys.executable, str(ROOT/'tools/environment_check.py')],
                            capture_output=True, text=True, encoding='utf8', timeout=30)
    report = json.loads(result.stdout)
    assert result.returncode == 0 and report['ok'], report
    assert any(row['name'] == '无损录像 / 目录写入' and row['ok'] for row in report['checks'])
    home = LaunchCenter()
    home.show()
    serve_window(home, 'test_home')
    assert notify_existing('test_home')
    wait_until(app, lambda: home.report is not None)
    assert home.report['ok'], home.details.toPlainText()
    assert set(home.buttons) == {'capture', 'settlement', 'reconstruction'}
    process = home.open_tool('settlement')
    wait_until(app, lambda: notify_existing('settlement'))
    assert home.open_tool('settlement') is process
    assert len(home.processes) == 1
    assert not home.close(), 'Closing the home must not kill a running tool'
    # This offscreen child has no camera/recording; Windows WM_CLOSE is unavailable.
    process.kill()
    assert process.waitForFinished(10000)
    app.processEvents()
    duplicate_target = QtCore.QProcess()
    duplicate_target.start(sys.executable, [str(ROOT/'launcher.py'), '--mode', 'home'])
    try:
        wait_until(app, lambda: notify_existing('home'))
        repeated = subprocess.run([sys.executable, str(ROOT/'launcher.py'), '--mode', 'home'], timeout=15)
        assert repeated.returncode == 0
        assert duplicate_target.state() == QtCore.QProcess.Running
    finally:
        duplicate_target.kill()
        assert duplicate_target.waitForFinished(10000)
    with tempfile.TemporaryDirectory() as folder:
        original = capture_app.ROLE_FILE
        capture_app.ROLE_FILE = Path(folder)/'camera_role.json'
        try:
            capture = create_window('capture')
            assert not capture.start_button.isEnabled()
            custom = capture_app.Main({'output_root': 'custom_capture'})
            assert Path(custom.path.text()) == ROOT/'custom_capture'
            custom.close()
            capture.close()
        finally:
            capture_app.ROLE_FILE = original
    for mode in ('settlement', 'reconstruction', 'legacy'):
        window = create_window(mode)
        window.show()
        app.processEvents()
        assert window.isVisible()
        window.close()
    (ROOT/'tests/artifacts').mkdir(exist_ok=True)
    home.grab().save(str(ROOT/'tests/artifacts/launcher_v7.png'))
    assert home.close()
    print('PASS v7: private environment, lossless codec, diagnostics, all tool windows, '
          'single instance, child lifetime, configured output directory')


if __name__ == '__main__':
    run()
