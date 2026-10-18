"""One desktop entry for capture, calibration and offline processing."""
import argparse
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys

os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

from PyQt5 import QtCore, QtGui, QtNetwork, QtWidgets

from qt_runtime import configure_qt
configure_qt()

ROOT = Path(__file__).resolve().parent
MODES = {
    'capture': ('采集与本机标定', '选择本机角色，预览三路画面，标定并录制音视频。'),
    'settlement': ('三机标定复核（可选）', '可选复核三份本机标定；录像无需等待此步骤。'),
    'reconstruction': ('重建 · 合成 · 回放', '处理本机 session，合成三机结果，查看三维轨迹。'),
    'legacy': ('旧版人工标点', '打开原有的人工指定同名 Mark 后处理流程。'),
    'videos': ('录像检查／转 MP4', '检查已有视频保存状态，导出通用 MP4 播放副本。'),
}


def default_capture_dir():
    try:
        import yaml
        cfg = yaml.safe_load((ROOT/'capture_config.yaml').read_text(encoding='utf8'))
        path = Path(cfg.get('output_root') or 'sessions')
        return path if path.is_absolute() else ROOT/path
    except Exception:
        return ROOT/'sessions'


def saved_role():
    role_file = Path(os.environ.get('LOCALAPPDATA', str(Path.home()/'.local')))/'D435I_ThreeHost'/'camera_role.json'
    try:
        return json.loads(role_file.read_text(encoding='utf8')).get('sync_role') or '未设置'
    except (OSError, ValueError):
        return '未设置（进入采集窗口选择）'


def create_window(mode):
    if mode == 'home':
        return LaunchCenter()
    if mode == 'capture':
        import yaml
        from capture_app import Main
        config = yaml.safe_load((ROOT/'capture_config.yaml').read_text(encoding='utf8'))
        if not isinstance(config, dict):
            raise ValueError('capture_config.yaml 必须为配置字典')
        return Main(config)
    if mode == 'settlement':
        from calibration_settlement_gui import SettlementWindow
        return SettlementWindow()
    if mode == 'reconstruction':
        from reconstruction_gui import ReconstructionWindow
        return ReconstructionWindow()
    if mode == 'legacy':
        from postprocess_gui import Window
        return Window()
    if mode == 'videos':
        from video_tools_gui import VideoToolsWindow
        return VideoToolsWindow()
    raise ValueError(mode)


def instance_name(mode):
    key = hashlib.sha256(str(ROOT).casefold().encode('utf8')).hexdigest()[:20]
    return f'D435i_{key}_{mode}'


def notify_existing(mode):
    socket = QtNetwork.QLocalSocket()
    socket.connectToServer(instance_name(mode))
    if not socket.waitForConnected(350):
        return False
    socket.write(b'show')
    socket.waitForBytesWritten(350)
    socket.disconnectFromServer()
    return True


def serve_window(window, mode):
    name = instance_name(mode)
    QtNetwork.QLocalServer.removeServer(name)
    server = QtNetwork.QLocalServer(window)
    if not server.listen(name):
        raise RuntimeError('无法建立窗口通信：'+server.errorString())

    def activate():
        while server.hasPendingConnections():
            connection = server.nextPendingConnection()
            connection.readyRead.connect(connection.readAll)
            connection.disconnected.connect(connection.deleteLater)
        window.showNormal()
        window.raise_()
        window.activateWindow()

    server.newConnection.connect(activate)
    window._instance_server = server


class LaunchCenter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.processes = {}
        self.report = None
        self._closing = False
        self._diagnostic_timedout = False
        self.setWindowTitle('D435i 数采工作台 | v1.0.0')
        self.resize(1000, 790)
        self.setMinimumSize(870, 700)
        root = QtWidgets.QWidget()
        root.setObjectName('home')
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(28, 23, 28, 23)
        layout.setSpacing(14)
        title = QtWidgets.QLabel('D435i 数采工作台')
        title.setStyleSheet('font-size:27px;font-weight:600;color:#17384d')
        layout.addWidget(title)
        subtitle = QtWidgets.QLabel('三台电脑使用同一程序，在采集窗口选择 Master / Slave 1 / Slave 2。')
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        workflow = QtWidgets.QLabel('本机标定  →  直接录像并保存位姿戳  →  本机重建  →  汇总时三机结算与合成回放')
        workflow.setWordWrap(True)
        workflow.setStyleSheet('background:#e8f1f6;padding:13px;border-radius:8px;color:#25536c')
        layout.addWidget(workflow)
        cards = QtWidgets.QHBoxLayout()
        self.buttons = {}
        for i, mode in enumerate(('capture', 'settlement', 'reconstruction'), 1):
            title_text, detail = MODES[mode]
            card = QtWidgets.QGroupBox(f'0{i}')
            box = QtWidgets.QVBoxLayout(card)
            button = QtWidgets.QPushButton(title_text)
            button.setMinimumHeight(46)
            button.clicked.connect(lambda checked=False, mode=mode: self.open_tool(mode))
            self.buttons[mode] = button
            box.addWidget(button)
            description = QtWidgets.QLabel(detail)
            description.setWordWrap(True)
            box.addWidget(description)
            cards.addWidget(card, 1)
        layout.addLayout(cards)
        hardware = QtWidgets.QHBoxLayout()
        self.environment_status = QtWidgets.QLabel('正在检查环境…')
        self.environment_status.setWordWrap(True)
        hardware.addWidget(self.environment_status, 1)
        self.check_button = QtWidgets.QPushButton('重新检查环境与设备')
        self.check_button.clicked.connect(self.check_environment)
        hardware.addWidget(self.check_button)
        layout.addLayout(hardware)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumHeight(140)
        layout.addWidget(self.details, 1)
        self.location = QtWidgets.QLabel()
        self.refresh_location()
        self.location.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.location.setWordWrap(True)
        layout.addWidget(self.location)
        links = QtWidgets.QHBoxLayout()
        for label, callback in (
            ('打开采集目录', lambda: self.open_folder('sessions')),
            ('打开日志目录', lambda: self.open_folder('logs')),
            ('使用说明', self.show_help),
            ('录像检查／转 MP4', lambda: self.open_tool('videos')),
            ('旧版人工标点', lambda: self.open_tool('legacy')),
        ):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            links.addWidget(button)
        layout.addLayout(links)
        self.setStyleSheet('''
            QWidget#home {background:#f5f8fa;}
            QWidget {font-family:"Microsoft YaHei UI";font-size:13px;color:#253e50;}
            QGroupBox {border:1px solid #d2dfe7;border-radius:9px;margin-top:10px;padding:12px;background:white;}
            QGroupBox::title {subcontrol-origin:margin;left:15px;color:#587e94;}
            QPushButton {background:white;border:1px solid #becfdc;padding:9px 12px;border-radius:6px;}
            QPushButton:hover {background:#e4f0f7;border-color:#3685b2;}
            QPushButton:disabled {color:#8093a0;background:#edf1f3;}
            QPlainTextEdit {background:white;border:1px solid #d2dfe7;border-radius:8px;padding:8px;}
        ''')
        self.diagnostic = QtCore.QProcess(self)
        self.diagnostic.setWorkingDirectory(str(ROOT))
        self.diagnostic.finished.connect(self.diagnostic_finished)
        self.diagnostic.errorOccurred.connect(self.diagnostic_error)
        self.diagnostic_timer = QtCore.QTimer(self)
        self.diagnostic_timer.setSingleShot(True)
        self.diagnostic_timer.timeout.connect(self.diagnostic_timeout)
        QtCore.QTimer.singleShot(0, self.check_environment)

    def check_environment(self):
        if self.diagnostic.state() != QtCore.QProcess.NotRunning:
            return
        self.check_button.setEnabled(False)
        self._diagnostic_timedout = False
        self.environment_status.setText('正在检查依赖、录像编码、相机和麦克风…')
        self.diagnostic.start(sys.executable, [str(ROOT/'tools'/'environment_check.py'), '--devices'])
        self.diagnostic_timer.start(30000)

    def diagnostic_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.diagnostic_timer.stop()
            self.check_button.setEnabled(True)
            self.environment_status.setText('环境检查无法启动：'+self.diagnostic.errorString())

    def diagnostic_timeout(self):
        self._diagnostic_timedout = True
        self.diagnostic.kill()
        self.environment_status.setText('设备检查超时，请检查 USB 连接后重新检查。')

    def diagnostic_finished(self, code, status):
        self.diagnostic_timer.stop()
        self.check_button.setEnabled(True)
        if self._closing or self._diagnostic_timedout:
            return
        output = bytes(self.diagnostic.readAllStandardOutput()).decode('utf8', errors='replace')
        errors = bytes(self.diagnostic.readAllStandardError()).decode('utf8', errors='replace')
        try:
            report = json.loads(output)
            self.report = report
            (ROOT/'logs').mkdir(exist_ok=True)
            (ROOT/'logs'/'environment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
            label = '运行环境已就绪' if report['ok'] else '运行环境需要修复，请查看下方详情'
            cameras = report.get('devices', {}).get('cameras', [])
            label += f'  ·  相机 {len(cameras)} 台'
            self.environment_status.setText(label)
            lines = [f'Python {report.get("python", "?")}  |  {report.get("executable", "")}', '']
            for row in report['checks']:
                state = '通过' if row['ok'] else ('失败' if row['required'] else '提示')
                lines.append(f'[{state}] {row["name"]}：{row["detail"]}')
            if not report['ok']:
                lines.append('\n关闭所有窗口后运行 start.bat -Repair，可重新安装依赖。')
            self.details.setPlainText('\n'.join(lines))
        except Exception:
            logging.exception('environment report failed')
            self.environment_status.setText('检查未完成，请重新检查或打开日志目录。')
            self.details.setPlainText(output+'\n'+errors)

    def open_tool(self, mode):
        process = self.processes.get(mode)
        if process is not None and process.state() != QtCore.QProcess.NotRunning:
            notify_existing(mode)
            return process
        running = {key for key, child in self.processes.items()
                   if child.state() != QtCore.QProcess.NotRunning}
        if ((mode in ('reconstruction', 'legacy') and 'capture' in running)
                or (mode == 'capture' and running.intersection(('reconstruction', 'legacy')))):
            self.statusBar().showMessage('请先关闭采集或离线处理窗口，再打开此功能，避免录像与计算争用资源。')
            return None
        process = QtCore.QProcess(self)
        process.setWorkingDirectory(str(ROOT))
        (ROOT/'logs').mkdir(exist_ok=True)
        process.setStandardOutputFile(str(ROOT/'logs'/f'{mode}_console.log'), QtCore.QIODevice.Append)
        process.setStandardErrorFile(str(ROOT/'logs'/f'{mode}_console.log'), QtCore.QIODevice.Append)
        process.errorOccurred.connect(lambda error, mode=mode: self.tool_error(mode, error))
        process.finished.connect(lambda code, status, mode=mode: self.tool_finished(mode, code, status))
        self.processes[mode] = process
        process.start(sys.executable, [str(ROOT/'launcher.py'), '--mode', mode])
        self.statusBar().showMessage('正在打开'+MODES[mode][0]+'…')
        return process

    def tool_error(self, mode, error):
        if error == QtCore.QProcess.FailedToStart:
            self.statusBar().showMessage('无法打开'+MODES[mode][0]+'，请查看日志。')

    def tool_finished(self, mode, code, status):
        self.refresh_location()
        text = MODES[mode][0]+('已关闭' if code == 0 else f'异常退出（{code}），请查看 logs/{mode}.log')
        self.statusBar().showMessage(text)

    def open_folder(self, folder):
        path = default_capture_dir() if folder == 'sessions' else ROOT/folder
        path.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def refresh_location(self):
        self.location.setText(f'本机角色：{saved_role()}\n默认采集目录：{default_capture_dir()}')

    def show_help(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('数采软件使用说明')
        dialog.resize(830, 650)
        layout = QtWidgets.QVBoxLayout(dialog)
        text = QtWidgets.QTextBrowser()
        text.setPlainText((ROOT/'README.md').read_text(encoding='utf8'))
        layout.addWidget(text)
        dialog.exec_()

    def closeEvent(self, event):
        if any(p.state() != QtCore.QProcess.NotRunning for p in self.processes.values()):
            self.statusBar().showMessage('请先关闭已打开的采集或处理窗口，再退出工作台。')
            event.ignore()
            return
        if self.diagnostic.state() != QtCore.QProcess.NotRunning:
            self._closing = True
            self.diagnostic.kill()
            self.diagnostic.waitForFinished(1500)
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['home', *MODES], default='home')
    args = parser.parse_args()
    (ROOT/'logs').mkdir(exist_ok=True)
    handler = RotatingFileHandler(ROOT/'logs'/f'{args.mode}.log', maxBytes=5_000_000, backupCount=2, encoding='utf8')
    logging.basicConfig(handlers=[handler], level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    app = QtWidgets.QApplication([sys.argv[0]])
    app.setApplicationName('D435i 数采工作台')
    if notify_existing(args.mode):
        return 0
    # Keep a shared read handle; bootstrap takes an exclusive handle before installing.
    (ROOT/'.runtime').mkdir(exist_ok=True)
    usage_path = ROOT/'.runtime'/'in_use.lock'
    if not usage_path.exists():
        usage_path.touch()
    app._environment_handle = usage_path.open('rb')

    def report_exception(exc_type, value, tb):
        logging.error('Uncaught exception', exc_info=(exc_type, value, tb))
        QtWidgets.QMessageBox.critical(None, '运行错误', f'{value}\n\n详细信息：{ROOT / "logs"}')

    sys.excepthook = report_exception
    try:
        window = create_window(args.mode)
        serve_window(window, args.mode)
    except Exception as exc:
        logging.exception('could not open tool')
        QtWidgets.QMessageBox.critical(None, '窗口未能打开', f'{exc}\n\n关闭所有窗口后运行 start.bat -Repair，或查看 logs 目录。')
        return 1
    window.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
