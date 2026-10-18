"""Background inspection/export of existing recordings from the unified launcher."""
import json
from pathlib import Path
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from qt_runtime import configure_qt

configure_qt()

ROOT = Path(__file__).resolve().parent


class VideoToolsWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('录像文件检查与 MP4 播放副本 | v1.0.0')
        self.resize(900, 620)
        self.process = None
        self.output = None
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel('选择已经停止并保存的单次录像文件夹。检查视频帧数与解码状态，或导出常见播放器可播放的 MP4 副本。')
        label.setWordWrap(True)
        layout.addWidget(label)
        row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit(str(ROOT/'sessions'))
        self.browse = QtWidgets.QPushButton('选择录像文件夹')
        self.browse.clicked.connect(self.choose_folder)
        row.addWidget(self.folder, 1)
        row.addWidget(self.browse)
        layout.addLayout(row)
        self.full = QtWidgets.QCheckBox('逐帧完整解码检查（耗时更长；默认检查首、中、尾帧）')
        layout.addWidget(self.full)
        row = QtWidgets.QHBoxLayout()
        self.inspect = QtWidgets.QPushButton('检查保存状态')
        self.export = QtWidgets.QPushButton('导出 MP4 播放副本')
        self.open_output = QtWidgets.QPushButton('打开输出目录')
        self.open_output.setEnabled(False)
        self.inspect.clicked.connect(lambda: self.start('inspect'))
        self.export.clicked.connect(lambda: self.start('export'))
        self.open_output.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.output)))
        for button in (self.inspect, self.export, self.open_output):
            row.addWidget(button)
        layout.addLayout(row)
        note = QtWidgets.QLabel('MP4 副本为有损播放文件；原始测量视频保留，导出不会覆盖原录像。麦克风音轨仍保存在原 session 中。')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

    def choose_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, '选择单次录像文件夹', self.folder.text())
        if path:
            self.folder.setText(path)

    def start(self, action):
        if self.process is not None:
            return
        process = QtCore.QProcess(self)
        self.process = process
        args = [str(ROOT/'video_tools.py'), action, self.folder.text().strip()]
        if action=='inspect' and self.full.isChecked():
            args.append('--full')
        process.readyReadStandardError.connect(lambda: self.log.appendPlainText(bytes(process.readAllStandardError()).decode('utf8', errors='replace').strip()))
        process.finished.connect(self.finished)
        process.errorOccurred.connect(self.failed)
        for control in (self.inspect, self.export, self.browse, self.folder, self.full):
            control.setEnabled(False)
        self.log.appendPlainText('开始检查…' if action=='inspect' else '开始导出播放副本，结束前请保持窗口打开…')
        process.start(sys.executable, args)

    def failed(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.log.appendPlainText('无法启动工具：'+self.process.errorString())
            self.reset()

    def reset(self):
        process, self.process = self.process, None
        if process:
            process.deleteLater()
        for control in (self.inspect, self.export, self.browse, self.folder, self.full):
            control.setEnabled(True)

    def finished(self, *_):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode('utf8', errors='replace')
        try:
            report = json.loads(text)
            self.log.appendPlainText('检查／导出通过。' if report.get('ok') else '检查／导出未通过。')
            if report.get('error'):
                self.log.appendPlainText(report['error'])
            for warning in report.get('warnings', []):
                self.log.appendPlainText(warning)
            for file in report.get('files', []):
                duration = file.get('duration_seconds')
                self.log.appendPlainText(f'{file["file"]} | {file["bytes"]/1e6:.1f} MB | '
                    f'{file.get("frames", "?")} 帧 | {duration if duration is not None else "?"} 秒 | '
                    f'{file.get("fourcc", "?")} | {file.get("error", "回读通过")}')
                if file.get('playback_note'):
                    self.log.appendPlainText(file['playback_note'])
            if report.get('note'):
                self.log.appendPlainText(report['note'])
            self.output = report.get('output') or self.folder.text().strip()
            self.open_output.setEnabled(True)
            self.log.appendPlainText('输出／原始目录：'+self.output)
            self.last_report = report
        except (ValueError, KeyError) as exc:
            self.log.appendPlainText('工具返回异常：'+str(exc)+' '+text[-1500:])
        self.reset()

    def closeEvent(self, event):
        if self.process is not None:
            self.log.appendPlainText('正在处理文件，请等待完成后关闭。')
            event.ignore()
        else:
            event.accept()
