import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets

from qt_runtime import configure_qt
configure_qt()

from calibration_settlement import calibration_digest, verify_settlement_record, validate_local_calibration
from capture_runtime import CaptureWorker, SYNC_ROLE_MODES
from calibration_guidance import bright_markers

ROOT = Path(__file__).resolve().parent
SYNC_ROLES = {'master': ('主机 Master', SYNC_ROLE_MODES['master']),
              'slave1': ('从机 Slave 1', SYNC_ROLE_MODES['slave1']),
              'slave2': ('从机 Slave 2', SYNC_ROLE_MODES['slave2'])}
ROLE_FILE = Path(os.environ.get('LOCALAPPDATA', str(Path.home()/'.local')))/'D435I_ThreeHost'/'camera_role.json'


def image_for_qt(image, bgr=False):
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if bgr else image
    rgb = np.ascontiguousarray(rgb)
    return QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                        QtGui.QImage.Format_RGB888).copy()


class ImageView(QtWidgets.QWidget):
    clicked = QtCore.pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.image = None
        self.marks = {}
        self.candidates = []
        self.confirmed = set()
        self.active_mark = None
        self.predictions = {}
        self.error_marks = set()
        self.reference_only = False
        self.magnifier = False
        self.zoomable = False
        self.view_region = None
        self.hover_point = None
        self.setMouseTracking(True)
        self.setMinimumSize(300, 220)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def image_rect(self):
        if self.image is None:
            return QtCore.QRectF()
        size = self.source_rect().size().toSize().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
        return QtCore.QRectF((self.width()-size.width())/2, (self.height()-size.height())/2,
                             size.width(), size.height())

    def source_rect(self):
        return self.view_region if self.view_region is not None else QtCore.QRectF(0, 0, self.image.width(), self.image.height())

    def image_position(self, pos):
        rect = self.image_rect()
        if self.image is None or not rect.contains(pos):
            return None
        source = self.source_rect()
        return (source.x()+(pos.x()-rect.x())/rect.width()*source.width(),
                source.y()+(pos.y()-rect.y())/rect.height()*source.height())

    def reset_view(self):
        self.view_region = None
        self.update()

    def wheelEvent(self, event):
        point = self.image_position(event.pos()) if self.image is not None else None
        if not self.zoomable or point is None:
            event.ignore()
            return
        source = self.source_rect()
        width = min(self.image.width(), max(80., source.width()/(1.25 if event.angleDelta().y() > 0 else .8)))
        if width >= self.image.width():
            self.reset_view()
        else:
            height = width*self.image.height()/self.image.width()
            x = point[0]-(point[0]-source.x())/source.width()*width
            y = point[1]-(point[1]-source.y())/source.height()*height
            self.view_region = QtCore.QRectF(max(0, min(self.image.width()-width, x)),
                                             max(0, min(self.image.height()-height, y)), width, height)
            self.update()
        event.accept()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor('#1e2226'))
        if self.image is None:
            return
        rect = self.image_rect()
        source = self.source_rect()
        p.drawImage(rect, self.image, source)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        scale = rect.width()/source.width()
        def position(x, y):
            return QtCore.QPointF(rect.x()+(x-source.x())*scale, rect.y()+(y-source.y())*scale)
        annotations = []
        p.setClipRect(rect)
        for x, y, radius in self.candidates:
            p.setPen(QtGui.QPen(QtGui.QColor('#53d8f5'), 2))
            p.setBrush(QtCore.Qt.NoBrush)
            r = max(7., radius*scale+4)
            p.drawEllipse(position(x, y), r, r)
        for name, (x, y) in self.predictions.items():
            pos = position(x, y)
            color = '#ff9670' if self.reference_only else '#ffe15b'
            p.setPen(QtGui.QPen(QtGui.QColor(color), 2, QtCore.Qt.DashLine))
            p.drawEllipse(pos, 13, 13)
            if name in self.marks:
                p.drawLine(position(*self.marks[name]), pos)
            if rect.contains(pos):
                annotations.append((pos, name+(' 参考' if self.reference_only else ' 预计'), color))
        for name, (x, y) in self.marks.items():
            pos = position(x, y)
            color = '#ff665e' if name in self.error_marks else '#ffe15b' if name == self.active_mark else '#48ec9a' if name in self.confirmed else '#f1ad35'
            p.setPen(QtGui.QPen(QtGui.QColor('#101b23'), 6))
            p.drawEllipse(pos, 9, 9)
            p.setPen(QtGui.QPen(QtGui.QColor(color), 3))
            p.drawEllipse(pos, 9, 9)
            p.drawLine(pos+QtCore.QPointF(-4, 0), pos+QtCore.QPointF(4, 0))
            p.drawLine(pos+QtCore.QPointF(0, -4), pos+QtCore.QPointF(0, 4))
            if rect.contains(pos):
                annotations.append((pos, name, color))
        obstacles = [QtCore.QRectF(position(x, y)-QtCore.QPointF(11, 11), QtCore.QSizeF(22, 22))
                     for x, y, radius in self.candidates if rect.contains(position(x, y))]
        occupied = []
        for pos, text, color in annotations:
            width = 86 if '预计' in text or '参考' in text else 44
            choices = [QtCore.QRectF(pos.x()+dx, pos.y()+dy, width, 21)
                       for dx, dy in ((14, -25), (-width-14, -25), (14, 5), (-width-14, 5),
                                      (-width/2, -46), (-width/2, 24), (14, -68), (-width-14, 46))]
            choices = [candidate for candidate in choices if rect.contains(candidate)]
            if not choices:
                continue
            def penalty(candidate):
                return sum(candidate.intersected(other).width()*candidate.intersected(other).height()
                           for other in obstacles+occupied if candidate.intersects(other))
            label = min(choices, key=penalty)
            occupied.append(label.adjusted(-3, -3, 3, 3))
            p.setPen(QtGui.QPen(QtGui.QColor(color), 1))
            p.drawLine(pos, label.center())
            p.fillRect(label, QtGui.QColor('#15242e'))
            p.drawText(label, QtCore.Qt.AlignCenter, text)
        p.setClipping(False)
        if self.magnifier and self.hover_point is not None:
            x, y = self.hover_point
            source = QtCore.QRectF(x-22, y-22, 44, 44)
            lens = QtCore.QRectF(self.width()-188, self.height()-210, 176, 176)
            p.fillRect(lens, QtGui.QColor('#131e26'))
            p.drawImage(lens, self.image, source)
            p.setPen(QtGui.QPen(QtGui.QColor('#ffe15b'), 2))
            p.drawRect(lens)
            center = lens.center()
            p.drawLine(center+QtCore.QPointF(-12, 0), center+QtCore.QPointF(12, 0))
            p.drawLine(center+QtCore.QPointF(0, -12), center+QtCore.QPointF(0, 12))
            p.drawText(QtCore.QRectF(lens.x(), lens.bottom()+3, lens.width(), 22),
                       QtCore.Qt.AlignCenter, '4× 放大 · 十字对准贴片中心')

    def mouseMoveEvent(self, event):
        self.hover_point = self.image_position(event.pos()) if self.image is not None else None
        if self.magnifier:
            self.update()

    def leaveEvent(self, event):
        self.hover_point = None
        self.update()

    def mousePressEvent(self, event):
        point = self.image_position(event.pos()) if self.image is not None else None
        if point is not None and event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(*point)


class Main(QtWidgets.QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        saved_role = None
        if ROLE_FILE.exists():
            try:
                saved_role = json.loads(ROLE_FILE.read_text(encoding='utf8')).get('sync_role')
            except Exception:
                logging.exception('could not read saved camera role')
        configured_role = cfg.get('sync_role') or saved_role
        self.sync_role = configured_role if configured_role in SYNC_ROLES else None
        if self.sync_role:
            self.cfg['sync_role'] = self.sync_role
            self.cfg['sync_mode'] = SYNC_ROLES[self.sync_role][1]
            self.cfg['host_id'] = self.sync_role
        self.worker = None
        self.meta = None
        self.calibration = None
        self.rig_calibration = None
        self.seen = {}
        self.reconstruction_window = None
        self.recording_active = False
        self.last_highlight = 0.
        self.benchmark_process = None
        target_file = ROOT/'points_blender52_final.json'
        if not target_file.exists():
            target_file = ROOT/'blender_target'/'points_blender52_final.json'
        self.target = json.loads(target_file.read_text(encoding='utf8'))['points']
        self.setWindowTitle('D435i 三机采集 / 引导标定 / 本机重建 | v1.0.0')
        self.resize(1380, 760)
        root = QtWidgets.QVBoxLayout()
        grid = QtWidgets.QGridLayout()
        self.views, self.info = {}, {}
        for col, name in enumerate(('rgb', 'ir1', 'ir2')):
            grid.addWidget(QtWidgets.QLabel(name.upper()), 0, col)
            view = ImageView()
            self.views[name] = view
            grid.addWidget(view, 1, col)
            info = QtWidgets.QLabel('未启动')
            info.setFixedHeight(23)
            info.setStyleSheet('font-size:11px;color:#39424b;')
            grid.addWidget(info, 2, col)
            self.info[name] = info
        grid.setRowStretch(1, 1)
        root.addLayout(grid, 1)
        syncrow = QtWidgets.QHBoxLayout()
        syncrow.addWidget(QtWidgets.QLabel('9-pin 硬件同步角色'))
        self.role_group = QtWidgets.QButtonGroup(self)
        self.role_group.setExclusive(True)
        self.role_buttons = {}
        for role, (label, sync_mode) in SYNC_ROLES.items():
            button = QtWidgets.QPushButton(label)
            button.setCheckable(True)
            button.setMinimumWidth(120)
            button.setToolTip('SDK sync mode %d；Slave 1/2 在 D435i 硬件上都使用 Slave 模式。' % sync_mode)
            self.role_group.addButton(button)
            self.role_buttons[role] = button
            syncrow.addWidget(button)
            button.clicked.connect(lambda checked=False, role=role: self.select_sync_role(role))
            if role == self.sync_role:
                button.setChecked(True)
        self.sync_status = QtWidgets.QLabel()
        syncrow.addWidget(self.sync_status, 1)
        root.addLayout(syncrow)
        self._update_sync_status()
        audiorow = QtWidgets.QHBoxLayout()
        self.audio_device = QtWidgets.QComboBox()
        self.audio_device.addItem('系统默认输入', None)
        refresh_audio = QtWidgets.QPushButton('刷新麦克风')
        refresh_audio.clicked.connect(self.refresh_microphones)
        self.audio_status = QtWidgets.QLabel('每台本机录制 48 kHz 单声道 WAV')
        self.sync_test = QtWidgets.QPushButton('记录同步线测试事件')
        self.sync_test.clicked.connect(self.record_sync_event)
        self.reconstruction_button = QtWidgets.QPushButton('本机重建 / 三机合成 / 回放')
        self.reconstruction_button.clicked.connect(self.open_reconstruction)
        for w in (QtWidgets.QLabel('本机麦克风'), self.audio_device, refresh_audio,
                  self.audio_status, self.sync_test, self.reconstruction_button):
            audiorow.addWidget(w)
        root.addLayout(audiorow)
        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton('开启相机')
        self.stop_button = QtWidgets.QPushButton('关闭相机')
        self.record_button = QtWidgets.QPushButton('开始录像')
        self.record_stop = QtWidgets.QPushButton('停止录像并保存')
        self.emitter_button = QtWidgets.QPushButton('IR emitter：关')
        self.emitter_button.setCheckable(True)
        self.emitter_button.setChecked(bool(cfg.get('emitter_enabled', False)))
        self.emitter_button.setToolTip('控制 D435i 红外投影器，不会关闭 IR1/IR2 相机。录像开始前会自动关闭；建议用 850 nm 泛光灯照亮反光 Mark。')
        self.state = QtWidgets.QLabel('未录像')
        for widget in (self.start_button, self.stop_button, self.emitter_button,
                       self.record_button, self.record_stop, self.state):
            row.addWidget(widget)
        root.addLayout(row)
        self.recording_hint = QtWidgets.QLabel('开启相机并完成本机标定后即可录像；三机结算在汇总时进行。')
        self.recording_hint.setWordWrap(True)
        root.addWidget(self.recording_hint)
        writer_row = QtWidgets.QHBoxLayout()
        writer_row.addWidget(QtWidgets.QLabel('录制方式'))
        self.recording_mode = QtWidgets.QComboBox()
        self.recording_mode.addItem('推荐：RGB MP4＋红外无损压缩', 'balanced')
        self.recording_mode.addItem('通用 MP4（三路有损，体积更小）', 'mp4')
        self.recording_mode.addItem('原始无压缩（占用很大）', 'fast_lossless')
        self.recording_mode.addItem('FFV1 无损压缩（CPU 占用较高）', 'ffv1')
        self.recording_mode.setCurrentIndex(max(0, self.recording_mode.findData(cfg.get('recording_mode', 'balanced'))))
        self.mode_hint = QtWidgets.QLabel()
        self.mode_hint.setWordWrap(True)
        self.benchmark_button = QtWidgets.QPushButton('测试保存目录录制速度')
        self.benchmark_button.clicked.connect(self.start_recording_benchmark)
        self.recording_mode.currentIndexChanged.connect(self.update_mode_hint)
        writer_row.addWidget(self.recording_mode)
        writer_row.addWidget(self.mode_hint, 1)
        writer_row.addWidget(self.benchmark_button)
        root.addLayout(writer_row)
        self.update_mode_hint()
        pathrow = QtWidgets.QHBoxLayout()
        output_root = Path(cfg.get('output_root') or 'sessions')
        if not output_root.is_absolute():
            output_root = ROOT/output_root
        self.path = QtWidgets.QLineEdit(str(output_root))
        browse = QtWidgets.QPushButton('选择保存目录')
        pathrow.addWidget(QtWidgets.QLabel('保存位置'))
        pathrow.addWidget(self.path, 1)
        pathrow.addWidget(browse)
        root.addLayout(pathrow)
        calrow = QtWidgets.QHBoxLayout()
        self.freeze_button = QtWidgets.QPushButton('引导标定（模型对点）')
        self.model_button = QtWidgets.QPushButton('模型点位图')
        self.load_cal_button = QtWidgets.QPushButton('加载已有标定')
        self.load_rig_button = QtWidgets.QPushButton('加载已有三机结算（可选）')
        self.calibration_status = QtWidgets.QLabel('本机未标定')
        self.marker_highlight = QtWidgets.QCheckBox('高亮反光点')
        self.marker_highlight.setChecked(True)
        self.marker_threshold = QtWidgets.QSpinBox()
        self.marker_threshold.setRange(80, 254)
        self.marker_threshold.setValue(210)
        self.marker_threshold.setToolTip('IR 亮点阈值；圈出的候选点还没有分配模型点号。不会改动保存的原始画面。')
        for widget in (self.freeze_button, self.model_button, self.load_cal_button, self.load_rig_button,
                       self.marker_highlight, QtWidgets.QLabel('阈值'), self.marker_threshold, self.calibration_status):
            calrow.addWidget(widget)
        root.addLayout(calrow)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(88)
        root.addWidget(self.log)
        body = QtWidgets.QWidget()
        body.setLayout(root)
        self.setCentralWidget(body)
        self.record_button.setEnabled(False)
        self.record_stop.setEnabled(False)
        self.freeze_button.setEnabled(False)
        self.start_button.setEnabled(self.sync_role is not None)
        self.load_cal_button.setEnabled(False)
        self.load_rig_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.record_button.clicked.connect(self.start_recording)
        self.record_stop.clicked.connect(lambda: self.worker.command('stop_record') if self.worker else None)
        self.emitter_button.toggled.connect(self.set_emitter)
        browse.clicked.connect(self.choose_folder)
        self.freeze_button.clicked.connect(self.open_calibration)
        self.model_button.clicked.connect(self.open_target_reference)
        self.marker_highlight.toggled.connect(self.highlight_changed)
        self.marker_threshold.valueChanged.connect(lambda value: setattr(self, 'last_highlight', 0.))
        self.load_cal_button.clicked.connect(self.load_calibration)
        self.load_rig_button.clicked.connect(self.load_rig_calibration)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)

    def refresh_microphones(self):
        if self.worker:
            return
        try:
            from audio_io import input_devices
            selected = self.audio_device.currentData()
            devices = input_devices()
            self.audio_device.clear()
            self.audio_device.addItem('系统默认输入', None)
            for index, name in devices:
                self.audio_device.addItem(f'{index}: {name}', index)
            self.audio_device.setCurrentIndex(max(0, self.audio_device.findData(selected)))
            self.msg(f'可用麦克风输入设备 {len(devices)} 个')
        except Exception as exc:
            self.msg('麦克风枚举失败：'+str(exc))

    def record_sync_event(self):
        if not self.worker:
            self.msg('请先启动相机；SDK 不能直接检测 Pin 接线状态')
            return
        label, ok = QtWidgets.QInputDialog.getItem(self, '同步测试记录',
            '在从机记录测试动作，再观察 IR 收帧与持续日志。该记录不能证明电气连通。',
            ['基线：Master 正常运行', '测试：Master 已停止', '恢复：Master 已重新运行'], 0, False)
        if ok:
            self.worker.command('sync_event', label)

    def show_health(self, health):
        parts = [f'{name.upper()} {row["fps"]:.1f}fps/{row["state"]}'
                 for name,row in health['streams'].items()]
        self.sync_status.setText(f'{health["sync_role"]} 模式 {health["sync_mode"]} | '+' | '.join(parts))
        self.sync_status.setToolTip('物理连线/曝光锁定状态未知。模式已回读，收帧证据不等于 Pin5 连线检测。')
        level = health.get('audio_level')
        performance = health.get('writer_performance')
        text = '未录音' if level is None else f'音量 {level*100:.1f}%'
        if performance:
            text += f' | 视频缓存 {performance["video_buffer_mb"]:.0f}/{performance["video_buffer_limit_mb"]:.0f} MB'
            self.audio_status.setToolTip(json.dumps(performance, ensure_ascii=False, indent=2))
        self.audio_status.setText(text)

    def open_reconstruction(self):
        if self.recording_active:
            self.msg('请停止录像后再启动离线重建，避免编码和重建争用资源')
            return
        from reconstruction_gui import ReconstructionWindow
        if self.reconstruction_window is None:
            self.reconstruction_window = ReconstructionWindow()
        self.reconstruction_window.show()
        self.reconstruction_window.raise_()

    def _update_sync_status(self):
        if self.sync_role:
            label, mode = SYNC_ROLES[self.sync_role]
            self.sync_status.setText(f'已选 {label}；SDK mode={mode}')
        else:
            self.sync_status.setText('未设置：请在启动相机前为本机选择唯一角色')

    def select_sync_role(self, role):
        if self.worker or role not in SYNC_ROLES:
            return
        self.sync_role = role
        label, mode = SYNC_ROLES[role]
        self.cfg.update(sync_role=role, sync_mode=mode, host_id=role)
        try:
            ROLE_FILE.parent.mkdir(parents=True, exist_ok=True)
            ROLE_FILE.write_text(
                json.dumps({'sync_role': role, 'sync_mode': mode}, indent=2), encoding='utf8')
        except Exception:
            logging.exception('could not save camera role')
        self._update_sync_status()
        self.start_button.setEnabled(True)
        if hasattr(self, 'log'):
            self.msg(f'本机同步角色设置为 {label}，RealSense sync mode={mode}')

    def msg(self, text):
        self.log.appendPlainText(time.strftime('%H:%M:%S ')+text)
        logging.info(text)

    def choose_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, '保存目录', self.path.text())
        if directory:
            self.path.setText(directory)

    def start_camera(self):
        if self.worker:
            return
        if self.sync_role not in SYNC_ROLES:
            QtWidgets.QMessageBox.warning(self, '未设置同步角色', '请先选择本机是 Master、Slave 1 还是 Slave 2。')
            return
        self.meta = None
        self.seen.clear()
        self.clear_preview_marks()
        self.cfg['audio_device'] = self.audio_device.currentData()
        self.cfg['audio_rate'] = 48000
        self.worker = CaptureWorker(self.cfg)
        self.worker.status.connect(self.msg)
        self.worker.ready.connect(self.camera_ready)
        self.worker.recording.connect(self.recording_changed)
        self.worker.emitter_state.connect(self.emitter_changed)
        self.worker.health.connect(self.show_health)
        self.worker.finished.connect(self.camera_finished)
        self.start_button.setEnabled(False)
        self.audio_device.setEnabled(False)
        for button in self.role_buttons.values():
            button.setEnabled(False)
        self.worker.start()

    def camera_ready(self, metadata):
        self.meta = metadata
        self.load_cal_button.setEnabled(True)
        self.load_rig_button.setEnabled(True)
        if self.calibration and self.calibration.get('serial') != metadata.get('serial'):
            self.calibration = None
            self.rig_calibration = None
            self.calibration_status.setText('相机序列号改变，请重新标定')
        if self.rig_calibration:
            try:
                self._select_rig_pose(self.rig_calibration, metadata)
            except Exception as exc:
                self.rig_calibration = None
                self.record_button.setEnabled(False)
                self.calibration_status.setText('当前相机与三机结算不匹配')
                self.msg('三机结算失效：'+str(exc))
        self.update_recording_ready()
        self.freeze_button.setEnabled(True)

    def update_recording_ready(self):
        reason = ''
        measuring = self.benchmark_process is not None and self.benchmark_process.state() != QtCore.QProcess.NotRunning
        if measuring:
            reason = '正在测试保存目录写入速度，请等待结果后再录像。'
        elif not self.worker or not self.meta:
            reason = '请先开启相机，等待画面就绪。'
        elif self.recording_active:
            reason = '正在录像；每帧时间戳和本机位姿编号随录像保存。'
        elif not self.calibration:
            reason = '请先完成或加载本机标定。无需先做三机结算。'
        else:
            try:
                validate_local_calibration(self.calibration, self.meta)
            except (ValueError, KeyError, TypeError) as exc:
                reason = '本机标定不可用：'+str(exc)
        ready = not reason
        text = reason or '本机标定已就绪，可以开始录像。位姿随录像保存，三机结算在汇总时完成。'
        self.record_button.setEnabled(ready)
        self.record_button.setToolTip(text)
        self.recording_hint.setText(text)
        self.recording_hint.setStyleSheet('color:#16784b;' if ready else 'color:#4c6475;')
        self.recording_mode.setEnabled(not self.recording_active and not measuring)
        self.benchmark_button.setEnabled(not self.recording_active and not measuring)

    def update_mode_hint(self):
        mode = self.recording_mode.currentData()
        if mode == 'balanced':
            self.mode_hint.setText('RGB 为通用 H.264 MP4；两路红外 FFV1 压缩保留原始像素，可在软件回放或转出 MP4。体积随画面变化。')
        elif mode == 'mp4':
            self.mode_hint.setText('三路 H.264 MP4，可用常见播放器打开；有损压缩，红外像素会变化，测量精度需重新验证。')
        elif mode == 'fast_lossless':
            w, h = self.cfg.get('resolution', [1280, 720])
            gb_min = w*h*6*int(self.cfg.get('fps', 30))*60/1e9
            self.mode_hint.setText(f'原始像素无损；约 {gb_min:.1f} GB/分钟，建议 SSD。可先测试保存目录。')
        else:
            self.mode_hint.setText('原始像素无损；文件大小取决于画面，编码较慢，建议先测速。')

    def start_recording_benchmark(self):
        if self.recording_active or (self.benchmark_process and self.benchmark_process.state() != QtCore.QProcess.NotRunning):
            return
        process = QtCore.QProcess(self)
        self.benchmark_process = process
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.finished.connect(self.recording_benchmark_finished)
        process.errorOccurred.connect(self.recording_benchmark_error)
        w, h = self.cfg.get('resolution', [1280, 720])
        args = [str(ROOT/'recording_benchmark.py'), '--folder', self.path.text().strip() or str(ROOT/'sessions'),
                '--mode', self.recording_mode.currentData(), '--width', str(w), '--height', str(h),
                '--fps', str(self.cfg.get('fps', 30))]
        process.start(sys.executable, args)
        self.msg('正在按所选格式测试三路录像和 IMU 写入，临时测试数据完成后自动清理。')
        self.update_recording_ready()

    def recording_benchmark_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            process, self.benchmark_process = self.benchmark_process, None
            self.msg('无法启动录制测速：'+process.errorString())
            process.deleteLater()
            self.update_recording_ready()

    def recording_benchmark_finished(self, exit_code, exit_status):
        process = self.benchmark_process
        if process is None:
            return
        output = bytes(process.readAllStandardOutput()).decode('utf8', errors='replace')
        self.benchmark_process = None
        process.deleteLater()
        try:
            report = json.loads(output)
            (ROOT/'logs').mkdir(exist_ok=True)
            (ROOT/'logs/recording_benchmark_latest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
            if report.get('ok'):
                quality = '无损流像素回读一致' if report.get('lossless_pixels_ok') else '视频回读通过'
                self.msg(f'录制测速通过：三路各 {report["expected_frames_per_stream"]} 帧，{quality}；'
                         f'平均文件写入量 {report["average_file_mb_s"]:.1f} MB/s。目录：{report["folder"]}')
            else:
                reason = report.get('error') or report.get('summary', {}).get('error') or '出现持续积压或结束排空过慢'
                self.msg('录制测速未通过：'+reason+'；请换更快的保存目录或调整录制方式，详细结果在 logs/recording_benchmark_latest.json。')
        except (ValueError, KeyError, OSError) as exc:
            self.msg('录制测速未完成：'+str(exc)+'；'+output[-1000:])
        self.update_recording_ready()

    def load_calibration(self):
        if not self.meta:
            self.msg('请先启动相机，再加载该相机的标定文件')
            return
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, '加载本机相机标定', str(ROOT/'sessions'), 'Calibration JSON (*.json)')
        if not file:
            return
        try:
            pose = json.loads(Path(file).read_text(encoding='utf8'))
            if pose.get('serial') != self.meta.get('serial'):
                raise ValueError('标定文件的相机序列号与当前设备不一致')
            if pose.get('stream') != 'ir1' or 'T_camera_from_target_mm' not in pose:
                raise ValueError('文件不是有效的 IR1 标定结果')
            saved = pose.get('intrinsics', {})
            current = self.meta.get('ir1_intrinsics', {})
            fields = ('width', 'height', 'fx', 'fy', 'cx', 'cy')
            for key in fields:
                if key not in saved or key not in current:
                    raise ValueError('标定内参不完整')
                tolerance = 0 if key in ('width', 'height') else 1e-3
                if abs(float(saved[key])-float(current[key])) > tolerance:
                    raise ValueError(f'IR1 内参 {key} 不匹配，请重新标定')
            validate_local_calibration(pose, self.meta)
            self.calibration = pose
            self.rig_calibration = None
            self.update_recording_ready()
            self.calibration_status.setText('本机标定已加载，可录像；位姿随录像保存')
            self.msg(f'已加载相机标定：{file}')
        except Exception as exc:
            logging.exception('load calibration failed')
            QtWidgets.QMessageBox.warning(self, '无法加载标定', str(exc))
            self.msg('加载标定失败：'+str(exc))

    def _select_rig_pose(self, record, metadata):
        verified = verify_settlement_record(record)
        serial = metadata.get('serial')
        entry = next((e for e in verified['cameras'] if e.get('serial') == serial), None)
        if entry is None:
            raise ValueError(f'结算文件不包含当前相机 {serial}')
        pose = entry['calibration']
        saved, current = pose.get('intrinsics', {}), metadata.get('ir1_intrinsics', {})
        for key in ('width', 'height', 'fx', 'fy', 'cx', 'cy'):
            if key not in saved or key not in current:
                raise ValueError('IR1 内参缺失')
            tolerance = 0 if key in ('width', 'height') else 1e-3
            if abs(float(saved[key])-float(current[key])) > tolerance:
                raise ValueError(f'当前相机 IR1 内参 {key} 与结算文件不一致')
        return pose

    def load_rig_calibration(self):
        if not self.meta:
            self.msg('请先启动相机，再加载三机结算文件')
            return
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, '加载已验收的三机结算文件', str(ROOT), 'Rig calibration JSON (*.json)')
        if not file:
            return
        try:
            record = json.loads(Path(file).read_text(encoding='utf8'))
            pose = self._select_rig_pose(record, self.meta)
            if self.calibration and calibration_digest(self.calibration) != calibration_digest(pose):
                raise ValueError('当前相机标定不在这份三机结算中；请用结算文件所含的同一组标定')
            self.calibration = pose
            self.rig_calibration = record
            self.update_recording_ready()
            self.calibration_status.setText(f'三机验收通过 {record["rig_id"]}')
            self.msg(f'三机结算已加载：rig_id={record["rig_id"]}；本机序列号 {self.meta["serial"]}')
        except Exception as exc:
            logging.exception('load rig calibration failed')
            self.rig_calibration = None
            self.update_recording_ready()
            QtWidgets.QMessageBox.warning(self, '三机结算未通过', str(exc))
            self.msg('三机结算未加载；本机标定有效时仍可录像，汇总时重新结算：'+str(exc))

    def camera_finished(self):
        self.worker = None
        self.meta = None
        self.recording_active = False
        self.audio_device.setEnabled(True)
        self.start_button.setEnabled(self.sync_role is not None)
        for button in self.role_buttons.values():
            button.setEnabled(True)
        self.update_recording_ready()
        self.load_cal_button.setEnabled(False)
        self.load_rig_button.setEnabled(False)
        self.freeze_button.setEnabled(False)
        self.record_stop.setEnabled(False)
        self.state.setText('相机已停止')

    def stop_camera(self):
        if self.worker:
            self.worker.command('quit')
            self.msg('正在停止并写完文件…')

    def start_recording(self):
        if not self.worker or not self.meta:
            return
        if self.benchmark_process and self.benchmark_process.state() != QtCore.QProcess.NotRunning:
            self.msg('请等待录制测速完成后再开始录像')
            return
        try:
            if not self.calibration:
                raise ValueError('请先完成或加载本机标定')
            validate_local_calibration(self.calibration, self.meta)
        except (ValueError, KeyError, TypeError) as exc:
            self.update_recording_ready()
            self.msg('尚不能录像：'+str(exc))
            return
        if self.reconstruction_window and self.reconstruction_window.job and self.reconstruction_window.job.isRunning():
            self.msg('请等待离线处理完成再录像')
            return
        self.clear_preview_marks()
        self.worker.command('record', self.path.text().strip() or str(ROOT/'sessions'),
                            self.calibration, self.rig_calibration,
                            {'recording_mode': self.recording_mode.currentData()})

    def set_emitter(self, enabled):
        if self.worker:
            self.emitter_button.setText('IR emitter：切换中…')
            self.worker.command('emitter', enabled)
        else:
            self.cfg['emitter_enabled'] = bool(enabled)
            self.emitter_button.setText('IR emitter：开' if enabled else 'IR emitter：关')

    def emitter_changed(self, enabled):
        self.cfg['emitter_enabled'] = bool(enabled)
        self.emitter_button.blockSignals(True)
        self.emitter_button.setChecked(enabled)
        self.emitter_button.setText('IR emitter：开' if enabled else 'IR emitter：关')
        self.emitter_button.blockSignals(False)

    def recording_changed(self, active, path):
        self.recording_active = active
        self.state.setText('录像中' if active else '录像已保存，详见日志')
        self.state.setStyleSheet('color:#bd2828' if active else '')
        self.update_recording_ready()
        self.record_stop.setEnabled(active)
        self.path.setEnabled(not active)
        self.freeze_button.setEnabled(not active and self.worker is not None and self.meta is not None)
        self.emitter_button.setEnabled(not active)
        for button in (self.load_cal_button, self.load_rig_button, self.reconstruction_button):
            button.setEnabled(not active)
        if not self.worker or not self.meta:
            self.load_cal_button.setEnabled(False)
            self.load_rig_button.setEnabled(False)

    def refresh(self):
        if not self.worker:
            return
        now = time.monotonic()
        highlight_due = now-self.last_highlight >= .20
        if highlight_due:
            self.last_highlight = now
        for name, (array, rec) in self.worker.snapshot().items():
            if self.seen.get(name) == rec['frame_number']:
                continue
            self.seen[name] = rec['frame_number']
            self.views[name].image = image_for_qt(array, bgr=name == 'rgb')
            if name in ('ir1', 'ir2') and highlight_due:
                self.views[name].candidates = bright_markers(array, self.marker_threshold.value()) if self.marker_highlight.isChecked() else []
            self.views[name].update()
            self.info[name].setText(f'{rec["width"]}x{rec["height"]} | {rec["received_fps"]:.1f} FPS | '
                                    f'{rec["device_ts_ms"]:.1f} ms | #{rec["frame_number"]}')

    def clear_preview_marks(self):
        for view in self.views.values():
            view.marks = {}
            view.confirmed = set()
            view.predictions = {}
            view.update()

    def highlight_changed(self, enabled):
        self.last_highlight = 0.
        if not enabled:
            for name in ('ir1', 'ir2'):
                self.views[name].candidates = []
                self.views[name].update()

    def open_target_reference(self):
        from calibration_wizard import TargetReferenceDialog
        TargetReferenceDialog(self.target, self).exec_()

    def open_calibration(self):
        if self.recording_active or not self.worker or not self.meta:
            self.msg('请先开启相机，并停止录像后再进行标定。')
            return
        from calibration_wizard import CalibrationWizard
        worker = self.worker
        reference = {}
        def snapshot():
            if self.worker is not worker:
                return None
            frame = worker.snapshot().get('ir1')
            if frame is not None:
                reference.clear()
                reference.update(frame[1])
            return frame[0].copy() if frame is not None else None
        image = snapshot()
        if image is None:
            self.msg('尚无 IR1 画面，请等待相机收帧。')
            return
        serial = self.meta['serial']
        dialog = CalibrationWizard(self.target, image, self.meta['ir1_intrinsics'], self,
                                   recapture=snapshot, threshold=self.marker_threshold.value())
        if dialog.exec_() == QtWidgets.QDialog.Accepted and dialog.result_pose:
            if self.worker is not worker or not self.meta or self.meta['serial'] != serial:
                self.msg('相机已停止或改变，本次标定未保存，请重新开启相机。')
                return
            pose = dict(dialog.result_pose, serial=serial)
            pose['calibration_frame'] = dict(reference)
            pose['saved_wall_ns'] = time.time_ns()
            self.save_calibration_pose(pose)

    def save_calibration_pose(self, pose):
        try:
            if pose.get('diagnostic_only'):
                raise ValueError('这是未通过检查的诊断结果，不能保存为标定')
            if not self.meta or pose.get('serial') != self.meta.get('serial'):
                raise ValueError('保存的标定与当前相机不匹配')
            validate_local_calibration(pose, self.meta)
            directory = Path(self.path.text().strip() or str(ROOT/'sessions'))
            directory.mkdir(parents=True, exist_ok=True)
            file = directory/('calibration_'+self.meta['serial']+'_'+str(time.time_ns())+'.json')
            file.write_text(json.dumps(pose, indent=2), encoding='utf8')
            self.calibration = pose
            self.rig_calibration = None
            self.update_recording_ready()
            self.calibration_status.setText(f'本机已标定 RMS {pose["undistorted_reprojection_rms_px"]:.2f}px，可直接录像')
            self.msg(f'相机中心(mm) {np.round(pose["camera_center_target_mm"], 2)}；'
                     f'RMS {pose["undistorted_reprojection_rms_px"]:.2f}px；{file}')
        except Exception as exc:
            logging.exception('calibration failed')
            self.msg('标定未保存：'+str(exc))

    def closeEvent(self, event):
        if self.benchmark_process and self.benchmark_process.state() != QtCore.QProcess.NotRunning:
            self.msg('录制测速仍在写入，请完成后再关闭窗口')
            event.ignore()
            return
        if self.reconstruction_window and self.reconstruction_window.job and self.reconstruction_window.job.isRunning():
            self.reconstruction_window.cancel_job()
            self.msg('正在结束重建任务，请完成后关闭窗口')
            event.ignore()
            return
        if self.worker:
            self.stop_camera()
            self.msg('文件写完后请再关闭窗口')
            event.ignore()
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='capture_config.yaml')
    args = parser.parse_args()
    logging.basicConfig(filename=str(ROOT/'capture_debug.log'), level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    def report(exc_type, value, tb):
        logging.error('Uncaught exception', exc_info=(exc_type, value, tb))
        sys.__excepthook__(exc_type, value, tb)
    sys.excepthook = report
    cfgpath = Path(args.config)
    if not cfgpath.is_absolute():
        cfgpath = ROOT/cfgpath
    cfg = yaml.safe_load(cfgpath.read_text(encoding='utf8'))
    app = QtWidgets.QApplication(sys.argv)
    win = Main(cfg)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
