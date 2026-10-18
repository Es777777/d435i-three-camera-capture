"""Guided correspondence picking for the existing 70% three-panel target."""
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from capture_app import ImageView, image_for_qt
from calibration_guidance import (DEFAULT_INSET_MM, VIEW_ORDERS, bright_markers,
                                  refine_marker, build_calibration, predicted_pixels)
from calibration_math import CalibrationReprojectionError

DESCRIPTIONS = {
    'C01': '正面 · 左上', 'C02': '正面 · 上中', 'C03': '正面 · 右上',
    'C04': '正面 · 左中', 'C05': '正面 · 中心', 'C06': '正面 · 右中',
    'C07': '正面 · 左下', 'C08': '正面 · 下中', 'C09': '正面 · 右下',
    'C10': '左翼 · 下', 'C11': '左翼 · 中', 'C12': '左翼 · 上',
    'C13': '右翼 · 下', 'C14': '右翼 · 中', 'C15': '右翼 · 上',
}


class TargetDiagram(QtWidgets.QWidget):
    selected = QtCore.pyqtSignal(str)

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.current = None
        self.confirmed = set()
        self.error_ids = set()
        self.visible_ids = {row[0] for row in target}
        self.setMinimumSize(310, 280)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def map_point(self, xyz):
        scale = min((self.width()-30)/460, (self.height()-90)/240)
        x0, y0 = self.width()/2, 30+(self.height()-90-240*scale)/2
        return QtCore.QPointF(x0+xyz[0]*scale, y0+(240-xyz[2])*scale)

    def point_positions(self):
        return {pid: self.map_point(xyz) for pid, xyz, normal, side in self.target}

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#192a37'))
        for x0, x1, z0, z1, color in (
            (-150, 150, 10, 230, '#455f73'), (-221, -154, 30, 210, '#344e62'),
            (154, 221, 30, 210, '#344e62'), (-225, 225, 0, 16, '#607583'),
        ):
            a, b = self.map_point([x0, 0, z1]), self.map_point([x1, 0, z0])
            painter.setPen(QtGui.QPen(QtGui.QColor('#8ca3b5'), 1))
            painter.setBrush(QtGui.QColor(color))
            painter.drawRoundedRect(QtCore.QRectF(a, b), 3, 3)
        painter.setPen(QtGui.QColor('#d7e5ef'))
        painter.drawText(QtCore.QRectF(0, 4, self.width(), 24), QtCore.Qt.AlignCenter,
                         '面对凹槽看模型 · 左右按模型正面定义')
        for pid, pos in self.point_positions().items():
            color = '#ff665e' if pid in self.error_ids else '#ffe15b' if pid == self.current else '#48ec9a' if pid in self.confirmed else '#d5e1e8'
            if pid not in self.visible_ids:
                color = '#72818c'
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 3 if pid == self.current else 2))
            painter.setBrush(QtGui.QColor('#253847'))
            painter.drawEllipse(pos, 10 if pid == self.current else 6, 10 if pid == self.current else 6)
            if pid == self.current:
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(pos, 15, 15)
            text_rect = QtCore.QRectF(max(1, min(self.width()-43, pos.x()-21)), pos.y()+11, 42, 21)
            painter.fillRect(text_rect, QtGui.QColor('#192a37'))
            painter.drawText(text_rect, QtCore.Qt.AlignCenter, pid)
        painter.setPen(QtGui.QColor('#b9cedc'))
        painter.drawText(QtCore.QRectF(0, self.height()-42, self.width(), 21), QtCore.Qt.AlignCenter,
                         '黄＝当前点    绿＝已选    红＝优先复核')
        painter.drawText(QtCore.QRectF(0, self.height()-22, self.width(), 21), QtCore.Qt.AlignCenter,
                         '点击模型上的点号可重选')

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        for pid, pos in self.point_positions().items():
            if QtCore.QLineF(pos, QtCore.QPointF(event.pos())).length() <= 17:
                self.selected.emit(pid)
                return


class TargetReferenceDialog(QtWidgets.QDialog):
    def __init__(self, target, parent=None):
        super().__init__(parent)
        self.setWindowTitle('原有凹槽标定模型与点号')
        self.resize(720, 630)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel('沿用原来的 70% 三面 15 凹槽模型，贴片放在凹槽内。')
        title.setWordWrap(True)
        layout.addWidget(title)
        tabs = QtWidgets.QTabWidget()
        diagram = TargetDiagram(target)
        diagram.selected.connect(lambda pid: self.select(diagram, pid))
        tabs.addTab(diagram, '可点击点号图')
        preview = QtWidgets.QLabel()
        preview.setAlignment(QtCore.Qt.AlignCenter)
        file = Path(__file__).resolve().parent/'assets'/'calibration_target_70.png'
        if file.exists():
            preview.setPixmap(QtGui.QPixmap(str(file)).scaled(660, 430, QtCore.Qt.KeepAspectRatio,
                                                            QtCore.Qt.SmoothTransformation))
            tabs.addTab(preview, '原模型预览')
        layout.addWidget(tabs, 1)
        self.description = QtWidgets.QLabel('正面 C01–C09 按行从左上到右下；左翼从上到下是 C12、C11、C10，右翼是 C15、C14、C13。')
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        note = QtWidgets.QLabel('默认：5 mm 反光贴 / 5.4 mm 单槽；模型比例 70%，软件自动换算。\n'
                                '标定时选择正面加一个可见翼面，至少 6 点；三台机位和模型保持不动。')
        note.setWordWrap(True)
        layout.addWidget(note)
        close = QtWidgets.QPushButton('知道了')
        close.clicked.connect(self.accept)
        layout.addWidget(close)

    def select(self, diagram, pid):
        diagram.current = pid
        diagram.update()
        self.description.setText(f'{pid}：{DESCRIPTIONS[pid]}。在这个凹槽内贴反光片，点击实拍图时对准贴片中心。')


class CalibrationWizard(QtWidgets.QDialog):
    def __init__(self, target, image, intrinsics, parent=None, recapture=None, threshold=210):
        super().__init__(parent)
        self.target = target
        self.raw = np.asarray(image).copy()
        self.intrinsics = dict(intrinsics)
        self.recapture = recapture
        self.observations = {}
        self.history = []
        self.skipped = set()
        self.current_id = None
        self.pose = None
        self.result_pose = None
        self.validation_error = ''
        self.diagnostic_pose = None
        self._validation_key = None
        self.setWindowTitle('引导标定 v1.0.0 · 跟着黄色点点击即可')
        self.resize(1260, 840)
        self.setMinimumSize(1030, 720)
        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel('① 选择能看到的翼面   →   ② 按黄色点逐个点击贴片中心   →   ③ 检查后保存')
        heading.setWordWrap(True)
        heading.setStyleSheet('font-size:16px;font-weight:600;padding:9px;background:#e7f0f6;')
        layout.addWidget(heading)
        controls = QtWidgets.QHBoxLayout()
        self.view_mode = QtWidgets.QComboBox()
        self.view_mode.addItem('先选择画面能看到的翼面…', None)
        self.view_mode.addItem('正面 + 左翼（C10–C12）', 'left')
        self.view_mode.addItem('正面 + 右翼（C13–C15）', 'right')
        self.view_mode.addItem('三面都能看见', 'both')
        controls.addWidget(self.view_mode)
        self.snap = QtWidgets.QCheckBox('点击时吸附附近亮点中心')
        self.snap.setChecked(True)
        controls.addWidget(self.snap)
        controls.addWidget(QtWidgets.QLabel('亮点阈值'))
        self.threshold = QtWidgets.QSpinBox()
        self.threshold.setRange(80, 254)
        self.threshold.setValue(threshold)
        controls.addWidget(self.threshold)
        controls.addStretch()
        layout.addLayout(controls)
        body = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel('已冻结 IR1 · 点击即确认并跳到下一点 · 滚轮缩放，悬停可 4× 放大')
        label.setWordWrap(True)
        left.addWidget(label)
        self.scene = ImageView()
        self.scene.magnifier = True
        self.scene.zoomable = True
        self.scene.setMinimumSize(560, 350)
        self.scene.image = image_for_qt(self.raw)
        left.addWidget(self.scene, 1)
        legend = QtWidgets.QLabel('青色＝候选亮点   绿色＝已选   红色＝优先复核   虚线＝计算位置，仅供核对，仍需点击实拍贴片')
        legend.setWordWrap(True)
        left.addWidget(legend)
        body.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        self.prompt = QtWidgets.QLabel('先选择可见的翼面')
        self.prompt.setStyleSheet('font-size:19px;font-weight:600;color:#735300;background:#fff2b8;padding:12px;')
        self.prompt.setWordWrap(True)
        right.addWidget(self.prompt)
        self.diagram = TargetDiagram(target)
        self.diagram.setMinimumHeight(230)
        right.addWidget(self.diagram, 1)
        self.progress = QtWidgets.QLabel('已选 0 点；至少 6 点，必须包含翼面。')
        self.progress.setWordWrap(True)
        right.addWidget(self.progress)
        self.quality = QtWidgets.QLabel()
        self.quality.setWordWrap(True)
        right.addWidget(self.quality)
        self.error_table = QtWidgets.QTableWidget(0, 3)
        self.error_table.setHorizontalHeaderLabels(['点号', '偏差(px)', '状态'])
        self.error_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.error_table.verticalHeader().hide()
        self.error_table.verticalHeader().setDefaultSectionSize(24)
        self.error_table.setMaximumHeight(112)
        self.error_table.setMinimumHeight(90)
        self.error_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.error_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.error_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.error_table.setToolTip('按偏差从大到小排列；点击行可重新选择该点。偏差用于优先排查，不代表该点一定点错。')
        self.error_table.cellClicked.connect(self.review_table_point)
        right.addWidget(self.error_table)
        review_actions = QtWidgets.QHBoxLayout()
        self.review_button = QtWidgets.QPushButton('定位最大偏差点')
        self.remove_button = QtWidgets.QPushButton('移除当前点（可撤销）')
        self.review_button.clicked.connect(self.review_worst)
        self.remove_button.clicked.connect(self.remove_current)
        review_actions.addWidget(self.review_button)
        review_actions.addWidget(self.remove_button)
        right.addLayout(review_actions)
        params = QtWidgets.QGroupBox('尺寸参数（沿用原模型，通常不用改）')
        params.setCheckable(True)
        params.setChecked(False)
        param_layout = QtWidgets.QVBoxLayout(params)
        self.inset = QtWidgets.QDoubleSpinBox()
        self.inset.setRange(0, 2)
        self.inset.setDecimals(3)
        self.inset.setSuffix(' mm')
        self.inset.setValue(DEFAULT_INSET_MM)
        param_body = QtWidgets.QWidget()
        param_content = QtWidgets.QVBoxLayout(param_body)
        param_note = QtWidgets.QLabel('反光面内缩：默认 0.75 mm 槽深 − 0.2 mm 贴片')
        param_note.setWordWrap(True)
        param_content.addWidget(param_note)
        param_content.addWidget(self.inset)
        param_layout.addWidget(param_body)
        param_body.setVisible(False)
        params.toggled.connect(param_body.setVisible)
        right.addWidget(params)
        reference = QtWidgets.QPushButton('查看原模型与点号说明')
        reference.clicked.connect(lambda: TargetReferenceDialog(self.target, self).exec_())
        right.addWidget(reference)
        panel = QtWidgets.QWidget()
        panel.setLayout(right)
        panel.setFixedWidth(350)
        body.addWidget(panel)
        layout.addLayout(body, 1)
        self.last_action = QtWidgets.QLabel('三台相机与标定模型保持不动；画面没有亮点时可调整阈值或手动选中心。')
        self.last_action.setWordWrap(True)
        layout.addWidget(self.last_action)
        self.save_status = QtWidgets.QLabel()
        self.save_status.setWordWrap(True)
        self.save_status.setMinimumHeight(32)
        layout.addWidget(self.save_status)
        actions = QtWidgets.QHBoxLayout()
        self.undo_button = QtWidgets.QPushButton('撤销上一步')
        self.skip_button = QtWidgets.QPushButton('看不见，跳过此点')
        self.recapture_button = QtWidgets.QPushButton('重拍并清空点位')
        self.recapture_button.setEnabled(recapture is not None)
        self.save_button = QtWidgets.QPushButton('检查并保存本机标定')
        self.save_button.setStyleSheet('''
            QPushButton {font-weight:600;padding:10px;background:#edf1f5;
                         color:#273e50;border:1px solid #93a6b5;border-radius:4px;}
            QPushButton[ready="true"] {background:#d7f3e3;color:#126440;border-color:#57a67c;}
            QPushButton:hover {border:2px solid #426c88;padding:9px;}
            QPushButton:pressed {background:#c8d9e5;}
        ''')
        cancel = QtWidgets.QPushButton('取消')
        fit = QtWidgets.QPushButton('适应全图')
        fit.clicked.connect(self.scene.reset_view)
        for button in (self.undo_button, self.skip_button, self.recapture_button):
            actions.addWidget(button)
        actions.addWidget(fit)
        self.diagnostic_button = QtWidgets.QPushButton('导出本次点位与画面')
        self.diagnostic_button.clicked.connect(self.export_diagnostic)
        actions.addWidget(self.diagnostic_button)
        actions.addStretch()
        actions.addWidget(cancel)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        self.view_mode.currentIndexChanged.connect(self.change_view)
        self.scene.clicked.connect(self.select_image_point)
        self.diagram.selected.connect(self.choose_point)
        self.threshold.valueChanged.connect(self.update_candidates)
        self.inset.valueChanged.connect(self.update_state)
        self.undo_button.clicked.connect(self.undo)
        self.skip_button.clicked.connect(self.skip)
        self.recapture_button.clicked.connect(self.take_new_frame)
        self.save_button.clicked.connect(self.prepare_save)
        cancel.clicked.connect(self.reject)
        self.update_candidates()

    @property
    def order(self):
        return VIEW_ORDERS.get(self.view_mode.currentData(), [])

    def change_view(self):
        self.skipped.clear()
        self.current_id = None
        self.advance()
        self.update_state()

    def advance(self):
        order = self.order
        start = order.index(self.current_id)+1 if self.current_id in order else 0
        cycle = order[start:]+order[:start]
        self.current_id = next((pid for pid in cycle if pid not in self.observations and pid not in self.skipped), None)

    def choose_point(self, pid):
        if pid not in self.order:
            self.last_action.setText('这个点不在当前可见翼面中，请先在上方切换可见翼面。')
            return
        self.current_id = pid
        self.skipped.discard(pid)
        self.update_state()

    def update_candidates(self):
        self.candidates = bright_markers(self.raw, self.threshold.value())
        self.scene.candidates = self.candidates
        self.update_state()

    def select_image_point(self, x, y):
        if self.current_id is None:
            self.last_action.setText('请先选择可见翼面，或点击右侧模型点号重选。')
            return
        point, method = refine_marker(self.raw, (x, y), self.candidates) if self.snap.isChecked() else ((x, y), None)
        for pid, other in self.observations.items():
            if pid != self.current_id and np.linalg.norm(np.asarray(point)-other) < 5:
                self.last_action.setText(f'这里已经是 {pid}，请按右侧模型找到 {self.current_id}，不要重复点击同一贴片。')
                return
        selected = self.current_id
        self.history.append((dict(self.observations), selected, set(self.skipped)))
        self.observations[selected] = point
        self.last_action.setText(f'{selected} 已确认（'+('已吸附'+method if method else '手动位置，请放大核对贴片中心')+'）；选错可撤销，或点击模型点号重选。')
        self.advance()
        self.update_state()

    def undo(self):
        if self.history:
            self.observations, self.current_id, self.skipped = self.history.pop()
            self.last_action.setText('已撤销，请重新点击 '+str(self.current_id))
            self.update_state()

    def skip(self):
        if self.current_id:
            self.history.append((dict(self.observations), self.current_id, set(self.skipped)))
            self.skipped.add(self.current_id)
            self.advance()
            self.update_state()

    def point_errors(self):
        pose = self.pose if self.pose is not None else self.diagnostic_pose
        return sorted(zip(pose['point_ids'], pose['per_point_error_px']),
                      key=lambda row: row[1], reverse=True) if pose else []

    def review_table_point(self, row, column):
        item = self.error_table.item(row, 0)
        if item is not None:
            self.focus_point(item.text())

    def focus_point(self, pid):
        # Already selected points may belong to a previously chosen wing view.
        if pid not in self.observations:
            return
        self.current_id = pid
        x, y = self.observations[pid]
        width = min(260., float(self.raw.shape[1]))
        height = width*self.raw.shape[0]/self.raw.shape[1]
        self.scene.view_region = QtCore.QRectF(
            max(0., min(self.raw.shape[1]-width, x-width/2)),
            max(0., min(self.raw.shape[0]-height, y-height/2)), width, height)
        self.update_state()
        self.last_action.setText(f'正在复核 {pid} · {DESCRIPTIONS[pid]}：点击实拍贴片中心重新确认；橙色虚线只供参考，不要照虚线盲点。')

    def review_worst(self):
        errors = self.point_errors()
        if errors:
            self.focus_point(errors[0][0])

    def remove_current(self):
        selected = self.current_id
        if selected not in self.observations:
            return
        self.history.append((dict(self.observations), selected, set(self.skipped)))
        del self.observations[selected]
        self.skipped.add(selected)
        self.advance()
        self.update_state()
        self.last_action.setText(f'已移除 {selected}；其他点位保留。仍需至少 6 点、含翼面并通过检查；可撤销恢复此点。')

    def write_diagnostic(self, path):
        self.update_state()
        report = dict(kind='calibration_pick_diagnostic', version='8.3',
                      usable_as_calibration=False, target=self.target, intrinsics=self.intrinsics,
                      marker_inset_mm=self.inset.value(), target_scale=.70,
                      view=self.view_mode.currentData(), observations=self.observations,
                      validation_error=self.validation_error,
                      checked_pose=self.pose, diagnostic_pose=self.diagnostic_pose)
        ok, encoded = cv2.imencode('.png', self.raw)
        if not ok:
            raise RuntimeError('无法保存原始 IR1 画面')
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('points_and_checks.json', json.dumps(report, ensure_ascii=False, indent=2))
            archive.writestr('ir1.png', encoded.tobytes())

    def export_diagnostic(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, '导出本次点位与原始画面',
                                                      'calibration_diagnostic.zip', 'ZIP (*.zip)')
        if path:
            try:
                self.write_diagnostic(path)
                self.last_action.setText('已导出点位、相机参数、检查结果和原始 IR1 画面：'+path)
            except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                self.last_action.setText('导出失败：'+str(exc))

    def take_new_frame(self):
        image = self.recapture() if self.recapture else None
        if image is None:
            self.last_action.setText('没有可用的新画面，请检查相机状态。')
            return
        self.raw = np.asarray(image).copy()
        self.scene.image = image_for_qt(self.raw)
        self.scene.reset_view()
        self.observations.clear()
        self.history.clear()
        self.skipped.clear()
        self.current_id = None
        self.advance()
        self.update_candidates()
        self.last_action.setText('已重新冻结画面并清空点位，请重新按提示点击。')

    def update_state(self):
        # Reuse the checked solution while its full inputs are unchanged. A second
        # randomized PnP solve on a save click could otherwise invalidate it.
        key = json.dumps([self.target, self.intrinsics, sorted(self.observations.items()),
                          self.inset.value()], sort_keys=True)
        if key != self._validation_key:
            self._validation_key = key
            self.pose = None
            self.diagnostic_pose = None
            self.validation_error = ''
            if len(self.observations) < 6:
                self.validation_error = f'已选 {len(self.observations)} 点，还差 {6-len(self.observations)} 点；至少 6 点，必须包含翼面。'
            else:
                try:
                    self.pose = build_calibration(self.target, self.observations, self.intrinsics, self.inset.value())
                except CalibrationReprojectionError as exc:
                    self.diagnostic_pose = exc.diagnostic_pose
                    self.validation_error = str(exc)
                except (ValueError, RuntimeError, cv2.error) as exc:
                    self.validation_error = str(exc)+'；点击模型上对应点号可重选。'
        if self.pose is not None:
            status = f'检查通过 · RMS {self.pose["undistorted_reprojection_rms_px"]:.2f} px。点击下方绿色按钮保存本机标定。'
            color = '#16784b'
            self.quality.setText('检查通过；下表可查看各点偏差。')
        else:
            status = '暂不能保存：'+self.validation_error
            color = '#a13a20' if len(self.observations) >= 6 else '#4c6475'
            self.quality.setText('未通过：点击下表或“定位最大偏差点”复核。' if self.diagnostic_pose else self.validation_error)
        self.quality.setStyleSheet(f'color:{color};')
        self.save_status.setText(status)
        self.save_status.setStyleSheet(f'color:{color};padding:5px;font-weight:600;')
        current = self.current_id
        if self.pose and current and current not in self.observations:
            prompt = f'检查通过，可直接保存\n可选补点：{current} · {DESCRIPTIONS[current]}'
        else:
            prompt = f'现在点 {current}\n{DESCRIPTIONS[current]}' if current else '本视角点位已选完' if self.order else '先选择可见的翼面'
        self.prompt.setText(prompt)
        self.diagram.current = current
        self.diagram.confirmed = set(self.observations)
        errors = self.point_errors()
        bad = {pid for pid, error in errors if error > 3.}
        if self.diagnostic_pose and not bad:
            bad = {pid for pid, error in errors[:3]}
        self.diagram.error_ids = bad
        self.diagram.visible_ids = set(self.order) if self.order else {row[0] for row in self.target}
        self.diagram.update()
        side_count = sum(pid in self.observations for pid, xyz, normal, side in self.target if side != 'front')
        self.progress.setText(f'已选 {len(self.observations)} 点（翼面 {side_count} 点）\n全图候选 {len(getattr(self, "candidates", []))} 个；点击可局部寻中心。')
        self.scene.marks = dict(self.observations)
        self.scene.confirmed = set(self.observations)
        self.scene.active_mark = current
        self.scene.error_marks = bad
        self.scene.reference_only = self.diagnostic_pose is not None
        self.scene.predictions = {}
        if self.pose and current and current not in self.observations:
            pixels = predicted_pixels(self.target, self.pose, self.intrinsics, self.inset.value())
            if current in pixels:
                self.scene.predictions = {current: pixels[current]}
        elif self.diagnostic_pose is not None:
            pixels = predicted_pixels(self.target, self.diagnostic_pose, self.intrinsics, self.inset.value())
            show = {current} if current in bad else bad
            self.scene.predictions = {pid: pixels[pid] for pid in show if pid in pixels}
        self.scene.update()
        self.error_table.setRowCount(len(errors))
        for row, (pid, error) in enumerate(errors):
            for col, text in enumerate((pid, f'{error:.2f}', '优先复核' if pid in bad else '已检查' if self.pose else '参考')):
                item = QtWidgets.QTableWidgetItem(text)
                item.setForeground(QtGui.QColor('#b53525' if pid in bad else '#16784b' if self.pose else '#4c6475'))
                self.error_table.setItem(row, col, item)
        self.review_button.setEnabled(bool(errors))
        self.remove_button.setEnabled(current in self.observations)
        self.undo_button.setEnabled(bool(self.history))
        self.skip_button.setEnabled(current is not None)
        self.save_button.setToolTip(status)
        self.save_button.setProperty('ready', self.pose is not None)
        self.save_button.style().unpolish(self.save_button)
        self.save_button.style().polish(self.save_button)
        self.save_button.update()

    def prepare_save(self):
        self.result_pose = None
        self.update_state()
        if self.pose is not None:
            self.result_pose = deepcopy(self.pose)
            self.accept()
        else:
            self.last_action.setText('未保存：请点击“定位最大偏差点”放大重选，其他点位保留。' if self.diagnostic_pose else '未保存：'+self.validation_error)
