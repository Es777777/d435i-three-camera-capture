import json
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from qt_runtime import configure_qt
configure_qt()

from calibration_settlement import settle_calibration_files

ROOT = Path(__file__).resolve().parent


class SettlementWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('D435i 三机标定结算与几何复核（可选）')
        self.resize(1100, 720)
        self.files = [None, None, None]
        self.record = None

        root = QtWidgets.QVBoxLayout()
        note = QtWidgets.QLabel(
            '本机 IR1 标定完成后即可录像，无需等待本工具。本工具可选用于汇总和人工复核三份标定 JSON。\n'
            '三次标定期间，3D 标定物必须保持同一位置和朝向。\n'
            '本工具会检查文件、位姿矩阵和重投影误差，并计算机位基线；无法仅凭 JSON 自动判断标定物是否被移动。')
        note.setWordWrap(True)
        root.addWidget(note)

        self.path_labels = []
        for i in range(3):
            line = QtWidgets.QHBoxLayout()
            line.addWidget(QtWidgets.QLabel(f'机位 {i+1} 标定 JSON'))
            label = QtWidgets.QLabel('未选择')
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            button = QtWidgets.QPushButton('选择文件')
            button.clicked.connect(lambda checked=False, i=i: self.choose(i))
            self.path_labels.append(label)
            line.addWidget(label, 1)
            line.addWidget(button)
            root.addLayout(line)

        self.confirm = QtWidgets.QCheckBox('我确认三份 JSON 对应同一个固定标定物坐标系，三次标定之间没有移动或转动标定物。')
        self.confirm.stateChanged.connect(self.refresh)
        root.addWidget(self.confirm)
        self.geometry_confirm = QtWidgets.QCheckBox('我已核对三机中心位置、两两基线和朝向角与实际支架布局相符；异常数据已返工。')
        self.geometry_confirm.stateChanged.connect(self.refresh)
        root.addWidget(self.geometry_confirm)

        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['机位', '序列号', '相机中心 X/Y/Z (mm)', 'RMS (px)', '最大点误差 (px)', '朝向偏差 (deg)', '状态'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, 1)

        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(180)
        root.addWidget(self.report)

        footer = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel('导入三份标定 JSON 后开始检查。')
        self.status.setWordWrap(True)
        self.save_button = QtWidgets.QPushButton('验收通过并生成 rig_calibration.json')
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save)
        footer.addWidget(self.status, 1)
        footer.addWidget(self.save_button)
        root.addLayout(footer)

        body = QtWidgets.QWidget()
        body.setLayout(root)
        self.setCentralWidget(body)

    def choose(self, index):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f'选择机位 {index+1} 标定文件', str(ROOT/'sessions'), 'Calibration JSON (*.json)')
        if file:
            self.files[index] = Path(file)
            self.path_labels[index].setText(file)
            self.refresh()

    def refresh(self, *_):
        self.record = None
        self.save_button.setEnabled(False)
        if any(p is None for p in self.files):
            self.status.setText('请导入三份标定 JSON。')
            return
        try:
            record = settle_calibration_files(self.files, self.confirm.isChecked(),
                                              self.geometry_confirm.isChecked())
            self.record = record
            self.table.setRowCount(3)
            for i, entry in enumerate(record['cameras']):
                serial = entry['serial']
                geometry = record['camera_geometry'].get(serial, {})
                center = geometry.get('camera_center_target_mm', [float('nan')]*3)
                center_text = ', '.join(f'{x:.1f}' for x in center)
                rms = geometry.get('reprojection_rms_px', float('nan'))
                max_error = geometry.get('max_point_error_px', float('nan'))
                aim = geometry.get('aim_error_to_target_center_deg', float('nan'))
                values = [entry['slot'], serial, center_text, f'{rms:.3f}', f'{max_error:.3f}', f'{aim:.1f}', 'PASS']
                for col, value in enumerate(values):
                    self.table.setItem(i, col, QtWidgets.QTableWidgetItem(str(value)))
            lines = []
            for pair in record['pairwise_geometry']:
                lines.append(f"{pair['camera_a']} ↔ {pair['camera_b']}: 基线 {pair['baseline_mm']:.1f} mm；"
                             f"相对转角 {pair['relative_rotation_angle_deg']:.1f}°")
            lines.extend('警告：'+w for w in record['warnings'])
            lines.extend('错误：'+e for e in record['errors'])
            if not lines:
                lines.append('三份文件的序列号、位姿、标定物参数、点位坐标和重投影误差检查通过。')
                lines.append('生成的 rig ID：'+record['rig_id'])
            self.report.setPlainText('\n'.join(lines))
            if record['validation_passed']:
                self.status.setText('结算通过，可保存用于复核。采集可凭本机标定直接录像，汇总时自动按位姿戳结算。')
                self.save_button.setEnabled(True)
            else:
                self.status.setText('三机复核未通过，请处理错误。本机标定有效时仍可录像，汇总前需解决三机数据不一致。')
        except Exception as exc:
            self.table.setRowCount(0)
            self.report.setPlainText(str(exc))
            self.status.setText('读取或结算失败，不能生成通过文件。')

    def save(self):
        if not self.record or not self.record.get('validation_passed'):
            return
        default = ROOT/f'rig_calibration_{self.record["rig_id"]}.json'
        file, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, '保存三机结算文件', str(default), 'Rig calibration JSON (*.json)')
        if not file:
            return
        path = Path(file)
        if path.suffix.lower() != '.json':
            path = path.with_suffix('.json')
        path.write_text(json.dumps(self.record, ensure_ascii=False, indent=2), encoding='utf8')
        QtWidgets.QMessageBox.information(
            self, '结算文件已生成',
            f'Rig ID: {self.record["rig_id"]}\n\n此文件可用于三机几何复核。各机凭本机标定即可录像，最终合成会按保存的位姿戳重新结算。')
        self.status.setText(f'已保存：{path}')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = SettlementWindow()
    window.show()
    sys.exit(app.exec_())
