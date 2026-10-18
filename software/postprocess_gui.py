"""Three-host event alignment, explicit marker correspondence and conservative tracking."""
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5 import QtCore, QtWidgets

from calibration_math import triangulate_observations
from calibration_settlement import calibration_digest, settle_recorded_calibrations, validate_local_calibration
from pose_stamps import read_pose_stamps
from capture_app import ImageView, image_for_qt


def normalized(k, pixel):
    intr = rs.intrinsics()
    intr.width, intr.height = k['width'], k['height']
    intr.fx, intr.fy, intr.ppx, intr.ppy = k['fx'], k['fy'], k['cx'], k['cy']
    intr.coeffs = k['coeffs']
    intr.model = getattr(rs.distortion, k['distortion'].split('.')[-1])
    p = rs.rs2_deproject_pixel_to_point(intr, list(map(float, pixel)), 1.)
    return [p[0]/p[2], p[1]/p[2]]


def read_gray(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    if not ok:
        raise RuntimeError(f'无法读取视频帧 {index}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def track(prev, gray, points):
    if not points:
        return {}
    ids = list(points)
    p = np.asarray([points[k] for k in ids], np.float32).reshape(-1, 1, 2)
    q, ok, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p, None, winSize=(21, 21), maxLevel=3)
    if q is None:
        return {}
    back, bok, _ = cv2.calcOpticalFlowPyrLK(gray, prev, q, None, winSize=(21, 21), maxLevel=3)
    if back is None:
        return {}
    result = {}
    for i, pid in enumerate(ids):
        if not ok[i, 0] or not bok[i, 0] or np.linalg.norm(back[i]-p[i]) > 1:
            continue
        x, y = q[i, 0]
        if x < 7 or y < 7 or x >= gray.shape[1]-7 or y >= gray.shape[0]-7:
            continue
        x0, y0 = int(x)-6, int(y)-6
        patch = gray[y0:y0+13, x0:x0+13]
        if int(patch.max())-int(np.median(patch)) < 20:
            continue
        _, mask = cv2.threshold(patch, max(80, int(patch.max())*.7), 255, cv2.THRESH_BINARY)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        candidates = [(np.linalg.norm(c+[x0, y0]-[x, y]), c+[x0, y0])
                      for j, c in enumerate(centers[1:], 1) if 2 <= stats[j, cv2.CC_STAT_AREA] <= 90]
        if not candidates:
            continue
        distance, center = min(candidates, key=lambda a: a[0])
        if distance <= 4:
            result[pid] = center.tolist()
    # Never assign two IDs to the same detected spot.
    ambiguous = {a for a in result for b in result if a != b
                 and np.linalg.norm(np.array(result[a])-result[b]) < 3}
    return {k: v for k, v in result.items() if k not in ambiguous}


class Export(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)

    def __init__(self, sessions, anchors, seeds, output):
        super().__init__()
        self.sessions, self.anchors, self.seeds, self.output = sessions, anchors, seeds, Path(output)

    def run(self):
        caps = []
        try:
            frames, poses, timelines, motion, rig_ids, sync_roles = [], [], [], [], [], []
            # Two shared visual events provide offset and linear clock-rate mapping.
            reference = None
            for i, folder in enumerate(self.sessions):
                meta = json.loads((folder/'session.json').read_text(encoding='utf8'))
                role = meta.get('sync_role')
                expected_mode = {'master': 1, 'slave1': 2, 'slave2': 2}.get(role)
                if expected_mode is None or int(meta.get('sync_mode', -1)) != expected_mode:
                    raise ValueError(f'机位 {i+1} 没有有效的 Master/Slave 同步模式记录')
                sync_roles.append(role)
                pose = meta.get('calibration')
                if not pose:
                    raise ValueError(f'机位 {i+1} 录像没有绑定标定；请先标定再录像')
                if pose['serial'] != meta['serial']:
                    raise ValueError('标定序列号与录像不一致')
                validate_local_calibration(pose)
                stamps=read_pose_stamps(folder,meta)
                if len(stamps)!=1:
                    raise ValueError('此录像包含多个位姿戳，请使用“本机重建 / 三机合成”按帧处理')
                poses.append(pose)
                records = [json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
                rec = sorted([r for r in records if r['stream'] == 'ir1'], key=lambda r: r['video_index'])
                pose_key=calibration_digest(pose)
                if any(r.get('pose_stamp_id',pose_key) not in stamps for r in rec):
                    raise ValueError('录像帧引用了不存在的位姿戳')
                if len(rec) < 2:
                    raise ValueError('IR1 视频不足两帧')
                domains = {r['timestamp_domain'] for r in rec}
                ts = np.array([r['device_ts_ms'] for r in rec])
                if len(domains) != 1 or np.any(np.diff(ts) <= 0):
                    raise ValueError('时间戳重置或时钟域变化，拒绝自动对齐')
                a, b = self.anchors[i]
                if not 0 <= a < b < len(rec):
                    raise ValueError('两个同步事件必须按先后顺序填写有效视频帧索引')
                if reference is None:
                    reference = ts[b]-ts[a]
                scale = reference/(ts[b]-ts[a])
                if abs(scale-1) > .02:
                    raise ValueError('两次事件对应关系疑似错误：时钟比例偏差超过 2%')
                timelines.append((ts-ts[a])*scale)
                frames.append(rec)
                caps.append(cv2.VideoCapture(str(folder/rec[0]['video_file'])))
                # IMU flags motion; it does not fabricate a corrected translation.
                first_motion = float('inf')
                imu_file = folder/'imu.csv'
                if imu_file.exists():
                    with imu_file.open(encoding='utf8') as f:
                        for row in csv.DictReader(f):
                            if row['timestamp_domain'] not in domains:
                                continue
                            norm = np.linalg.norm([float(row[k]) for k in ('x', 'y', 'z')])
                            alarm = (row['stream'] == 'gyro' and norm > .05) or (row['stream'] == 'accel' and abs(norm-9.80665) > 1.)
                            mapped = (float(row['device_ts_ms'])-ts[a])*scale
                            if alarm and mapped >= 0:
                                first_motion = min(first_motion, mapped)
                motion.append(first_motion)
            rig=settle_recorded_calibrations(poses)
            rig_ids=[rig['rig_id']]
            default_pose_ids=[calibration_digest(p) for p in poses]
            if set(sync_roles) != {'master', 'slave1', 'slave2'}:
                raise ValueError('三份录像必须分别来自 Master、Slave 1、Slave 2，不能重复或缺位')
            ids = sorted(set().union(*(s.keys() for s in self.seeds)))
            if not ids:
                raise ValueError('请标出同名面部 Mark 后再处理')
            positions = [dict(s) for s in self.seeds]
            current = [a for a, b in self.anchors]
            previous = [read_gray(caps[i], current[i]) for i in range(3)]
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output/'rig_calibration.json').write_text(json.dumps(rig,indent=2),encoding='utf8')
            output_file = self.output/'markers_3d.jsonl'
            if output_file.exists():
                raise ValueError('输出目录已存在 markers_3d.jsonl，请选新的目录')
            end = min(t[-1] for t in timelines)
            with output_file.open('w', encoding='utf8') as f:
                for index in range(self.anchors[0][0], len(frames[0])):
                    timestamp = timelines[0][index]
                    if timestamp > end:
                        break
                    selected = [index]
                    for t in timelines[1:]:
                        pos = int(np.searchsorted(t, timestamp))
                        selected.append(min((max(0, pos-1), min(len(t)-1, pos)), key=lambda j: abs(t[j]-timestamp)))
                    for i in range(3):
                        while current[i] < selected[i]:
                            current[i] += 1
                            gray = read_gray(caps[i], current[i])
                            positions[i] = track(previous[i], gray, positions[i])
                            previous[i] = gray
                    for pid in ids:
                        observations, used = [], []
                        for i in range(3):
                            error_ms = abs(timelines[i][selected[i]]-timestamp)
                            if (pid not in positions[i] or error_ms > 5 or timestamp >= motion[i]
                                    or not frames[i][selected[i]].get('pose_valid', True)):
                                continue
                            k = poses[i]['intrinsics']
                            observations.append(dict(normalized_xy=normalized(k, positions[i][pid]),
                                                     T_camera_from_target_mm=poses[i]['T_camera_from_target_mm'],
                                                     fx=k['fx'], fy=k['fy']))
                            used.append(i)
                        row = dict(time_ms=float(timestamp), marker_id=pid, cameras=used,
                                   xyz_mm=None, status='insufficient_views_or_timing_or_motion',
                                   source_video_indices=selected,
                                   pose_stamp_ids=[frames[i][selected[i]].get('pose_stamp_id',default_pose_ids[i]) for i in range(3)],
                                   image_points=[positions[i].get(pid) for i in range(3)])
                        if len(observations) >= 2:
                            try:
                                row.update(xyz_mm=triangulate_observations(observations).tolist(), status='observed')
                            except ValueError as exc:
                                row['status'] = str(exc)
                        f.write(json.dumps(row)+'\n')
                    if index % 100 == 0:
                        self.progress.emit(f'已处理 {index} 帧')
            # Each camera was calibrated against the same physical target frame.
            # The local videos do not need to be merged to derive this rig geometry.
            relative = {}
            target_from_camera = [np.linalg.inv(np.asarray(p['T_camera_from_target_mm'], float))
                                  for p in poses]
            for i in range(3):
                relative[str(i)] = {
                    'camera_center_target_mm': target_from_camera[i][:3, 3].tolist(),
                    'camera_from_target': poses[i]['T_camera_from_target_mm'],
                    'camera_from_camera_0': (np.asarray(poses[i]['T_camera_from_target_mm']) @
                                             target_from_camera[0]).tolist(),
                }
            config = dict(sessions=list(map(str, self.sessions)), event_video_indices=self.anchors,
                          seeds=self.seeds, max_time_residual_ms=5,
                          rig_id=rig_ids[0],
                          sync_roles=sync_roles,
                          relative_camera_poses=relative,
                          imu_first_motion_ms=[v if np.isfinite(v) else None for v in motion],
                          note='No IMU translation correction; occluded tracks are invalid, not interpolated.')
            (self.output/'processing.json').write_text(json.dumps(config, indent=2), encoding='utf8')
            self.progress.emit('处理完成：'+str(output_file))
        except Exception as exc:
            logging.exception('postprocess failed')
            self.progress.emit('处理停止：'+str(exc))
        finally:
            for c in caps:
                c.release()


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('三机后处理：时间对齐、同名 Mark、3D 重建')
        self.resize(1380, 720)
        self.sessions = [None]*3
        self.seeds = [{}, {}, {}]
        self.views, self.first, self.second, self.labels = [], [], [], []
        self.job = None
        root = QtWidgets.QVBoxLayout()
        grid = QtWidgets.QGridLayout()
        for i in range(3):
            button = QtWidgets.QPushButton(f'选择机位 {i+1} 录像目录')
            button.clicked.connect(lambda checked=False, i=i: self.load(i))
            grid.addWidget(button, 0, i)
            v = ImageView()
            v.clicked.connect(lambda x, y, i=i: self.point(i, x, y))
            self.views.append(v)
            grid.addWidget(v, 1, i)
            a, b = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
            for spin in (a, b):
                spin.setRange(0, 10000000)
            a.valueChanged.connect(lambda value, i=i: self.preview(i, value))
            self.first.append(a)
            self.second.append(b)
            line = QtWidgets.QHBoxLayout()
            line.addWidget(QtWidgets.QLabel('共同事件1 / 2 帧索引'))
            line.addWidget(a)
            line.addWidget(b)
            grid.addLayout(line, 2, i)
            label = QtWidgets.QLabel('未载入')
            self.labels.append(label)
            grid.addWidget(label, 3, i)
        root.addLayout(grid, 1)
        line = QtWidgets.QHBoxLayout()
        self.id = QtWidgets.QLineEdit('M001')
        line.addWidget(QtWidgets.QLabel('同名面部点 ID：依次点击三幅图中的同一点'))
        line.addWidget(self.id)
        start = QtWidgets.QPushButton('选择输出目录并重建')
        start.clicked.connect(self.export)
        line.addWidget(start)
        root.addLayout(line)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        root.addWidget(self.log)
        self.log.appendPlainText('事件1需选择相同静止姿势的时刻，事件2选择末段共同可见事件。三机标定物必须保持不动。遮挡后不猜测点号；IMU 检出机位运动后该机位失效。')
        body = QtWidgets.QWidget()
        body.setLayout(root)
        self.setCentralWidget(body)

    def load(self, i):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, '选择包含 session.json 的录像目录')
        if folder:
            self.sessions[i] = Path(folder)
            self.preview(i, self.first[i].value())

    def preview(self, i, index):
        if self.sessions[i] is None:
            return
        cap = None
        try:
            records = [json.loads(s) for s in (self.sessions[i]/'frames.jsonl').read_text().splitlines()]
            rec = next(r for r in records if r['stream'] == 'ir1' and r['video_index'] == index)
            cap = cv2.VideoCapture(str(self.sessions[i]/rec['video_file']))
            gray = read_gray(cap, index)
            self.views[i].image = image_for_qt(gray)
            self.views[i].marks = {}
            self.seeds[i] = {}
            self.views[i].update()
            self.labels[i].setText(f'{rec["serial"]} | {rec["device_ts_ms"]:.1f} ms | 帧 {index}')
        except Exception as exc:
            self.log.appendPlainText(str(exc))
        finally:
            if cap:
                cap.release()

    def point(self, i, x, y):
        pid = self.id.text().strip()
        if pid:
            self.seeds[i][pid] = [x, y]
            self.views[i].marks = dict(self.seeds[i])
            self.views[i].update()

    def export(self):
        if self.job and self.job.isRunning():
            return
        if any(p is None for p in self.sessions):
            self.log.appendPlainText('请先载入三台录像')
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, '选择新的输出目录')
        if folder:
            anchors = [(a.value(), b.value()) for a, b in zip(self.first, self.second)]
            self.job = Export(list(self.sessions), anchors, [dict(s) for s in self.seeds], folder)
            self.job.progress.connect(self.log.appendPlainText)
            self.job.start()

    def closeEvent(self, event):
        if self.job and self.job.isRunning():
            self.log.appendPlainText('正在输出，请等待处理完成后关闭')
            event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
