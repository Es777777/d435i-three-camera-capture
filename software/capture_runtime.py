import json
import logging
import queue
import shutil
import threading
import time
import traceback
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5 import QtCore
import pyrealsense2 as rs

from calibration_settlement import calibration_digest, verify_settlement_record, validate_local_calibration
from recording_io import SessionWriter
from audio_io import AudioRecorder
from sync_health import frame_health


SYNC_ROLE_MODES = {'master': 1, 'slave1': 2, 'slave2': 2}


def configure_sync_mode(sensor, role):
    if role not in SYNC_ROLE_MODES:
        raise RuntimeError('未选择有效同步角色')
    if not sensor.supports(rs.option.inter_cam_sync_mode):
        raise RuntimeError('当前 D435i 深度传感器不支持 inter_cam_sync_mode')
    mode = SYNC_ROLE_MODES[role]
    sensor.set_option(rs.option.inter_cam_sync_mode, mode)
    actual = int(round(sensor.get_option(rs.option.inter_cam_sync_mode)))
    if actual != mode:
        raise RuntimeError(f'设置同步模式失败：请求 {mode}，设备确认 {actual}')
    return mode


class CaptureWorker(QtCore.QThread):
    status = QtCore.pyqtSignal(str)
    ready = QtCore.pyqtSignal(object)
    recording = QtCore.pyqtSignal(bool, str)
    emitter_state = QtCore.pyqtSignal(bool)
    health = QtCore.pyqtSignal(object)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.commands = queue.Queue()
        self.lock = threading.RLock()
        self.latest = {}
        self.windows = {}
        self.writer = None
        self.metadata = {}
        self.pipe = None
        self.first = threading.Event()
        self.running = True
        self.last_sample = time.monotonic()
        self.callback_error = None
        self.audio = None
        self.health_samples = {}
        self.telemetry = queue.Queue(maxsize=20000)
        self.telemetry_file = None

    def snapshot(self):
        with self.lock:
            return dict(self.latest)

    def command(self, *args):
        self.commands.put(args)

    def recording_metadata(self, calibration, rig=None, options=None):
        """A checked local pose is sufficient; three-camera settlement is deferred."""
        validate_local_calibration(calibration, self.metadata)
        meta = deepcopy(self.metadata)
        meta.update(schema='d435_recording_v4', started_wall_ns=time.time_ns(),
                    started_host_monotonic_ns=time.monotonic_ns(),
                    calibration=deepcopy(calibration), fixed_target_assumed=True,
                    settlement_status='deferred_until_fusion')
        options = options or {}
        meta.update(recording_mode=options.get('recording_mode', self.cfg.get('recording_mode', 'balanced')),
                    video_buffer_mb=self.cfg.get('video_buffer_mb', 256))
        if rig:
            try:
                checked = verify_settlement_record(rig)
                entry = next(e for e in checked['cameras'] if e['serial'] == self.serial)
                if calibration_digest(entry['calibration']) != calibration_digest(calibration):
                    raise ValueError('本机标定已更新')
                meta.update(rig_calibration=deepcopy(rig), rig_id=rig['rig_id'])
            except (ValueError, KeyError, StopIteration):
                self.status.emit('已有三机结算与本机标定不匹配，将随录像保存本机位姿，汇总时重新结算')
        return meta

    def _emitter_off(self, sensor):
        self._set_emitter(sensor, False)

    def _read_emitter_enabled(self, sensor):
        if not sensor.supports(rs.option.emitter_enabled):
            return None
        return bool(round(sensor.get_option(rs.option.emitter_enabled)))

    def _set_emitter(self, sensor, enabled):
        if not sensor.supports(rs.option.emitter_enabled):
            raise RuntimeError('Cannot verify emitter status on this sensor')
        enabled = bool(enabled)
        if sensor.supports(rs.option.laser_power):
            power = float(self.cfg.get('laser_power', 150.0)) if enabled else 0.0
            if enabled and power <= 0:
                power = 150.0
            sensor.set_option(rs.option.laser_power, power)
        sensor.set_option(rs.option.emitter_enabled, 1 if enabled else 0)
        if self._read_emitter_enabled(sensor) != enabled:
            raise RuntimeError('Emitter state did not change')
        if (not enabled and sensor.supports(rs.option.laser_power)
                and sensor.get_option(rs.option.laser_power) != 0):
            raise RuntimeError('Laser power is not zero; capture stopped')

    def _callback(self, frame):
        try:
            if frame.is_frameset():
                for child in frame.as_frameset():
                    self._callback(child)
                return
            profile = frame.get_profile()
            st = profile.stream_type()
            name = ('rgb' if st == rs.stream.color else
                    'ir' + str(profile.stream_index()) if st == rs.stream.infrared else
                    'accel' if st == rs.stream.accel else 'gyro' if st == rs.stream.gyro else '')
            if not name:
                return
            now = time.monotonic_ns()
            rec = dict(serial=self.serial, stream=name, frame_number=int(frame.get_frame_number()),
                       device_ts_ms=float(frame.get_timestamp()),
                       timestamp_domain=str(frame.get_frame_timestamp_domain()),
                       host_monotonic_ns=now, host_wall_ns=time.time_ns())
            image = None
            if frame.is_motion_frame():
                xyz = frame.as_motion_frame().get_motion_data()
                rec.update(x=xyz.x, y=xyz.y, z=xyz.z,
                           units='m/s^2' if name == 'accel' else 'rad/s')
                self.telemetry.put_nowait(dict(rec))
            elif frame.is_video_frame():
                image = np.asanyarray(frame.get_data()).copy()
                rec.update(width=image.shape[1], height=image.shape[0], nominal_fps=profile.fps())
                for option, field in [(rs.frame_metadata_value.actual_exposure, 'exposure_us'),
                                      (rs.frame_metadata_value.gain_level, 'gain')]:
                    if frame.supports_frame_metadata(option):
                        rec[field] = frame.get_frame_metadata(option)
                with self.lock:
                    health_window = self.health_samples.setdefault(name, deque(maxlen=120))
                    health_window.append((rec['frame_number'], now, rec['device_ts_ms']))
                    window = self.windows.setdefault(name, deque(maxlen=60))
                    window.append((rec['frame_number'], now))
                    span = (window[-1][1]-window[0][1])/1e9
                    rec['received_fps'] = (len(window)-1)/span if span > 0 else 0.
                    self.latest[name] = (image, rec)
                    if all(k in self.latest for k in ('rgb', 'ir1', 'ir2')):
                        self.first.set()
                self.last_sample = time.monotonic()
            with self.lock:
                if self.writer:
                    self.writer.submit(rec, image)
        except Exception:
            self.callback_error = traceback.format_exc()

    def _finish_recording(self):
        with self.lock:
            writer, self.writer = self.writer, None
        if writer:
            self.status.emit('正在写完录像并检查视频解码，请稍候…')
            if self.audio:
                try:
                    self.audio.close()
                    if self.audio.error:
                        writer.error = 'Audio: '+self.audio.error
                except Exception as exc:
                    writer.error = 'Audio close: '+str(exc)
                self.audio = None
            writer.close()
            self.recording.emit(False, str(writer.root))
            self.status.emit(('录像不完整：' + writer.error) if writer.error else '已保存：' + str(writer.root))
            if not writer.error:
                summary = json.loads((writer.root/'summary.json').read_text(encoding='utf8'))
                for file in summary.get('video_files', []):
                    self.status.emit(f'{file["file"]}：{file["frames"]} 帧，'
                        f'{file["duration_seconds"]:.1f} 秒，{file["bytes"]/1e6:.1f} MB；首、中、尾帧回读通过')

    def run(self):
        try:
            telemetry_root = Path(__file__).resolve().parent/'telemetry'
            telemetry_root.mkdir(exist_ok=True)
            telemetry_path = telemetry_root/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.jsonl')
            self.telemetry_file = telemetry_path.open('w', encoding='utf8')
            self.status.emit('启动至关闭的 IMU/同步记录：'+str(telemetry_path))
            ctx = rs.context()
            devices = sorted(list(ctx.query_devices()), key=lambda d: d.get_info(rs.camera_info.serial_number))
            if not devices:
                raise RuntimeError('没有发现 RealSense')
            dev = devices[0]
            self.serial = dev.get_info(rs.camera_info.serial_number)
            usb = dev.get_info(rs.camera_info.usb_type_descriptor)
            self.status.emit(f'相机 {self.serial} | USB {usb}')
            logging.info('serial=%s USB=%s firmware=%s', self.serial, usb,
                         dev.get_info(rs.camera_info.firmware_version))
            if not usb.startswith('3'):
                raise RuntimeError(f'USB {usb} 不满足采集要求。请接 USB 3.x 数据线及接口。')
            dep = dev.first_depth_sensor()
            self.depth_sensor = dep
            self.emitter_enabled = bool(self.cfg.get('emitter_enabled', False))
            self._set_emitter(dep, self.emitter_enabled)
            self.emitter_state.emit(self.emitter_enabled)
            sync = int(self.cfg.get('sync_mode', -1))
            sync_role = self.cfg.get('sync_role')
            if sync not in (1, 2) or sync_role not in SYNC_ROLE_MODES or SYNC_ROLE_MODES[sync_role] != sync:
                raise RuntimeError('未选择有效硬件同步角色；请选择 Master、Slave 1 或 Slave 2')
            configure_sync_mode(dep, sync_role)
            self.status.emit(f'9-pin 同步角色：{sync_role}；硬件模式={sync}（1=Master，2=Slave）')
            for sensor in dev.query_sensors():
                if sensor.supports(rs.option.power_line_frequency):
                    sensor.set_option(rs.option.power_line_frequency, 1)  # 50 Hz mains.
            w, h = self.cfg.get('resolution', [1280, 720])
            fps = int(self.cfg.get('fps', 30))
            config = rs.config()
            config.enable_device(self.serial)
            for idx in (1, 2):
                config.enable_stream(rs.stream.infrared, idx, w, h, rs.format.y8, fps)
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            # Select motion rates from advertised profiles instead of assuming accel=200.
            for stream in (rs.stream.accel, rs.stream.gyro):
                rates = sorted({p.fps() for s in dev.query_sensors() for p in s.get_stream_profiles()
                                if p.stream_type() == stream and p.format() == rs.format.motion_xyz32f})
                if not rates:
                    raise RuntimeError('没有找到 IMU 流：' + str(stream))
                rate = min(rates, key=lambda n: abs(n-200))
                config.enable_stream(stream, rs.format.motion_xyz32f, rate)
                logging.info('IMU %s selected=%d supported=%s', stream, rate, rates)
            self.pipe = rs.pipeline(ctx)
            wrapper = rs.pipeline_wrapper(self.pipe)
            if not config.can_resolve(wrapper):
                raise RuntimeError(f'设备不支持当前流组合 {w}x{h}@{fps}；无法仅由此判定带宽不足。')
            profile = self.pipe.start(config, self._callback)
            self.depth_sensor = profile.get_device().first_depth_sensor()
            actual_sync = int(round(self.depth_sensor.get_option(rs.option.inter_cam_sync_mode)))
            if actual_sync != sync:
                raise RuntimeError(f'相机启动后同步模式改变：期望 {sync}，实际 {actual_sync}')
            self._set_emitter(self.depth_sensor, self.emitter_enabled)
            refs = {p.stream_name() + str(p.stream_index()): p for p in profile.get_streams()}
            ref = profile.get_stream(rs.stream.infrared, 1)
            streams = {}
            for p in profile.get_streams():
                key = str(p.stream_type()) + ':' + str(p.stream_index())
                row = {'fps': p.fps(), 'format': str(p.format())}
                if p.is_video_stream_profile():
                    intr = p.as_video_stream_profile().get_intrinsics()
                    ext = ref.get_extrinsics_to(p)
                    row.update(width=intr.width, height=intr.height, fx=intr.fx, fy=intr.fy,
                               cx=intr.ppx, cy=intr.ppy, coeffs=list(intr.coeffs),
                               distortion=str(intr.model), ir1_to_stream_rotation=list(ext.rotation),
                               ir1_to_stream_translation_m=list(ext.translation))
                streams[key] = row
            self.metadata = dict(schema='d435_recording_v3', serial=self.serial,
                                 host_id=self.cfg.get('host_id', 'host_b'), usb=usb,
                                 emitter_enabled=int(self.emitter_enabled),
                                 laser_power=float(self.depth_sensor.get_option(rs.option.laser_power))
                                 if self.depth_sensor.supports(rs.option.laser_power) else None,
                                 streams=streams,
                                 sync_role=sync_role, sync_mode=sync,
                                 timestamp_note='Host monotonic clocks are local, not cross-host synchronized.',
                                 ir1_intrinsics=streams['stream.infrared:1'])
            self.metadata['telemetry_file'] = str(telemetry_path)
            self.metadata['audio_requested'] = True
            self.metadata['ir1_intrinsics']['serial'] = self.serial
            self.ready.emit(self.metadata)
            self.status.emit(f'{w}x{h}@{fps} 三路已启动；IR emitter={int(self.emitter_enabled)}')
            last_health = 0.
            while self.running:
                while not self.telemetry.empty():
                    self.telemetry_file.write(json.dumps(self.telemetry.get_nowait())+'\n')
                if time.monotonic()-last_health >= 1:
                    with self.lock:
                        health = frame_health(self.health_samples, time.monotonic_ns(), fps)
                        if self.writer:
                            performance = self.writer.snapshot_stats()
                            health['writer_queue'] = performance['metadata_queued']+sum(s['queued'] for s in performance['streams'].values())
                            health['written_samples'] = performance['written_samples']
                            health['writer_performance'] = performance
                    # Filesystem queries can block on removable/network storage;
                    # never hold the RealSense callback lock while doing them.
                    if self.writer:
                        free = shutil.disk_usage(self.writer.root).free
                        health['disk_free_gb'] = free/1e9
                        if free < 512*1024*1024:
                            self.writer._fail('保存目录剩余空间不足 512 MB，已停止录像并写完已接收数据。', 'disk_space_low')
                    health.update(sync_role=sync_role, sync_mode=actual_sync,
                                  host_monotonic_ns=time.monotonic_ns(),
                                  audio_level=self.audio.level if self.audio else None)
                    self.health.emit(health)
                    self.telemetry_file.write(json.dumps(dict(event='health', **health))+'\n')
                    self.telemetry_file.flush()
                    last_health = time.monotonic()
                try:
                    command = self.commands.get(timeout=.1)
                except queue.Empty:
                    command = None
                if command:
                    if command[0] == 'quit':
                        self.running = False
                    elif command[0] == 'stop_record':
                        self._finish_recording()
                    elif command[0] == 'sync_event':
                        self.telemetry_file.write(json.dumps(dict(event='operator_sync_test',
                            label=str(command[1]), host_monotonic_ns=time.monotonic_ns()))+'\n')
                        self.telemetry_file.flush()
                        self.status.emit('同步测试事件已记入持续日志：'+str(command[1]))
                    elif command[0] == 'emitter' and self.writer is None:
                        try:
                            self._set_emitter(self.depth_sensor, bool(command[1]))
                            self.emitter_enabled = bool(command[1])
                            self.metadata['emitter_enabled'] = int(self.emitter_enabled)
                            self.metadata['laser_power'] = float(self.depth_sensor.get_option(rs.option.laser_power)) if self.depth_sensor.supports(rs.option.laser_power) else None
                            self.emitter_state.emit(self.emitter_enabled)
                            self.status.emit(f'IR emitter 已切换为 {int(self.emitter_enabled)}')
                        except Exception as exc:
                            actual = self._read_emitter_enabled(self.depth_sensor)
                            if actual is not None:
                                self.emitter_enabled = actual
                                self.metadata['emitter_enabled'] = int(actual)
                                self.emitter_state.emit(actual)
                            self.status.emit(f'IR emitter 切换失败: {exc}')
                    elif command[0] == 'record' and self.writer is None:
                        if not self.first.is_set():
                            self.status.emit('三路画面尚未到齐，未开始录像')
                            continue
                        try:
                            meta = self.recording_metadata(command[2], command[3] if len(command) > 3 else None,
                                                           command[4] if len(command) > 4 else None)
                        except Exception as exc:
                            self.status.emit(f'本机标定校验失败，录像未开始：{exc}')
                            continue
                        if self.emitter_enabled:
                            try:
                                self._set_emitter(self.depth_sensor, False)
                            except Exception as exc:
                                actual = self._read_emitter_enabled(self.depth_sensor)
                                if actual is not None:
                                    self.emitter_enabled = actual
                                    self.metadata['emitter_enabled'] = int(actual)
                                    self.emitter_state.emit(actual)
                                self.status.emit(f'无法确认 IR emitter 已关闭，录像未开始: {exc}')
                                continue
                            self.emitter_enabled = False
                            self.metadata['emitter_enabled'] = 0
                            if self.depth_sensor.supports(rs.option.laser_power):
                                self.metadata['laser_power'] = float(
                                    self.depth_sensor.get_option(rs.option.laser_power))
                            self.emitter_state.emit(False)
                            self.status.emit('录像前已自动关闭 IR emitter')
                        root = Path(command[1]) / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + self.serial)
                        # Emitter state may have changed after taking the pose snapshot.
                        meta['emitter_enabled'] = self.metadata['emitter_enabled']
                        meta['laser_power'] = self.metadata.get('laser_power')
                        try:
                            Path(command[1]).mkdir(parents=True, exist_ok=True)
                            if shutil.disk_usage(command[1]).free < 512*1024*1024:
                                raise OSError('保存目录剩余空间不足 512 MB')
                            writer = SessionWriter(root, meta)
                        except Exception as exc:
                            logging.exception('start recording failed')
                            self.status.emit('无法开始写入，请检查空间、路径和权限：'+str(exc))
                            continue
                        try:
                            self.audio = AudioRecorder(root, self.cfg.get('audio_device'),
                                                       self.cfg.get('audio_rate', 48000))
                            self.audio.start()
                        except Exception as exc:
                            writer.error = 'Microphone startup failed: '+str(exc)
                            if self.audio:
                                try:
                                    self.audio.close()
                                except Exception:
                                    logging.exception('microphone cleanup failed')
                                self.audio = None
                            writer.close()
                            self.status.emit('麦克风启动失败，录像未开始：'+str(exc))
                            continue
                        with self.lock:
                            self.writer = writer
                        self.recording.emit(True, str(root))
                        self.status.emit('正在录像：' + str(root))
                if self.callback_error:
                    raise RuntimeError(self.callback_error)
                if self.writer and self.writer.error:
                    self._finish_recording()
                if self.writer and self.audio and self.audio.error:
                    self._finish_recording()
                if self.first.is_set():
                    with self.lock:
                        stale = [name for name in ('rgb','ir1','ir2')
                                 if time.monotonic_ns()-self.health_samples[name][-1][1] > 10e9]
                    if stale:
                        raise RuntimeError('连续 10 秒未收到视频流：'+', '.join(stale))
                if time.monotonic()-self.last_sample > 10:
                    raise RuntimeError('连续 10 秒未收到视频，录像已停止；请检查连接或同步触发。')
        except Exception as exc:
            with self.lock:
                if self.writer:
                    self.writer.error = self.writer.error or str(exc)
            logging.exception('capture failed')
            self.status.emit('错误：' + str(exc))
        finally:
            if self.pipe and getattr(self, 'depth_sensor', None) and self.emitter_enabled:
                try:
                    self._set_emitter(self.depth_sensor, False)
                    self.emitter_enabled = False
                except Exception:
                    logging.exception('failed to disable emitter on shutdown')
            if self.pipe:
                try:
                    self.pipe.stop()
                except RuntimeError:
                    pass
            self._finish_recording()
            if self.telemetry_file:
                while not self.telemetry.empty():
                    self.telemetry_file.write(json.dumps(self.telemetry.get_nowait())+'\n')
                self.telemetry_file.close()
