"""Hardware-free regression checks for stop/save and recoverable audio status.

Uses the actual Qt control loop, writers and FFmpeg, with a simulated camera and
microphone. This does not validate RealSense hardware or Windows drivers.
"""
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import numpy as np
from PyQt5 import QtCore, QtWidgets

# The entire camera boundary is simulated, even on hosts with the SDK installed.
fake_rs = MagicMock()
for name, names in {
    'stream': ('color', 'infrared', 'accel', 'gyro'),
    'option': ('emitter_enabled', 'laser_power', 'inter_cam_sync_mode', 'power_line_frequency'),
    'camera_info': ('serial_number', 'usb_type_descriptor', 'firmware_version'),
}.items():
    setattr(fake_rs, name, SimpleNamespace(**{n: name+'.'+n for n in names}))
sys.modules['pyrealsense2'] = fake_rs
import capture_runtime
import capture_app
from audio_io import AudioRecorder
from recording_io import SessionWriter
from video_encoding import FFmpegWriter


def spin(app, predicate, seconds=5):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(.01)
    import faulthandler
    faulthandler.dump_traceback()
    raise AssertionError('Timed out waiting for test condition')


class Microphone(AudioRecorder):
    status_warning = True

    def start(self):
        self.thread = threading.Thread(target=self._write)
        self.thread.start()
        clock = SimpleNamespace(inputBufferAdcTime=1., currentTime=1.01)
        data = np.zeros(1024, np.int16).tobytes()
        self._callback(data, 1024, clock, 'input overflow' if self.status_warning else '')
        self._callback(data, 1024, clock, '')
        assert self.error is None and self.samples == 2048


def setup_camera(worker):
    sensor = MagicMock()
    options = {fake_rs.option.emitter_enabled: 0., fake_rs.option.laser_power: 0.,
               fake_rs.option.inter_cam_sync_mode: 1.}
    sensor.supports.side_effect = lambda opt: opt in options
    sensor.get_option.side_effect = options.__getitem__
    sensor.set_option.side_effect = options.__setitem__
    dev = MagicMock()
    dev.get_info.side_effect = lambda key: {fake_rs.camera_info.serial_number: 'demo',
        fake_rs.camera_info.usb_type_descriptor: '3.2', fake_rs.camera_info.firmware_version: 'test'}[key]
    dev.first_depth_sensor.return_value = sensor
    dev.query_sensors.return_value = [sensor]
    streams = []
    for st, index in ((fake_rs.stream.color, 0), (fake_rs.stream.infrared, 1),
                      (fake_rs.stream.infrared, 2), (fake_rs.stream.accel, 0), (fake_rs.stream.gyro, 0)):
        p = MagicMock()
        p.stream_type.return_value, p.stream_index.return_value = st, index
        p.fps.return_value = 30 if st in (fake_rs.stream.color, fake_rs.stream.infrared) else 200
        p.is_video_stream_profile.return_value = p.fps() == 30
        p.format.return_value = fake_rs.format.motion_xyz32f
        p.as_video_stream_profile.return_value.get_intrinsics.return_value = SimpleNamespace(
            width=64, height=48, fx=60., fy=60., ppx=32., ppy=24., coeffs=[0.]*5, model='none')
        p.get_extrinsics_to.return_value = SimpleNamespace(rotation=np.eye(3).ravel(), translation=[0.,0.,0.])
        streams.append(p)
    sensor.get_stream_profiles.return_value = streams
    profile = MagicMock()
    profile.get_device.return_value = dev
    profile.get_streams.return_value = streams
    profile.get_stream.return_value = streams[1]
    fake_rs.context.return_value.query_devices.return_value = [dev]
    stopped = threading.Event()
    pipe = MagicMock()
    def produce(callback):
        n = 0
        while not stopped.is_set():
            for p in streams:
                frame = MagicMock()
                frame.is_frameset.return_value = False
                frame.get_profile.return_value = p
                frame.get_frame_number.return_value = n
                frame.get_timestamp.return_value = n*33.
                video = p.is_video_stream_profile()
                frame.is_motion_frame.return_value = not video
                frame.is_video_frame.return_value = video
                frame.supports_frame_metadata.return_value = False
                frame.as_motion_frame.return_value.get_motion_data.return_value = SimpleNamespace(x=0., y=0., z=0.)
                frame.get_data.return_value = np.zeros((48,64,3) if p.stream_type() == fake_rs.stream.color else (48,64), np.uint8)
                callback(frame)
            n += 1
            stopped.wait(.02)
    def start(config, callback):
        pipe.producer = threading.Thread(target=produce, args=(callback,))
        pipe.producer.start()
        return profile
    def stop():
        stopped.set()
        pipe.producer.join(3)
    pipe.start.side_effect, pipe.stop.side_effect = start, stop
    fake_rs.pipeline.return_value = pipe
    return pipe


def run():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        worker = capture_runtime.CaptureWorker({'sync_role':'master', 'sync_mode':1, 'resolution':[64,48]})
        worker.recording_metadata = lambda *args: {'serial':'demo', 'recording_mode':'balanced'}
        pipe = setup_camera(worker)
        win = capture_app.Main({'sync_role':'master'})
        win.worker = worker
        worker.ready.connect(win.camera_ready)
        worker.recording.connect(win.recording_changed)
        worker.saving.connect(win.recording_save_started)
        worker.status.connect(win.msg)
        worker.finished.connect(win.camera_finished)
        entered, gate = threading.Event(), threading.Event()
        original_close = SessionWriter.close
        def slow_close(writer):
            entered.set()
            if not gate.wait(8):
                raise TimeoutError('Test did not release save gate')
            original_close(writer)
        try:
            with patch.object(capture_runtime, 'rs', fake_rs), \
                    patch.object(capture_runtime, 'AudioRecorder', Microphone), \
                    patch.object(SessionWriter, 'close', slow_close):
                worker.start()
                spin(app, lambda: win.meta is not None and worker.first.is_set())
                worker.command('record', str(root), None)
                spin(app, lambda: win.recording_active)
                writer = worker.writer
                spin(app, lambda: writer.accepted.get('demo_rgb', 0) >= 5)
                # An overflow followed by valid audio must not stop the video.
                assert worker.writer is writer and worker.audio.status_count == 1
                win.record_stop.click()
                assert win.recording_saving and not win.record_stop.isEnabled()
                assert '保存' in win.state.text()
                assert worker.stop_requested.is_set()
                spin(app, entered.is_set)
                assert worker.writer is None and worker.finalizer.is_alive()
                accepted = dict(writer.accepted)
                first_preview = worker.snapshot()['rgb'][1]['frame_number']
                worker.command('stop_record')
                worker.command('record', str(root), None)  # Cannot restart while saving.
                spin(app, lambda: worker.snapshot()['rgb'][1]['frame_number'] > first_preview+4)
                assert writer.accepted == accepted and worker.writer is None
                assert worker.callback_error is None and worker.telemetry.qsize() < 100
                gate.set()
                spin(app, lambda: not win.recording_active)
                assert not win.recording_saving and win.record_stop.text() == '停止录像并保存'
                summary = json.loads((writer.root/'summary.json').read_text())
                assert summary['stop_reason'] == 'manual_stop'
                assert summary['error_category'] == 'audio_discontinuity'
                assert summary['audio_status']['status_count'] == 1 and not summary['complete']
                assert summary['accepted_sample_counts'] == summary['received_sample_counts']
                assert len(summary['video_files']) == 3
                assert json.loads((writer.root/'audio.json').read_text())['samples'] == 2048
                # Start another session and close the camera while finalizing it.
                entered.clear(); gate.clear()
                Microphone.status_warning = False
                worker.command('record', str(root), None)
                spin(app, lambda: win.recording_active)
                spin(app, lambda: worker.writer.accepted.get('demo_rgb', 0) >= 3)
                second_writer = worker.writer
                win.record_stop.click()
                spin(app, entered.is_set)
                worker.command('quit')
                spin(app, lambda: pipe.stop.called)
                assert worker.isRunning(), 'Camera shutdown must wait for save completion'
                gate.set()
                spin(app, lambda: not worker.isRunning())
                app.processEvents()
                assert win.worker is None and win.path.isEnabled()
                second_summary = json.loads((second_writer.root/'summary.json').read_text())
                assert second_summary['complete'] and second_summary['audio_status']['complete']
        finally:
            gate.set()
            worker.command('quit')
            worker.wait(10000)
            win.worker = None
            win.close()

        # Buffer exhaustion is still fatal; only recoverable input status changed.
        audio = AudioRecorder(root)
        audio.queue = queue.Queue(maxsize=1)
        clock = SimpleNamespace(inputBufferAdcTime=1., currentTime=1.01)
        for _ in range(2):
            audio._callback(bytes(2048), 1024, clock, '')
        assert audio.error == 'Audio writer queue full'

        # A producer which never becomes empty must still yield to commands.
        class ContinuousQueue:
            def get_nowait(self): return {'event':'imu'}
        worker.telemetry = ContinuousQueue()
        worker.telemetry_file = io.StringIO()
        before = time.monotonic()
        worker._drain_telemetry()
        assert time.monotonic()-before < .5
        assert 0 < len(worker.telemetry_file.getvalue().splitlines()) <= 1000

        # A failed save must still unlock the GUI and clearly report failure.
        failing = SessionWriter(root/'failed', {'recording_mode':'balanced'})
        worker.writer = failing
        with patch.object(failing, 'close', side_effect=OSError('disk disconnected')), \
                patch('capture_runtime.logging.exception'):
            worker._finish_recording()
            worker.finalizer.join(3)
            worker._poll_recording_finish()
        assert worker.finalizer is None and failing.error_category == 'recording_finalize_failed'
        failing.close()  # Clean up real writer threads after the injected failure.

        # Real child process with a full stdin pipe: watchdog must break the hang.
        popen = subprocess.Popen
        def stalled_encoder(command, **kwargs):
            return popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        with patch('video_encoding.subprocess.Popen', stalled_encoder):
            encoder = FFmpegWriter(root/'stalled.mp4', 'H264_RGB', 30, (720,1280,3), write_timeout=.5)
            before = time.monotonic()
            try:
                encoder.write(np.zeros((720,1280,3), np.uint8))
                raise AssertionError('Stalled encoder was not detected')
            except RuntimeError as exc:
                assert 'stalled' in str(exc)
            finally:
                try:
                    encoder.release()
                except RuntimeError:
                    pass
            assert time.monotonic()-before < 5
            assert encoder.process.poll() is not None and not encoder.watchdog.is_alive()
    print('PASS: actual Qt stop button, async save with live preview/telemetry, repeated stop/restart, '
          'quit during save, audio overflow continuation and integrity flags, save failure, encoder pipe timeout')


if __name__ == '__main__':
    run()
