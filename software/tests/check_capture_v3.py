import json
import os
import tempfile
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5 import QtWidgets

import capture_app as capture_app_module
from capture_app import Main, image_for_qt
from capture_runtime import CaptureWorker, SYNC_ROLE_MODES, configure_sync_mode
from calibration_math import solve_target_pose, triangulate_observations
from recording_io import SessionWriter


def run():
    app = QtWidgets.QApplication([])
    bgr = np.zeros((48, 64, 3), np.uint8)
    bgr[:, :, 2] = 255
    color = image_for_qt(bgr, bgr=True).pixelColor(20, 20)
    assert color.red() == 255 and color.blue() == 0
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)/'session'
        writer = SessionWriter(out, {'test': True, 'recording_mode': 'fast_lossless'})
        for number in range(12):
            for name in ('rgb', 'ir1', 'ir2', 'accel', 'gyro'):
                rec = dict(serial='test', stream=name, frame_number=number,
                           device_ts_ms=number*1000/30, timestamp_domain='hardware_clock',
                           host_monotonic_ns=number*33333333, host_wall_ns=number*33333333,
                           nominal_fps=30, x=1, y=2, z=3, units='test')
                image = bgr.copy() if name == 'rgb' else np.full((48, 64), number, np.uint8)
                assert writer.submit(rec, image if name in ('rgb', 'ir1', 'ir2') else None)
        writer.close()
        summary = json.loads((out/'summary.json').read_text())
        assert summary['complete'] and len(summary['received_sample_counts']) == 5
        assert len((out/'frames.jsonl').read_text().splitlines()) == 36
        assert len((out/'imu.csv').read_text().splitlines()) == 25
        for name in ('rgb', 'ir1', 'ir2'):
            cap = cv2.VideoCapture(str(out/('test_'+name+'.avi')))
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
            ok, frame = cap.read()
            assert ok
            if name == 'rgb':
                assert np.array_equal(frame, bgr)
            else:
                assert np.max(frame) == 0
            cap.release()

    class Sensor:
        def __init__(self, stuck=False):
            self.values = {rs.option.emitter_enabled: 1., rs.option.laser_power: 150.}
            self.stuck = stuck
        def supports(self, opt): return opt in self.values
        def set_option(self, opt, value):
            if not self.stuck: self.values[opt] = value
        def get_option(self, opt): return self.values[opt]
    worker = CaptureWorker({})
    good = Sensor()
    worker._emitter_off(good)
    assert all(v == 0 for v in good.values.values())
    worker_on = CaptureWorker({'laser_power': 0})
    worker_on._set_emitter(good, True)
    assert good.values[rs.option.emitter_enabled] == 1
    assert good.values[rs.option.laser_power] == 150
    worker_on._emitter_off(good)
    class SyncSensor:
        def __init__(self, supported=True):
            self.values = {rs.option.inter_cam_sync_mode: 0.}
            self.supported = supported
        def supports(self, opt): return self.supported and opt in self.values
        def set_option(self, opt, value): self.values[opt] = value
        def get_option(self, opt): return self.values[opt]
    sync_sensor = SyncSensor()
    assert configure_sync_mode(sync_sensor, 'master') == 1
    assert configure_sync_mode(sync_sensor, 'slave1') == 2
    assert configure_sync_mode(sync_sensor, 'slave2') == 2
    assert SYNC_ROLE_MODES['slave1'] == SYNC_ROLE_MODES['slave2']
    try:
        configure_sync_mode(SyncSensor(False), 'master')
        raise AssertionError('Unsupported sync mode must fail')
    except RuntimeError:
        pass
    try:
        worker._emitter_off(Sensor(True))
        raise AssertionError('Unverified emitter must fail')
    except RuntimeError:
        pass

    with tempfile.TemporaryDirectory() as role_dir:
        capture_app_module.ROLE_FILE = Path(role_dir)/'camera_role.json'
        win = Main({'host_id': 'host_b'})
        assert set(win.role_buttons) == {'master', 'slave1', 'slave2'}
        assert not win.start_button.isEnabled()
        win.select_sync_role('slave2')
        assert win.cfg['sync_mode'] == 2 and win.cfg['host_id'] == 'slave2'
        assert json.loads(capture_app_module.ROLE_FILE.read_text())['sync_role'] == 'slave2'
        win.show()
        app.processEvents()
        win.views['rgb'].image = image_for_qt(bgr, True)
        for name in ('ir1', 'ir2'):
            win.views[name].image = image_for_qt(np.full((720, 1280), 120, np.uint8))
        for name in win.info:
            win.info[name].setText('1280x720 | 29.9 FPS | 1234567.8 ms | #12345')
        app.processEvents()
        rect = win.views['ir1'].image_rect()
        assert rect.height() <= win.views['ir1'].height()
        assert all(label.height() == 23 for label in win.info.values())
        assert Path(win.path.text()).name == 'sessions'
        win.grab().save(str(Path(__file__).parent/'capture_v3_check.png'))
        win.close()
    xyz = np.array([p[1] for p in win.target])*.7
    R = cv2.Rodrigues(np.array([2.0, .15, .05]))[0]
    t = np.array([10., 20., 1000.])
    cp = (R@xyz.T).T+t
    normalized = cp[:, :2]/cp[:, 2:3]
    solved = solve_target_pose(xyz, normalized, 900, 900)
    assert np.allclose(solved['camera_center_target_mm'], -R.T@t, atol=.01)
    observations = []
    target_point = np.array([15., 25., 50.])
    for tx in (-250, 0, 250):
        T = np.eye(4)
        T[:3, 3] = [tx, 0, 1000]
        cp = T@np.r_[target_point, 1]
        observations.append(dict(T_camera_from_target_mm=T, normalized_xy=cp[:2]/cp[2], fx=900, fy=900))
    assert np.allclose(triangulate_observations(observations), target_point, atol=1e-6)
    try:
        solve_target_pose(xyz[:9], normalized[:9], 900, 900)
        raise AssertionError('Planar-only target must be rejected')
    except ValueError:
        pass
    print('PASS: Qt window, RGB channels, lossless video roundtrip, frame timestamps, IMU CSV, emitter verification, pose inversion, three-view triangulation')


if __name__ == '__main__':
    run()
