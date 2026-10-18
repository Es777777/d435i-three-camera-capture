import copy
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from calibration_settlement import settle_calibration_files, verify_settlement_record
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt5 import QtWidgets
from calibration_settlement_gui import SettlementWindow


def run():
    app = QtWidgets.QApplication([])
    window = SettlementWindow()
    window.show()
    app.processEvents()
    assert not window.save_button.isEnabled()
    window.close()
    target_xyz = [[-10., 0., 0.], [10., 0., 0.], [0., 10., 0.],
                  [0., -10., 0.], [0., 0., 10.], [0., 0., -10.]]
    point_ids = [f'P{i+1}' for i in range(6)]
    intrinsics = dict(width=640, height=480, fx=500., fy=500., cx=320., cy=240.,
                      distortion='distortion.none', coeffs=[0.]*5)
    with tempfile.TemporaryDirectory() as directory:
        files = []
        for i, x in enumerate((-300., 0., 300.)):
            T_target_from_camera = np.eye(4)
            T_target_from_camera[:3, 3] = [x, -1000., 100.]
            T_camera_from_target = np.linalg.inv(T_target_from_camera)
            pose = dict(serial=f'serial-{i}', stream='ir1', target='test-target',
                        target_scale=1., marker_inset_mm=0., units='mm', intrinsics=intrinsics,
                        T_camera_from_target_mm=T_camera_from_target.tolist(),
                        T_target_from_camera_mm=T_target_from_camera.tolist(),
                        camera_center_target_mm=[x, -1000., 100.], point_ids=point_ids,
                        target_xyz_mm=target_xyz, per_point_error_px=[.2]*6,
                        undistorted_reprojection_rms_px=.2)
            path = Path(directory)/f'camera_{i}.json'
            path.write_text(json.dumps(pose), encoding='utf8')
            files.append(path)

        record = settle_calibration_files(files, target_stationary_confirmed=True,
                                          geometry_confirmed=True)
        assert record['validation_passed'], record['errors']
        assert len(record['pairwise_geometry']) == 3
        assert len(record['rig_id']) == 20
        verify_settlement_record(record)

        unconfirmed = settle_calibration_files(files, target_stationary_confirmed=False,
                                               geometry_confirmed=True)
        assert not unconfirmed['validation_passed']

        tampered = copy.deepcopy(record)
        tampered['pairwise_geometry'][0]['baseline_mm'] += 1
        try:
            verify_settlement_record(tampered)
            raise AssertionError('Tampered geometry must be rejected')
        except ValueError:
            pass
    print('PASS: three-camera settlement, common-target confirmation, rig ID, tamper detection')


if __name__ == '__main__':
    run()
