import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PyQt5 import QtWidgets
from postprocess_gui import Export, Window
from calibration_settlement import settle_calibration_files
from recording_io import SessionWriter

app = QtWidgets.QApplication([])
window = Window()
window.show()
app.processEvents()
with tempfile.TemporaryDirectory() as directory:
    folders, seeds = [], []
    xyz = np.array([0., 0., 1000.])
    target_xyz = [[-10., 0., 0.], [10., 0., 0.], [0., 10., 0.],
                  [0., -10., 0.], [0., 0., 10.], [0., 0., -10.]]
    point_ids = [f'P{i+1}' for i in range(6)]
    calibration_files = []
    for i, tx in enumerate((-100., 0., 100.)):
        folder = Path(directory)/str(i)
        T = np.eye(4)
        T[0, 3] = tx
        k = dict(width=640, height=480, fx=500., fy=500., cx=320., cy=240.,
                 distortion='distortion.none', coeffs=[0.]*5)
        pose = dict(serial=str(i), stream='ir1', target='synthetic_target', target_scale=1.,
                    marker_inset_mm=0., units='mm', intrinsics=k,
                    T_camera_from_target_mm=T.tolist(), T_target_from_camera_mm=np.linalg.inv(T).tolist(),
                    camera_center_target_mm=np.linalg.inv(T)[:3, 3].tolist(),
                    point_ids=point_ids, target_xyz_mm=target_xyz,
                    per_point_error_px=[0.]*6, undistorted_reprojection_rms_px=0.)
        cal_file = Path(directory)/f'calibration_{i}.json'
        cal_file.write_text(json.dumps(pose), encoding='utf8')
        calibration_files.append(cal_file)
        pixel = [320+tx*.5, 240.]
        folders.append(folder)
        seeds.append({'M001': pixel})
    rig = settle_calibration_files(calibration_files, True, True)
    assert rig['validation_passed'], rig
    for i, (tx, folder) in enumerate(zip((-100., 0., 100.),
                                          [Path(directory)/str(i) for i in range(3)])):
        pose = json.loads(calibration_files[i].read_text(encoding='utf8'))
        role = ('master', 'slave1', 'slave2')[i]
        writer = SessionWriter(folder, dict(serial=str(i), sync_role=role,
                                             sync_mode=1 if role == 'master' else 2,
                                             calibration=pose))
        pixel = seeds[i]['M001']
        for n in range(12):
            image = np.zeros((480, 640), np.uint8)
            cv2.circle(image, tuple(map(int, pixel)), 3, 255, -1)
            rec = dict(serial=str(i), stream='ir1', frame_number=n, device_ts_ms=1000*i+n*1000/30,
                       timestamp_domain='hardware_clock', host_monotonic_ns=n, host_wall_ns=n, nominal_fps=30)
            writer.submit(rec, image)
        writer.close()
    out = Path(directory)/'output'
    job = Export(folders, [(0, 11)]*3, seeds, out)
    job.progress.connect(print)
    job.run()
    rows = [json.loads(s) for s in (out/'markers_3d.jsonl').read_text().splitlines()]
    assert len(rows) == 12 and all(r['status'] == 'observed' for r in rows), rows
    assert all(np.allclose(r['xyz_mm'], xyz, atol=1e-3) for r in rows)
window.close()
print('PASS: three recordings -> affine time alignment -> tracking -> 12 correct 3D samples')
