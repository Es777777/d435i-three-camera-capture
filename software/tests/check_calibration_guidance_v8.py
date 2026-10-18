import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2
import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets

from calibration_guidance import (VIEW_ORDERS, bright_markers, snap_marker, marker_xyz,
                                  build_calibration, predicted_pixels)
from calibration_wizard import CalibrationWizard, TargetReferenceDialog
from capture_app import Main
from stereo_geometry import project

ROOT = Path(__file__).resolve().parents[1]


def run():
    app = QtWidgets.QApplication([])
    target = json.loads((ROOT/'points_blender52_final.json').read_text(encoding='utf8'))['points']
    xyz = marker_xyz(target)
    assert len(xyz) == 15 and np.allclose(xyz['C01'], [-73.5, -2.95, 126])
    k = dict(width=1280, height=720, fx=780., fy=780., cx=640., cy=360.,
             coeffs=[0.]*5, distortion='distortion.none')
    rotation = cv2.Rodrigues(np.array([np.pi/2, 0., 0.]))[0]
    translation = np.array([0., 140., 800.])
    pixels = dict(zip(xyz, project(np.array(list(xyz.values()))@rotation.T+translation, k)))
    raw = np.random.default_rng(3).integers(15, 40, (720, 1280), dtype=np.uint8)
    for p in pixels.values():
        cv2.circle(raw, tuple(np.rint(p).astype(int)), 4, 245, -1)
    before = raw.copy()
    candidates = bright_markers(raw)
    assert len(candidates) == 15
    snapped, yes = snap_marker(pixels['C01']+[2., 1.], candidates)
    assert yes and np.linalg.norm(snapped-pixels['C01']) < 1
    assert not snap_marker((12., 10.), [(10., 10., 2.), (14., 10., 2.)])[1]
    huge = np.zeros((80, 80), np.uint8)
    huge[5:70, 5:70] = 255
    assert not bright_markers(huge)
    precise = {pid: tuple(pixels[pid]) for pid in VIEW_ORDERS['left'][:6]}
    pose = build_calibration(target, precise, k)
    assert np.allclose(pose['T_camera_from_target_mm'][:3], np.column_stack((rotation, translation)), atol=1e-3)
    expected = predicted_pixels(target, pose, k)
    assert all(np.allclose(expected[pid], p, atol=1e-3) for pid, p in pixels.items())
    try:
        build_calibration(target, {pid: pixels[pid] for pid in list(xyz)[:9]}, k)
        raise AssertionError('Front-only points must not calibrate')
    except ValueError:
        pass
    reversed_rotation = cv2.Rodrigues(np.array([-np.pi/2, 0., 0.]))[0]
    reversed_pixels = dict(zip(xyz, project(np.array(list(xyz.values()))@reversed_rotation.T+[0., -70., 800.], k)))
    try:
        build_calibration(target, {pid: reversed_pixels[pid] for pid in VIEW_ORDERS['left'][:6]}, k)
        raise AssertionError('A back-facing target pose must not be accepted')
    except ValueError:
        pass
    wizard = CalibrationWizard(target, raw, k, recapture=lambda: before+1)
    wizard.show()
    app.processEvents()
    assert wizard.current_id is None and wizard.save_button.isEnabled()
    assert not wizard.save_button.property('ready') and wizard.pose is None
    QtTest.QTest.mouseClick(wizard.save_button, QtCore.Qt.LeftButton)
    assert wizard.isVisible() and wizard.result_pose is None
    assert '还差 6 点' in wizard.last_action.text()
    wizard.view_mode.setCurrentIndex(1)
    assert wizard.current_id == 'C01'
    positions = wizard.diagram.point_positions()
    assert positions['C12'].y() < positions['C10'].y()
    assert positions['C15'].y() < positions['C13'].y()

    def click(pid):
        rect = wizard.scene.image_rect()
        p = pixels[pid]
        pos = QtCore.QPoint(round(rect.x()+p[0]/1280*rect.width()),
                           round(rect.y()+p[1]/720*rect.height()))
        QtTest.QTest.mouseClick(wizard.scene, QtCore.Qt.LeftButton, pos=pos)
        app.processEvents()

    for pid in VIEW_ORDERS['left'][:6]:
        assert wizard.current_id == pid
        click(pid)
        assert pid in wizard.observations
        if len(wizard.observations) == 5:
            QtTest.QTest.mouseClick(wizard.save_button, QtCore.Qt.LeftButton)
            assert wizard.isVisible() and wizard.result_pose is None
            assert '还差 1 点' in wizard.last_action.text()
    assert len(wizard.observations) == 6 and wizard.pose is not None, wizard.quality.text()
    assert wizard.save_button.isEnabled() and wizard.save_button.property('ready')
    assert wizard.current_id == 'C02' and 'C02' in wizard.scene.predictions
    click('C01')
    assert len(wizard.observations) == 6 and 'C02' not in wizard.observations
    wizard.undo()
    assert len(wizard.observations) == 5 and wizard.current_id == 'C10'
    assert wizard.save_button.isEnabled() and not wizard.save_button.property('ready')
    click('C10')
    assert wizard.save_button.isEnabled() and wizard.save_button.property('ready')
    wizard.skip()
    assert wizard.current_id == 'C04'
    wizard.undo()
    assert wizard.current_id == 'C02'
    assert np.array_equal(wizard.raw, before) and np.array_equal(raw, before)
    # Selection coordinates still refer to the full sensor image after zooming.
    wizard.scene.view_region = QtCore.QRectF(400, 300, 400, 225)
    rect = wizard.scene.image_rect()
    actual = wizard.scene.image_position(rect.center())
    assert np.allclose(actual, [600, 412.5])
    wizard.scene.reset_view()
    assert wizard.scene.confirmed == set(wizard.observations)
    (ROOT/'tests/artifacts').mkdir(exist_ok=True)
    wizard.grab().save(str(ROOT/'tests/artifacts/calibration_guidance_v8.png'))
    # An actual button click saves exactly the solution shown as passing; changing
    # display-only controls or clicking save must not rerun randomized PnP.
    checked_pose = wizard.pose
    with patch('calibration_wizard.build_calibration', side_effect=AssertionError('Unexpected re-solve')):
        wizard.threshold.setValue(211)
        QtTest.QTest.mouseClick(wizard.save_button, QtCore.Qt.LeftButton)
    assert wizard.result() == QtWidgets.QDialog.Accepted
    assert len(wizard.result_pose['point_ids']) == 6
    assert wizard.result_pose == checked_pose and wizard.result_pose is not checked_pose

    invalid = CalibrationWizard(target, raw, k)
    invalid.show()
    invalid.view_mode.setCurrentIndex(1)
    app.processEvents()
    with patch('calibration_wizard.build_calibration', wraps=build_calibration) as solver:
        invalid.observations = {pid: tuple(pixels[pid]) for pid in list(xyz)[:6]}
        invalid.update_state()
        assert solver.call_count == 1 and invalid.pose is None
        assert not invalid.save_button.property('ready')
        QtTest.QTest.mouseClick(invalid.save_button, QtCore.Qt.LeftButton)
        assert solver.call_count == 1 and invalid.result_pose is None and invalid.isVisible()
        assert '共面' in invalid.save_status.text() and '共面' in invalid.last_action.text()
        invalid.grab().save(str(ROOT/'tests/artifacts/calibration_not_ready_v8.png'))
        invalid.observations = dict(precise)
        invalid.update_state()
        assert solver.call_count == 2 and invalid.save_button.property('ready')
        # Even a coordinate edit made immediately before save invalidates the
        # cached solution. The previously passing pose must not be written.
        invalid.observations['C10'] = (-1., 0.)
        QtTest.QTest.mouseClick(invalid.save_button, QtCore.Qt.LeftButton)
        assert solver.call_count == 3 and invalid.result_pose is None and invalid.isVisible()
        assert not invalid.save_button.property('ready') and 'C10' in invalid.save_status.text()
        invalid.observations['C10'] = precise['C10']
        QtTest.QTest.mouseClick(invalid.save_button, QtCore.Qt.LeftButton)
        assert solver.call_count == 4 and invalid.result() == QtWidgets.QDialog.Accepted
    invalid.close()
    reference = TargetReferenceDialog(target)
    assert reference.findChild(QtWidgets.QTabWidget).count() == 2
    reference.close()
    with tempfile.TemporaryDirectory() as tmp:
        main = Main({})
        main.meta = dict(serial='demo', ir1_intrinsics=k)
        main.path.setText(tmp)
        main.rig_calibration = {'old': True}
        main.save_calibration_pose(dict(wizard.result_pose, serial='demo'))
        result = json.loads(next(Path(tmp).glob('calibration_demo_*.json')).read_text(encoding='utf8'))
        assert result['target'] == 'body70_single_mark54' and result['target_scale'] == .7
        assert result['marker_inset_mm'] == .55 and len(result['image_points']) == 6
        assert main.rig_calibration is None and not main.record_button.isEnabled()
        class Preview:
            def snapshot(self):
                record = dict(frame_number=1, width=1280, height=720, received_fps=30., device_ts_ms=0)
                return {'ir1': (raw, record), 'ir2': (raw, record)}
        main.worker = Preview()
        main.refresh()
        assert len(main.views['ir1'].candidates) == 15 and len(main.views['ir2'].candidates) == 15
        main.marker_highlight.setChecked(False)
        assert not main.views['ir1'].candidates and not main.views['ir2'].candidates
        main.worker = None
        main.close()
    wizard.take_new_frame()
    assert not wizard.observations and wizard.pose is None
    assert wizard.current_id == 'C01' and not wizard.save_button.property('ready')
    wizard.view_mode.setCurrentIndex(2)
    assert wizard.current_id == 'C01'
    wizard.close()
    print('PASS v8: original model coordinates and wing order, bright-point snap, ambiguity rejection, '
          'guided clicks, undo/skip, projected highlight, raw frames intact, save/rig invalidation, live IR highlights, '
          'responsive save button, visible rejection reasons, cached validated pose and stale-pose rejection')


if __name__ == '__main__':
    run()
