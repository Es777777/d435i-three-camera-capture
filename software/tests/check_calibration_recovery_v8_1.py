"""Reproduce a single bad pick and repair it without restarting calibration."""
import json
import os
from pathlib import Path
import tempfile
import zipfile

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2
import numpy as np
from PyQt5 import QtCore, QtTest, QtWidgets

from calibration_guidance import VIEW_ORDERS, bright_markers, build_calibration, marker_xyz, refine_marker
from calibration_math import CalibrationReprojectionError
from calibration_settlement import _validate_pose
from calibration_wizard import CalibrationWizard
from stereo_geometry import project

ROOT = Path(__file__).resolve().parents[1]


def run():
    app = QtWidgets.QApplication([])
    target = json.loads((ROOT/'points_blender52_final.json').read_text(encoding='utf8'))['points']
    coordinates = marker_xyz(target)
    k = dict(width=1280, height=720, fx=780., fy=780., cx=640., cy=360.,
             coeffs=[0.]*5, distortion='distortion.none')
    R = cv2.Rodrigues(np.array([np.pi/2, 0., 0.]))[0]
    pixels = dict(zip(coordinates, project(np.array(list(coordinates.values()))@R.T+[0., 140., 800.], k)))
    raw = np.full((720, 1280), 30, dtype=np.uint8)
    for point in pixels.values():
        cv2.circle(raw, tuple(np.rint(point).astype(int)), 4, 125, -1)
    before = raw.copy()
    assert not bright_markers(raw, 210), 'Fixed threshold must miss the dim test markers'
    for pid in VIEW_ORDERS['left']:
        center, method = refine_marker(raw, pixels[pid]+[3., 2.], [])
        assert method and np.linalg.norm(np.asarray(center)-pixels[pid]) < 1., (pid, center)
    assert refine_marker(np.full((100, 100), 90, np.uint8), (50., 50.), [])[1] is None
    assert refine_marker(np.full((100, 100), 255, np.uint8), (50., 50.), [])[1] is None
    ambiguous = np.full((100, 100), 30, np.uint8)
    for x in (44, 56):
        cv2.circle(ambiguous, (x, 50), 3, 125, -1)
    assert refine_marker(ambiguous, (50., 50.), [])[1] is None
    assert refine_marker(raw, (50., 50.), [(48., 50., 2.), (52., 50., 2.)])[1] is None

    observations = {pid: tuple(pixels[pid]) for pid in VIEW_ORDERS['left']}
    bad = dict(observations, C11=tuple(pixels['C11']+[8.4, 0.]))
    try:
        build_calibration(target, bad, k)
        raise AssertionError('One erroneous optional pick must not pass validation')
    except CalibrationReprojectionError as exc:
        assert exc.diagnostic_pose['diagnostic_only'] is True
        errors = dict(zip(exc.diagnostic_pose['point_ids'], exc.diagnostic_pose['per_point_error_px']))
        assert max(errors, key=errors.get) == 'C11' and errors['C11'] > 8.
        assert 'C11' in str(exc)
    marked_diagnostic = dict(build_calibration(target, observations, k), serial='demo', diagnostic_only=True)
    assert any('诊断' in error for error in _validate_pose(marked_diagnostic, 'test')['errors'])

    wizard = CalibrationWizard(target, raw, k)
    wizard.show()
    wizard.view_mode.setCurrentIndex(1)
    wizard.observations = dict(bad)
    wizard.update_state()
    app.processEvents()
    assert wizard.pose is None and wizard.diagnostic_pose is not None
    assert not wizard.save_button.property('ready') and 'C11' in wizard.save_status.text()
    assert wizard.error_table.item(0, 0).text() == 'C11'
    assert 'C11' in wizard.scene.error_marks and wizard.scene.reference_only
    QtTest.QTest.mouseClick(wizard.save_button, QtCore.Qt.LeftButton)
    assert wizard.result_pose is None and wizard.isVisible()
    artifacts = ROOT/'tests/artifacts'
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        file = Path(folder)/'diagnostic.zip'
        wizard.write_diagnostic(file)
        with zipfile.ZipFile(file) as archive:
            report = json.loads(archive.read('points_and_checks.json'))
            assert report['usable_as_calibration'] is False and report['checked_pose'] is None
            assert report['diagnostic_pose']['diagnostic_only'] is True
            image = cv2.imdecode(np.frombuffer(archive.read('ir1.png'), np.uint8), cv2.IMREAD_UNCHANGED)
            assert np.array_equal(image, before)
    QtTest.QTest.mouseClick(wizard.review_button, QtCore.Qt.LeftButton)
    assert wizard.current_id == 'C11' and wizard.scene.view_region is not None
    app.processEvents()
    wizard.grab().save(str(artifacts/'calibration_recovery_v8_1.png'))
    old = dict(wizard.observations)
    # Click the actual dim marker in the zoomed image, not its diagnostic projection.
    rect, source = wizard.scene.image_rect(), wizard.scene.source_rect()
    point = pixels['C11']
    pos = QtCore.QPoint(round(rect.x()+(point[0]-source.x())/source.width()*rect.width()),
                       round(rect.y()+(point[1]-source.y())/source.height()*rect.height()))
    QtTest.QTest.mouseClick(wizard.scene, QtCore.Qt.LeftButton, pos=pos)
    assert wizard.pose is not None and wizard.diagnostic_pose is None, wizard.validation_error
    assert wizard.save_button.property('ready') and not wizard.scene.reference_only
    assert np.linalg.norm(np.asarray(wizard.observations['C11'])-pixels['C11']) < 1.
    assert all(wizard.observations[pid] == old[pid] for pid in old if pid != 'C11')
    assert np.array_equal(wizard.raw, before)
    wizard.undo()
    assert wizard.pose is None and wizard.diagnostic_pose is not None
    QtTest.QTest.mouseClick(wizard.review_button, QtCore.Qt.LeftButton)
    QtTest.QTest.mouseClick(wizard.remove_button, QtCore.Qt.LeftButton)
    assert 'C11' not in wizard.observations and len(wizard.observations) == 11
    assert wizard.pose is not None and wizard.save_button.property('ready')
    wizard.undo()
    assert wizard.pose is None and len(wizard.observations) == 12
    wizard.focus_point('C11')
    wizard.remove_current()
    QtTest.QTest.mouseClick(wizard.save_button, QtCore.Qt.LeftButton)
    assert wizard.result() == QtWidgets.QDialog.Accepted
    assert len(wizard.result_pose['point_ids']) == 11 and 'C11' not in wizard.result_pose['point_ids']

    # Removing a required sixth point cannot bypass the six-point gate.
    small = CalibrationWizard(target, raw, k)
    small.view_mode.setCurrentIndex(1)
    small.observations = {pid: observations[pid] for pid in VIEW_ORDERS['left'][:6]}
    small.update_state()
    assert small.pose is not None
    small.focus_point('C10')
    small.remove_current()
    small.prepare_save()
    assert len(small.observations) == 5 and small.pose is None and small.result_pose is None
    small.close()
    wizard.close()
    print('PASS v8.1: dim-marker local centering and ambiguity rejection; bad optional pick identified; '
          'error table, red overlays, zoomed correction, removal/undo, six-point gate, '
          'diagnostic export and unchanged raw image')


if __name__ == '__main__':
    run()
