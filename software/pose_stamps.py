"""Immutable local target-pose snapshots referenced by individual video frames."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from calibration_settlement import calibration_digest, validate_local_calibration

SCHEMA = 'd435i_target_pose_stamps_v1'


def make_pose_stamp(calibration):
    pose = deepcopy(calibration)
    checked = validate_local_calibration(pose)
    return dict(pose_stamp_id=calibration_digest(pose), serial=pose['serial'],
                coordinate_frame='target', units='mm', reference_stream='ir1', source='fixed_target_calibration',
                validity_assumption='camera_and_target_remain_fixed',
                T_target_from_camera_mm=checked['T_target_from_camera'].tolist(),
                camera_center_target_mm=checked['T_target_from_camera'][:3, 3].tolist(),
                calibration=pose)


def read_pose_stamps(folder, metadata):
    folder = Path(folder).resolve()
    name = metadata.get('pose_stamps_file')
    if not name:
        stamp = make_pose_stamp(metadata['calibration'])
        return {stamp['pose_stamp_id']: stamp}
    path = (folder/name).resolve()
    if path.parent != folder:
        raise ValueError('位姿戳文件必须位于原始 session 目录')
    document = json.loads(path.read_text(encoding='utf8'))
    if document.get('schema') != SCHEMA:
        raise ValueError('不支持的位姿戳格式')
    stamps = document.get('stamps', {})
    if not stamps:
        raise ValueError('录像缺少本机位姿戳')
    for key, stamp in stamps.items():
        expected = make_pose_stamp(stamp['calibration'])
        # Derived matrix inverses can differ in the last bits across CPUs. The
        # calibration digest stays exact; derived geometry uses a small tolerance.
        derived = ('T_target_from_camera_mm', 'camera_center_target_mm')
        fields = {k: v for k, v in stamp.items() if k not in derived}
        expected_fields = {k: v for k, v in expected.items() if k not in derived}
        geometry_matches = all(
            np.asarray(stamp.get(name), float).shape == np.asarray(expected[name]).shape
            and np.allclose(stamp[name], expected[name], atol=1e-6, rtol=0)
            for name in derived)
        if (key != expected['pose_stamp_id'] or fields != expected_fields or not geometry_matches
                or stamp['serial'] != metadata['serial']):
            raise ValueError('录像位姿戳与标定内容或相机序列号不一致')
    default = metadata.get('pose_stamp_id')
    if default not in stamps or default != calibration_digest(metadata['calibration']):
        raise ValueError('录像默认位姿戳与本机标定不一致')
    return stamps


def motion_alarm(record):
    norm = float(np.linalg.norm([float(record[k]) for k in ('x', 'y', 'z')]))
    return ((record['stream'] == 'gyro' and norm > .05)
            or (record['stream'] == 'accel' and abs(norm-9.80665) > 1.))
