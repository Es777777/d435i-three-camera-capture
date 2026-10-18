"""Validate and settle three per-camera target-pose calibration files."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA = 'd435i_three_camera_rig_calibration_v1'
DEFERRED_SCHEMA = 'd435i_deferred_three_camera_calibration_v1'


def _canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(data.encode('utf8')).hexdigest()


def calibration_digest(value):
    return _canonical_hash(value)


def _validate_pose(pose, slot):
    errors = []
    if pose.get('diagnostic_only'):
        errors.append(f'{slot}: 诊断结果不能用于标定结算')
    serial = str(pose.get('serial', '')).strip()
    if not serial:
        errors.append(f'{slot}: 缺少相机序列号')
    if pose.get('stream') != 'ir1':
        errors.append(f'{slot}: 标定流不是 IR1')
    if not pose.get('target') or pose.get('units') != 'mm':
        errors.append(f'{slot}: 缺少标定物版本或单位不是 mm')

    try:
        T = np.asarray(pose['T_camera_from_target_mm'], dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError
        R = T[:3, :3]
        if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
            errors.append(f'{slot}: 位姿矩阵最后一行无效')
        if not np.allclose(R.T@R, np.eye(3), atol=1e-3) or abs(np.linalg.det(R)-1) > 1e-3:
            errors.append(f'{slot}: 旋转矩阵不是有效刚体旋转')
        inverse = np.linalg.inv(T)
        if 'T_target_from_camera_mm' in pose and not np.allclose(
                inverse, np.asarray(pose['T_target_from_camera_mm'], dtype=float), atol=1e-2):
            errors.append(f'{slot}: 正逆位姿矩阵不一致')
        if 'camera_center_target_mm' in pose and not np.allclose(
                inverse[:3, 3], np.asarray(pose['camera_center_target_mm'], dtype=float), atol=1.):
            errors.append(f'{slot}: 相机中心与位姿矩阵不一致')
    except Exception:
        errors.append(f'{slot}: 位姿矩阵缺失或格式错误')
        T = None
        inverse = None

    try:
        ids = list(pose['point_ids'])
        xyz = np.asarray(pose['target_xyz_mm'], dtype=float)
        errors_px = np.asarray(pose['per_point_error_px'], dtype=float)
        rms = float(pose['undistorted_reprojection_rms_px'])
        if len(ids) < 6 or len(set(ids)) != len(ids) or xyz.shape != (len(ids), 3):
            errors.append(f'{slot}: 标定点少于 6 个、重复或坐标数量不匹配')
        elif np.linalg.matrix_rank(xyz-xyz.mean(axis=0), tol=.01) < 3:
            errors.append(f'{slot}: 标定点共面，缺少有效侧面约束')
        if errors_px.shape != (len(ids),) or not np.all(np.isfinite(errors_px)):
            errors.append(f'{slot}: 单点重投影误差记录不完整')
            max_error = float('inf')
        else:
            max_error = float(errors_px.max())
            calculated_rms = float(np.sqrt(np.mean(errors_px**2)))
            if abs(calculated_rms-rms) > .05:
                errors.append(f'{slot}: 保存的 RMS 与单点误差不一致')
        if not np.isfinite(rms) or rms > 1.5:
            errors.append(f'{slot}: RMS {rms:.3f}px 超过 1.5px')
        if max_error > 3.0:
            errors.append(f'{slot}: 最大单点误差 {max_error:.3f}px 超过 3px')
    except Exception:
        errors.append(f'{slot}: 标定点或重投影误差数据缺失')
        ids, xyz, rms, max_error = [], np.empty((0, 3)), float('inf'), float('inf')

    try:
        intr = pose['intrinsics']
        if min(float(intr[k]) for k in ('fx', 'fy')) <= 0:
            errors.append(f'{slot}: 焦距内参无效')
        if min(int(intr[k]) for k in ('width', 'height')) <= 0:
            errors.append(f'{slot}: 图像尺寸无效')
    except Exception:
        errors.append(f'{slot}: IR1 内参缺失')

    return dict(slot=slot, serial=serial, T=T, T_target_from_camera=inverse,
                point_ids=ids, target_xyz_mm=xyz, rms_px=rms,
                max_error_px=max_error, errors=errors)


def validate_local_calibration(pose, metadata=None):
    """Check one local calibration independently of any three-camera settlement."""
    checked = _validate_pose(pose, '本机')
    if checked['errors']:
        raise ValueError('；'.join(checked['errors']))
    if metadata is not None:
        if pose.get('serial') != metadata.get('serial'):
            raise ValueError('本机标定的相机序列号不匹配')
        saved, current = pose['intrinsics'], metadata.get('ir1_intrinsics', {})
        for key in ('width', 'height', 'fx', 'fy', 'cx', 'cy'):
            if key not in current or abs(float(saved[key])-float(current[key])) > (0 if key in ('width', 'height') else 1e-3):
                raise ValueError(f'本机标定 IR1 内参 {key} 不匹配')
        if saved.get('distortion') != current.get('distortion') or not np.allclose(
                saved.get('coeffs', []), current.get('coeffs', []), atol=1e-6, rtol=0):
            raise ValueError('本机标定 IR1 畸变参数不匹配')
    return checked


def _analyze(entries, target_stationary_confirmed, geometry_confirmed, deferred=False):
    errors, warnings = [], []
    if len(entries) != 3:
        errors.append('必须正好导入三台相机的标定 JSON')
    checked = [_validate_pose(e['calibration'], e['slot']) for e in entries]
    for result in checked:
        errors.extend(result['errors'])
    serials = [r['serial'] for r in checked]
    if len(set(serials)) != len(serials):
        errors.append('三个标定 JSON 中存在重复相机序列号')
    if not deferred and not target_stationary_confirmed:
        errors.append('尚未确认三份 JSON 使用同一个未移动的标定物坐标系')
    if not deferred and not geometry_confirmed:
        errors.append('尚未核对计算出的机位中心、基线和朝向与实际支架一致')

    poses = [e['calibration'] for e in entries]
    if poses:
        reference = poses[0]
        for e in entries[1:]:
            pose = e['calibration']
            if reference.get('target') != pose.get('target'):
                errors.append(f'{e["slot"]}: 标定物版本不一致')
            for field in ('target_scale', 'marker_inset_mm'):
                if field not in reference or field not in pose or not np.isclose(
                        float(reference.get(field, np.nan)), float(pose.get(field, np.nan)), atol=1e-6):
                    errors.append(f'{e["slot"]}: 标定物缩放或贴片内缩参数不一致')
                    break
        point_maps = []
        for pose in poses:
            try:
                point_maps.append({pid: np.asarray(xyz, float) for pid, xyz in
                                   zip(pose['point_ids'], pose['target_xyz_mm'])})
            except Exception:
                point_maps.append({})
        for i in range(len(point_maps)):
            for j in range(i+1, len(point_maps)):
                common = set(point_maps[i]) & set(point_maps[j])
                for pid in common:
                    if not np.allclose(point_maps[i][pid], point_maps[j][pid], atol=.02):
                        errors.append(f'机位标定的 {pid} 三维坐标不一致；检查贴片内缩或标定模型')
                        break
                if len(common) < 3:
                    warnings.append(f'{entries[i]["slot"]} 与 {entries[j]["slot"]} 共同标定点少于 3 个')

    camera_centers = {}
    valid = all(r['T'] is not None and r['T_target_from_camera'] is not None
                and len(r['target_xyz_mm']) >= 6 for r in checked)
    if valid:
        target_points = np.concatenate([r['target_xyz_mm'] for r in checked if len(r['target_xyz_mm'])])
        target_center = np.mean(target_points, axis=0)
        for entry, result in zip(entries, checked):
            Ttc = result['T_target_from_camera']
            center = Ttc[:3, 3]
            axis = Ttc[:3, 2]
            toward_target = target_center-center
            if np.linalg.norm(toward_target) > 0:
                cosang = np.clip(np.dot(axis, toward_target) /
                                 (np.linalg.norm(axis)*np.linalg.norm(toward_target)), -1, 1)
                aim_error = float(np.degrees(np.arccos(cosang)))
            else:
                aim_error = float('nan')
            camera_centers[result['serial']] = dict(
                slot=entry['slot'], camera_center_target_mm=center.tolist(),
                optical_axis_target=axis.tolist(), aim_error_to_target_center_deg=aim_error,
                reprojection_rms_px=result['rms_px'], max_point_error_px=result['max_error_px'])
            if np.isfinite(aim_error) and aim_error > 45:
                warnings.append(f'{entry["slot"]} 光轴与标定物中心偏差 {aim_error:.1f}°，请检查点击点位及朝向')

    pairwise, relative = [], {}
    if valid:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = checked[i], checked[j]
                baseline = float(np.linalg.norm(a['T_target_from_camera'][:3, 3] -
                                                b['T_target_from_camera'][:3, 3]))
                relative_T = a['T'] @ b['T_target_from_camera']
                relative_angle = float(np.degrees(np.arccos(np.clip(
                    (np.trace(relative_T[:3, :3])-1)/2, -1, 1))))
                pairwise.append(dict(camera_a=a['serial'], camera_b=b['serial'],
                                     baseline_mm=baseline,
                                     relative_rotation_angle_deg=relative_angle,
                                     T_camera_a_from_camera_b_mm=relative_T.tolist()))
                if baseline < 80 or baseline > 2500:
                    warnings.append(f'{a["slot"]}-{b["slot"]} 基线 {baseline:.1f}mm 超出常见面部三机布置范围，请与实物核对')
        relative = {r['serial']: dict(slot=e['slot'],
                                      T_camera_from_target_mm=r['T'].tolist(),
                                      T_target_from_camera_mm=r['T_target_from_camera'].tolist())
                    for e, r in zip(entries, checked)}

    calibration_entries = []
    for entry in entries:
        pose = entry['calibration']
        calibration_entries.append(dict(
            slot=entry['slot'], serial=str(pose.get('serial', '')),
            source_file=entry.get('source_file', ''),
            calibration_sha256=_canonical_hash(pose), calibration=pose))
    id_seed = [{'slot': e['slot'], 'serial': e['serial'], 'calibration_sha256': e['calibration_sha256']}
               for e in sorted(calibration_entries, key=lambda x: x['serial'])]
    rig_id = _canonical_hash(id_seed)[:20]
    record = dict(schema=DEFERRED_SCHEMA if deferred else SCHEMA, rig_id=rig_id,
                  settled_at_utc=datetime.now(timezone.utc).isoformat(),
                  operator_confirmed_same_target_pose=bool(target_stationary_confirmed),
                  operator_confirmed_geometry=bool(geometry_confirmed),
                  validation_passed=not errors, errors=errors, warnings=warnings,
                  cameras=calibration_entries, camera_geometry=camera_centers,
                  pairwise_geometry=pairwise, relative_camera_poses=relative,
                  target_frame={'units': 'mm', 'target': poses[0].get('target') if poses else None,
                                'scale': poses[0].get('target_scale') if poses else None,
                                'marker_inset_mm': poses[0].get('marker_inset_mm') if poses else None})
    if deferred:
        record.update(settlement_source='recorded_frame_poses', fixed_target_assumed=True,
                      geometry_reviewed_by_operator=False)
    return record


def settle_calibration_files(files, target_stationary_confirmed=False, geometry_confirmed=False):
    if len(files) != 3:
        raise ValueError('请提供三份相机标定 JSON')
    entries = []
    for index, path in enumerate(files):
        with open(path, 'r', encoding='utf8') as f:
            pose = json.load(f)
        entries.append(dict(slot=f'camera_{index+1}', source_file=Path(path).name,
                            calibration=pose))
    return _analyze(entries, target_stationary_confirmed, geometry_confirmed)


def settle_recorded_calibrations(poses):
    """Settle recorded poses at fusion time, assuming the same fixed physical board.

    This checks geometry and file consistency; it does not claim an operator has
    reviewed the layout or that hardware exposure synchronization was verified.
    """
    if len(poses) != 3:
        raise ValueError('录后结算需要三个机位的本机位姿')
    entries = [dict(slot=f'camera_{i+1}', source_file='recorded_pose', calibration=p)
               for i, p in enumerate(poses)]
    record = _analyze(entries, False, False, deferred=True)
    if not record['validation_passed']:
        raise ValueError('录后标定结算失败：'+'；'.join(record['errors']))
    return record


def verify_settlement_record(record):
    deferred = record.get('schema') == DEFERRED_SCHEMA
    if record.get('schema') not in (SCHEMA, DEFERRED_SCHEMA):
        raise ValueError('不是受支持的三机标定结算文件')
    if deferred and (not record.get('fixed_target_assumed') or record.get('settlement_source') != 'recorded_frame_poses'):
        raise ValueError('录后结算缺少固定标定板坐标系约定')
    if not deferred and not record.get('operator_confirmed_same_target_pose'):
        raise ValueError('结算文件没有确认三份标定使用同一个固定坐标系')
    if not deferred and not record.get('operator_confirmed_geometry'):
        raise ValueError('结算文件没有确认计算机位关系与实物布局一致')
    entries = record.get('cameras', [])
    rebuilt = _analyze(entries, not deferred, not deferred, deferred=deferred)
    if rebuilt['errors'] or not rebuilt['validation_passed']:
        raise ValueError('三机标定结算文件校验失败：'+'；'.join(rebuilt['errors']))
    if rebuilt['rig_id'] != record.get('rig_id'):
        raise ValueError('结算文件中的标定内容与 rig ID 不一致')
    if not record.get('validation_passed'):
        raise ValueError('三机标定结算没有通过验收')
    for field in ('camera_geometry', 'pairwise_geometry', 'relative_camera_poses', 'target_frame'):
        if _canonical_hash(record.get(field)) != _canonical_hash(rebuilt.get(field)):
            raise ValueError(f'结算文件的 {field} 与三份标定 JSON 计算结果不一致')
    return rebuilt
