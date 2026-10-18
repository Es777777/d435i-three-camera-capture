"""Geometry and conservative bright-marker assistance for the existing target."""
import cv2
import numpy as np
import pyrealsense2 as rs

from calibration_math import CalibrationReprojectionError, solve_target_pose
from stereo_geometry import project

TARGET_NAME = 'body70_single_mark54'
TARGET_SCALE = .70
DEFAULT_INSET_MM = .55
VIEW_ORDERS = {
    'left': ['C01', 'C03', 'C07', 'C09', 'C12', 'C10', 'C02', 'C04', 'C05', 'C06', 'C08', 'C11'],
    'right': ['C01', 'C03', 'C07', 'C09', 'C15', 'C13', 'C02', 'C04', 'C05', 'C06', 'C08', 'C14'],
    'both': ['C01', 'C03', 'C07', 'C09', 'C12', 'C10', 'C15', 'C13', 'C02', 'C04', 'C05', 'C06', 'C08', 'C11', 'C14'],
}


def marker_xyz(target, inset=DEFAULT_INSET_MM):
    return {pid: np.asarray(position, float)*TARGET_SCALE
            - np.asarray(normal, float)/np.linalg.norm(normal)*inset
            for pid, position, normal, side in target}


def bright_markers(image, threshold=210):
    """Candidates only: brightness never establishes the physical point identity."""
    gray = np.asarray(image)
    if gray.ndim != 2:
        return []
    count, labels, stats, centers = cv2.connectedComponentsWithStats((gray >= threshold).astype('uint8'))
    spots = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if not 3 <= area <= 900 or max(w, h) > 4*min(w, h):
            continue
        if x == 0 or y == 0 or x+w >= gray.shape[1] or y+h >= gray.shape[0]:
            continue
        mask = labels[y:y+h, x:x+w] == i
        weights = np.maximum(gray[y:y+h, x:x+w].astype(float)-threshold+1, 0)*mask
        yy, xx = np.indices(weights.shape)
        spots.append((float(x+(xx*weights).sum()/weights.sum()),
                      float(y+(yy*weights).sum()/weights.sum()), float(max(w, h)/2)))
    # Reject a flooded/noisy frame rather than drawing hundreds of competing marks.
    return spots if len(spots) <= 120 else []


def snap_marker(point, candidates, radius=16.):
    distances = sorted((np.hypot(x-point[0], y-point[1]), x, y) for x, y, r in candidates)
    if not distances or distances[0][0] > radius:
        return tuple(map(float, point)), False
    if len(distances) > 1 and distances[1][0] <= radius and distances[1][0]-distances[0][0] < 3:
        return tuple(map(float, point)), False
    return (distances[0][1], distances[0][2]), True


def refine_marker(image, point, candidates, radius=16.):
    """Snap to a unique nearby blob, including dim markers on a local background.

    No pose prediction is used here: pixels must independently support the center.
    Ambiguous global candidates remain ambiguous rather than falling back to a
    different threshold that could silently select the other marker.
    """
    snapped, ok = snap_marker(point, candidates, radius)
    if ok:
        return snapped, '亮点中心'
    if any(np.hypot(x-point[0], y-point[1]) <= radius for x, y, r in candidates):
        return tuple(map(float, point)), None
    gray = np.asarray(image)
    x, y = map(float, point)
    if gray.ndim != 2 or not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
        return (x, y), None
    half = int(np.ceil(radius))+10
    x0, y0 = max(0, int(x)-half), max(0, int(y)-half)
    patch = gray[y0:min(gray.shape[0], int(y)+half+1),
                 x0:min(gray.shape[1], int(x)+half+1)]
    if patch.size < 25:
        return (x, y), None
    background = float(np.median(patch))
    mad = float(np.median(np.abs(patch.astype(float)-background)))
    threshold = background+max(12., 6.*mad)
    if threshold > 254:
        return (x, y), None
    local = [(cx+x0, cy+y0, r) for cx, cy, r in bright_markers(patch, threshold)]
    snapped, ok = snap_marker(point, local, radius)
    return (snapped, '局部亮点中心') if ok else ((x, y), None)


def build_calibration(target, observations, intrinsics, inset=DEFAULT_INSET_MM):
    intr = rs.intrinsics()
    intr.width, intr.height = intrinsics['width'], intrinsics['height']
    intr.fx, intr.fy = intrinsics['fx'], intrinsics['fy']
    intr.ppx, intr.ppy = intrinsics['cx'], intrinsics['cy']
    intr.coeffs = intrinsics['coeffs']
    intr.model = getattr(rs.distortion, intrinsics['distortion'].split('.')[-1])
    coordinates = marker_xyz(target, inset)
    ids = [pid for pid, *rest in target if pid in observations]
    xyz, normalized = [], []
    for pid in ids:
        pixel = np.asarray(observations[pid], float)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError('点位坐标无效：'+pid)
        if not (0 <= pixel[0] < intr.width and 0 <= pixel[1] < intr.height):
            raise ValueError('点位超出 IR1 画面：'+pid)
        ray = rs.rs2_deproject_pixel_to_point(intr, pixel.tolist(), 1.)
        xyz.append(coordinates[pid])
        normalized.append([ray[0]/ray[2], ray[1]/ray[2]])
    rejected = False
    try:
        pose = solve_target_pose(xyz, normalized, intr.fx, intr.fy)
    except CalibrationReprojectionError as exc:
        pose = exc.diagnostic_pose
        rejected = True
    transform = np.asarray(pose['T_camera_from_target_mm'])
    for pid, position, normal, side in target:
        if pid not in observations:
            continue
        cp = transform[:3, :3]@coordinates[pid]+transform[:3, 3]
        facing = transform[:3, :3]@np.asarray(normal, float)
        if np.dot(cp, facing) >= 0:
            raise ValueError('点号可能左右或上下反了：计算出的贴片背对相机，请按模型正面核对点号')
    pose.update(stream='ir1', intrinsics=dict(intrinsics), units='mm', target=TARGET_NAME,
                target_scale=TARGET_SCALE, marker_inset_mm=float(inset), point_ids=ids,
                target_xyz_mm=np.asarray(xyz).tolist(), image_points=[list(observations[p]) for p in ids])
    if rejected:
        ranked = sorted(zip(ids, pose['per_point_error_px']), key=lambda row: row[1], reverse=True)
        suspects = [(pid, error) for pid, error in ranked if error > 3.] or ranked[:3]
        worst = '、'.join(f'{pid} {error:.2f}px' for pid, error in suspects[:3])
        raise CalibrationReprojectionError(pose,
            f'重投影检查未通过：RMS {pose["undistorted_reprojection_rms_px"]:.2f}px（需 ≤1.50px）；'
            f'优先复核 {worst}（单点需 ≤3.00px）。可点击误差表中的点重选，或移除后重新检查')
    return pose


def predicted_pixels(target, pose, intrinsics, inset=DEFAULT_INSET_MM):
    coordinates = marker_xyz(target, inset)
    ids = list(coordinates)
    xyz = np.array([coordinates[p] for p in ids])
    transform = np.asarray(pose['T_camera_from_target_mm'])
    camera_xyz = xyz@transform[:3, :3].T+transform[:3, 3]
    pixels = project(camera_xyz, intrinsics)
    return {pid: tuple(map(float, pixel)) for pid, pixel, cp in zip(ids, pixels, camera_xyz)
            if cp[2] > 0 and 0 <= pixel[0] < intrinsics['width'] and 0 <= pixel[1] < intrinsics['height']}
