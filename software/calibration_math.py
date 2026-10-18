import cv2
import numpy as np


class CalibrationReprojectionError(ValueError):
    """A rejected fit retained only to help the operator correct correspondences."""

    def __init__(self, diagnostic_pose, message=None):
        self.diagnostic_pose = dict(diagnostic_pose, diagnostic_only=True)
        errors = diagnostic_pose['per_point_error_px']
        super().__init__(message or
                         f'去畸变重投影误差过大：RMS={diagnostic_pose["undistorted_reprojection_rms_px"]:.2f}px，最大={max(errors):.2f}px')


def solve_target_pose(xyz_mm, normalized_xy, fx, fy):
    xyz = np.asarray(xyz_mm, dtype=np.float64)
    xy = np.asarray(normalized_xy, dtype=np.float64)
    if len(xyz) < 6 or xyz.shape != (len(xy), 3):
        raise ValueError('至少选择六个对应点，包含中间面和至少一个侧面')
    if np.linalg.matrix_rank(xyz-xyz.mean(axis=0), tol=.01) < 3:
        raise ValueError('所选点共面，请增加侧面的点')
    ok, r, t, inliers = cv2.solvePnPRansac(
        xyz, xy, np.eye(3), None, iterationsCount=500,
        reprojectionError=2.0/max(fx, fy), confidence=.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        raise ValueError('可靠对应点不足，检查点号和点击位置')
    idx = inliers.ravel()
    if np.linalg.matrix_rank(xyz[idx]-xyz[idx].mean(axis=0), tol=.01) < 3:
        raise ValueError('有效点没有覆盖侧面，请重新标点')
    r, t = cv2.solvePnPRefineLM(xyz[idx], xy[idx], np.eye(3), None, r, t)
    R = cv2.Rodrigues(r)[0]
    camera_xyz = (R @ xyz.T + t).T
    if np.any(camera_xyz[:, 2] <= 0):
        raise ValueError('位姿把点放到了相机背后，拒绝保存')
    prediction = camera_xyz[:, :2]/camera_xyz[:, 2:3]
    errors = np.linalg.norm((prediction-xy)*[fx, fy], axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t.ravel()
    pose = {'T_camera_from_target_mm': T.tolist(),
            'T_target_from_camera_mm': np.linalg.inv(T).tolist(),
            'camera_center_target_mm': (-R.T@t).ravel().tolist(),
            'rvec': r.ravel().tolist(), 'tvec_mm': t.ravel().tolist(),
            'undistorted_reprojection_rms_px': rms, 'per_point_error_px': errors.tolist()}
    if np.max(errors) > 3 or rms > 1.5:
        raise CalibrationReprojectionError(pose)
    return pose


def triangulate_observations(observations):
    """Each observation: normalized_xy, T_camera_from_target_mm, fx, fy.

    Caller must supply synchronized observations of the SAME marker ID.
    """
    if len(observations) < 2:
        raise ValueError('同一点至少需要两个有效视角')
    rows, rays = [], []
    for obs in observations:
        P = np.asarray(obs['T_camera_from_target_mm'], float)[:3]
        u, v = obs['normalized_xy']
        rows.extend([u*P[2]-P[0], v*P[2]-P[1]])
        ray = P[:, :3].T @ np.array([u, v, 1.])
        rays.append(ray/np.linalg.norm(ray))
    best_angle = max(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))
                     for i, a in enumerate(rays) for b in rays[i+1:])
    if best_angle < np.deg2rad(1):
        raise ValueError('交会角过小，深度不可靠')
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h = vt[-1]
    if abs(h[3]) < 1e-10:
        raise ValueError('三角化退化')
    xyz = h[:3]/h[3]
    for obs in observations:
        p = np.asarray(obs['T_camera_from_target_mm']) @ np.r_[xyz, 1]
        if p[2] <= 0:
            raise ValueError('点在相机后方')
        err = np.linalg.norm((p[:2]/p[2]-obs['normalized_xy'])*[obs['fx'], obs['fy']])
        if err > 3:
            raise ValueError('跨视角匹配误差超过 3px')
    return xyz
