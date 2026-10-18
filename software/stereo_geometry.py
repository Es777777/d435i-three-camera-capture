"""Calibrated passive IR stereo; RealSense rotations are column-major."""
import cv2
import numpy as np


def matrix(k):
    return np.array([[k['fx'], 0., k['cx']], [0., k['fy'], k['cy']], [0., 0., 1.]])


def extrinsics(k):
    return (np.asarray(k['ir1_to_stream_rotation'], float).reshape(3, 3, order='F'),
            np.asarray(k['ir1_to_stream_translation_m'], float)*1000.)


def project(xyz, k):
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    model = k['distortion'].split('.')[-1]
    coeff = np.asarray(k['coeffs'], float)
    xy = xyz[:, :2]/xyz[:, 2:3]
    if model == 'none' or not np.any(coeff):
        return xy*[k['fx'], k['fy']]+[k['cx'], k['cy']]
    if model == 'brown_conrady':
        return cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), matrix(k), coeff)[0].reshape(-1, 2)
    # Match librealsense rs2_project_point_to_pixel for both models.
    if model in ('modified_brown_conrady', 'inverse_brown_conrady'):
        x, y = xy.T
        r2 = x*x+y*y
        f = 1+coeff[0]*r2+coeff[1]*r2*r2+coeff[4]*r2*r2*r2
        x, y = x*f, y*f
        xx = x+2*coeff[2]*x*y+coeff[3]*(r2+2*x*x)
        yy = y+2*coeff[3]*x*y+coeff[2]*(r2+2*y*y)
        return np.column_stack((xx, yy))*[k['fx'], k['fy']]+[k['cx'], k['cy']]
    raise ValueError('Unsupported distortion model: '+model)


class PassiveStereo:
    def __init__(self, metadata, near_mm=300., far_mm=1800., threshold=210):
        streams = metadata['streams']
        self.left, self.right, self.color = [streams[key] for key in
            ('stream.infrared:1', 'stream.infrared:2', 'stream.color:0')]
        self.size = (self.left['width'], self.left['height'])
        if self.size != (self.right['width'], self.right['height']):
            raise ValueError('IR dimensions differ')
        self.near, self.far, self.threshold = near_mm, far_mm, threshold
        R, t = extrinsics(self.right)
        if not 10 < np.linalg.norm(t) < 200:
            raise ValueError('Invalid stereo baseline in mm')
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            matrix(self.left), np.zeros(5), matrix(self.right), np.zeros(5), self.size,
            np.ascontiguousarray(R), np.ascontiguousarray(t.reshape(3,1)),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        if self.P2[0, 3] >= 0 or abs(self.P2[1, 3]) > .01:
            raise ValueError('Expected horizontal IR1-left / IR2-right stereo')
        self.maps = []
        u, v = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]))
        for k, rot, P in zip((self.left, self.right), (self.R1, self.R2), (self.P1, self.P2)):
            rays = np.column_stack(((u.ravel()-P[0, 2])/P[0, 0],
                                    (v.ravel()-P[1, 2])/P[1, 1], np.ones(u.size))) @ rot
            pixels = project(rays, k).reshape(self.size[1], self.size[0], 2).astype(np.float32)
            self.maps.append((pixels[..., 0], pixels[..., 1]))
        max_d = int(np.ceil((-self.P2[0, 3]/near_mm+16)/16))*16
        self.num_d = min(max_d, ((self.size[0]//2)//16)*16)
        if self.num_d < 16:
            raise ValueError('Stereo image too small')
        args = dict(numDisparities=self.num_d, blockSize=5, P1=8*25, P2=32*25,
                    uniquenessRatio=12, speckleWindowSize=60, speckleRange=2,
                    disp12MaxDiff=1, preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.matcher = cv2.StereoSGBM_create(minDisparity=0, **args)
        self.reverse = cv2.StereoSGBM_create(minDisparity=-self.num_d, **args)

    def rectify(self, left, right):
        return [cv2.remap(im, *maps, cv2.INTER_LINEAR) for im, maps in zip((left, right), self.maps)]

    def blobs(self, im):
        count, labels, stats, centers = cv2.connectedComponentsWithStats((im>=self.threshold).astype('uint8'))
        spots = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if not 3<=area<=300 or max(w, h)>4*min(w, h) or min(x, y)<4:
                continue
            mask = labels[y:y+h, x:x+w] == i
            weights = np.maximum(im[y:y+h, x:x+w].astype(float)-self.threshold+1, 0)*mask
            yy, xx = np.indices(weights.shape)
            spots.append([x+(xx*weights).sum()/weights.sum(), y+(yy*weights).sum()/weights.sum()])
        return np.asarray(spots, float).reshape(-1, 2)

    def marker_points(self, left, right, disparity):
        a, b = self.blobs(left), self.blobs(right)
        candidates = []
        for i, p in enumerate(a):
            choices = []
            for j, q in enumerate(b):
                d = p[0]-q[0]
                if d<=0 or abs(p[1]-q[1])>1.5:
                    continue
                z = -self.P2[0, 3]/d
                if not self.near <= z <= self.far:
                    continue
                x, y = np.rint(p).astype(int)
                patch = disparity[max(0,y-3):y+4, max(0,x-3):x+4]
                good = patch[patch>0]
                if len(good) and abs(float(np.median(good))-d)>3:
                    continue
                choices.append(j)
            # Ambiguous identical spots on one epipolar line are not assigned arbitrarily.
            if len(choices)==1:
                candidates.append((i, choices[0]))
        pairs = [(i,j) for i,j in candidates if sum(v==j for _,v in candidates)==1]
        if not pairs:
            return np.empty((0, 3)), np.empty((0, 2)), np.empty(0)
        p, q = np.array([a[i] for i,j in pairs]), np.array([b[j] for i,j in pairs])
        h = cv2.triangulatePoints(self.P1, self.P2, p.T, q.T)
        xyz = (h[:3]/h[3]).T
        errors = []
        for P, xy in ((self.P1,p), (self.P2,q)):
            projected = np.column_stack((xyz, np.ones(len(xyz)))) @ P.T
            errors.append(np.linalg.norm(projected[:,:2]/projected[:,2:]-xy, axis=1))
        err = np.maximum(*errors)
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:,2]>self.near) & (xyz[:,2]<self.far) & (err<1.5)
        return xyz[valid]@self.R1, p[valid], err[valid]

    def reconstruct(self, left, right, rgb, roi=None, stride=4):
        left, right = self.rectify(left, right)
        disp = self.matcher.compute(left, right).astype(np.float32)/16
        rev = self.reverse.compute(right, left).astype(np.float32)/16
        yy, xx = np.indices(left.shape)
        xr = np.rint(xx-disp).astype(int)
        inside = (xr>=0)&(xr<left.shape[1])
        reverse = rev[yy, np.clip(xr,0,left.shape[1]-1)]
        lr = np.abs(disp+reverse)
        xyz = cv2.reprojectImageTo3D(disp, self.Q)
        mean = cv2.boxFilter(left.astype(np.float32), -1, (7,7))
        var = cv2.boxFilter(left.astype(np.float32)**2, -1, (7,7))-mean**2
        valid = inside & (disp>.5) & (reverse<0) & (lr<=1.) & (var>9) & \
                np.isfinite(xyz).all(axis=2) & (xyz[:,:,2]>=self.near) & (xyz[:,:,2]<=self.far)
        if roi:
            x,y,w,h = roi
            valid &= (xx>=x)&(xx<x+w)&(yy>=y)&(yy<y+h)
        selected = valid & (xx%stride==0) & (yy%stride==0)
        cloud = xyz[selected]@self.R1
        colors = np.repeat(left[selected,None],3,axis=1)
        color_valid = np.zeros(len(cloud), bool)
        if rgb is not None and len(cloud):
            R, t = extrinsics(self.color)
            color_xyz = cloud@R.T+t
            uv = np.rint(project(color_xyz, self.color)).astype(int)
            color_valid = (color_xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<rgb.shape[1])& \
                          (uv[:,1]>=0)&(uv[:,1]<rgb.shape[0])
            idx = np.flatnonzero(color_valid)
            colors[idx] = rgb[uv[idx,1],uv[idx,0],::-1]
        points, pixels, errors = self.marker_points(left, right, disp)
        if roi and len(points):
            x,y,w,h = roi
            keep = (pixels[:,0]>=x)&(pixels[:,0]<x+w)&(pixels[:,1]>=y)&(pixels[:,1]<y+h)
            points, errors = points[keep], errors[keep]
        depth = np.where(valid, xyz[:,:,2], 0).astype(np.float32)
        return dict(xyz_mm=cloud.astype(np.float32), rgb=colors.astype(np.uint8),
                    color_valid=color_valid, depth_mm=depth, markers_mm=points,
                    marker_errors_px=errors, valid_fraction=float(valid.mean()),
                    rectified_ir1=left, rectified_ir2=right)
