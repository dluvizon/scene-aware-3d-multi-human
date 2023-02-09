import numpy as np
import torch

def transform_3dpoints(pts3d, RT):
    """Transform a set of 3D points given a rotation matrix and translation vector.

    # Arguments
        pts3d: numpy with shape (N, 3)
        RT: numpy with shape (3, 4) composed of [R | T]

    # Returns
        Transformed 3D points with shape (N, 3)
    """
    tpts3d = pts3d @ RT[:, :3].T + RT[:, 3:].T

    return tpts3d


def camera_projection(pts3d, K, return_depth=False, Kd=None):
    """Project a set of 3D points into the camera plane given camera intrinsics.
    The given 3D keypoints must be in Euclidean coordinates (in mm) with the
    origin in the camera center.

    # Arguments
        pts3d: numpy tensor with shape (N, 3)
        K: intrinsic matrix (3, 3) as [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        Kd: distortion parameters Kd=[k1,k2,p1,p2,k3], or None.

    # Returns
        When `return_depth=False`, returns the projected points into the
        image plane as a numpy tensor with shape (N, 2). Otherwise, return
        also a concatenated depth values as a UVD tensor with shape (N, 3).
    """
    ptsuv = pts3d[:, :2] / pts3d[:, 2:3]
    
    if Kd is not None:
        r = ptsuv[:, 0] * ptsuv[:, 0] + ptsuv[:, 1] * ptsuv[:, 1]
        ptsuv[:, 0] = (
            ptsuv[:, 0] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
            + 2 * Kd[2] * ptsuv[:, 0] * ptsuv[:, 1]
            + Kd[3] * (r + 2 * ptsuv[:, 0] * ptsuv[:, 0])
        )
        ptsuv[:, 1] = (
            ptsuv[:, 1] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
            + 2 * Kd[3] * ptsuv[:, 1] * ptsuv[:, 1]
            + Kd[2] * (r + 2 * ptsuv[:, 1] * ptsuv[:, 1])
        )

    ptsuv = (ptsuv[:, :2] @ K[:2, :2].T) + K[0:2, 2:3].T

    if return_depth:
        return np.concatenate([ptsuv, pts3d[:, 2:]], axis=-1)
    else:
        return ptsuv


def camera_projection_torch(pts3d, K, return_depth=False, Kd=None):
    """Project a set of 3D points into the camera plane given camera intrinsics.
    The given 3D keypoints must be in Euclidean coordinates (in mm) with the
    origin in the camera center.

    # Arguments
        pts3d: torch tensor with shape (N, M, 3), where N is the number of samples
            and M is the number of points in each sample
        K: intrinsic matrix per sample with shape
            (N, 3, 3) as [[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],...]
        Kd: distortion parameters Kd=[k1,k2,p1,p2,k3], or None.

    # Returns
        When `return_depth=False`, returns the projected points into the
        image plane as a torch tensor with shape (N, M, 2). Otherwise, return
        also a concatenated depth values as a UVD tensor with shape (N, M, 3).
    """
    K = torch.transpose(K, 1, 2)
    pts_z = pts3d[..., 2:]
    pts2d = pts3d[..., :2] / pts_z

    if Kd is not None:
        r = pts2d[..., 0] * pts2d[..., 0] + pts2d[..., 1] * pts2d[..., 1]
        xx = (
            pts2d[..., 0] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
            + 2 * Kd[2] * pts2d[..., 0] * pts2d[..., 1]
            + Kd[3] * (r + 2 * pts2d[..., 0] * pts2d[..., 0])
        )
        yy = (
            pts2d[..., 1] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
            + 2 * Kd[3] * pts2d[..., 1] * pts2d[..., 1]
            + Kd[2] * (r + 2 * pts2d[..., 1] * pts2d[..., 1])
        )
        pts2d = torch.stack([xx, yy], axis=-1)

    pts2d = torch.bmm(pts2d, K[:, :2, :2]) + K[:, 2:, :2]
    if return_depth:
        return torch.cat((pts2d, pts_z), -1)
    return pts2d


def camera_inverse_projection(ptsuvd, K):
    """Inverse project a set of points in UVD (2D pixels + absolute depth)
    to 3D coordinate space.

    # Arguments
        ptsuvd: numpy tensor with shape (N, 3)
        K: intrinsic matrix (3, 3) as [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    # Returns
        Inverse projected points in 3D as a numpy with shape (N, 3)
    """
    ptsxy = ptsuvd[:, 2:3] * ((ptsuvd[:, :2] - K[0:2, 2:3].T) @ np.linalg.inv(K[:2, :2].T))

    return np.concatenate([ptsxy, ptsuvd[:, 2:3]], axis=-1)


def camera_inverse_projection_torch(ptsuvd, K):
    """Inverse project a set of points in UVD (2D pixels + absolute depth)
    to 3D coordinate space.
    
    # Arguments
        ptsuvd: torch tensor with shape (N, M, 3), where N is the number of samples
            and M is the number of points in each sample
        K: intrinsic matrix per sample with shape
            (N, 3, 3) as [[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],...]

    # Returns
        Inverse projected points in 3D as a tensor with shape (N, M, 3)
    """
    K = torch.transpose(K, 1, 2)
    ptsxy = ptsuvd[..., 2:3] * ((ptsuvd[..., :2] - K[:, 2:3, 0:2]) @ torch.linalg.inv(K[:, :2, :2]))

    return torch.cat((ptsxy, ptsuvd[..., 2:3]), axis=-1)


def batch_orthographic_projection(p3d, cam, image_size):
    """
    # Arguments
        p3d: Numpy array with shape (N, P, 3)
        cam: Numpy array with shape (N, 3), where [scale, t_x, t_y]
        image_size: tuple (img_w, img_h)
    
    # Returns
        A tensor with shape (N, P, 2) with values in pixel coordinates
    """
    cam = cam[:, np.newaxis]
    p2d = cam[..., 0:1] * p3d[..., :2]
    txy = np.array([image_size], np.float32) / max(image_size)
    p2d = p2d + cam[..., 1:]
    p2d = p2d / 2.0 + txy / 2.0
    p2d = max(image_size) * p2d
    
    return p2d


def recover_camera_intrinsics(pts3d, pts2d):
    """Recover the camera intrinsics in the form of a 3x3 matrix as
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    given a set of points in both 3D and 2D (image plane).

    # Arguments
        pts3d: numpy array with shape (N, 3)
        pts2d: numpy array with shape (N, 2)

    # Returns
        Camera intrinsics K with shape (3, 3)
    """
    pts3d_x = pts3d[:, 0:1] / pts3d[:, 2:3]
    pts3d_y = pts3d[:, 1:2] / pts3d[:, 2:3]
    pts2d_u = pts2d[:, 0:1]
    pts2d_v = pts2d[:, 1:2]

    def _solve_for_single_axis(p3d, p2d):
        p3d = np.concatenate([p3d, np.ones_like(p3d)], axis=-1)
        A = p3d.T @ p2d
        B = p3d.T @ p3d
        K = np.linalg.inv(B) @ A
        return K[0, 0], K[1, 0]

    fx, cx = _solve_for_single_axis(pts3d_x, pts2d_u)
    fy, cy = _solve_for_single_axis(pts3d_y, pts2d_v)

    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)


def recover_camera_intrinsics_simplified(pts3d, pts2d, image_size):
    """Recover the camera intrinsics simplified as a single focal length
    value, i.e., it is assumed that `c_x=W/2` and `c_y=H/2`.
    Returns a 3x3 matrix as
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    given a set of points in both 3D and 2D (image plane).

    # Arguments
        pts3d: numpy array with shape (N, 3)
        pts2d: numpy array with shape (N, 2)
        image_size: tuple (W, H)

    # Returns
        Camera intrinsics K with shape (3, 3)
    """
    cx = image_size[0] / 2
    cy = image_size[1] / 2
    p3p = pts3d[:, 0:2] / pts3d[:, 2:3]
    p2 = pts2d - np.array([[cx, cy]], dtype=np.float32)

    def _solve_for_single_axis(p3d, p2d):
        A = p3d.T @ p2d
        B = p3d.T @ p3d
        K = np.linalg.inv(B) @ A
        print ('K', K.shape)
        return K[0, 0]

    print ('p3p', p3p.shape)
    print ('p2', p2.shape)
    fx = _solve_for_single_axis(p3p[:, 0:1], p2[:, 0:1])
    fy = _solve_for_single_axis(p3p[:, 1:2], p2[:, 1:2])

    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)


def compute_calibration_matrix(znear, zfar, cam_K, image_size):
    """Please refer to https://pytorch3d.org/docs/cameras and
    https://pytorch3d.readthedocs.io/en/latest/modules/renderer/cameras.html
    """
    if image_size[0] > image_size[1]:
        # Landscape image -> use the smaller side (height) as reference
        s1 = 2 * cam_K[1, 1] / image_size[1]
        u = image_size[0] / image_size[1]
        w1 = u * (image_size[0] - 2 * cam_K[0, 2]) / image_size[0]
        h1 = (image_size[1] - 2 * cam_K[1, 2]) / image_size[1]

    elif image_size[1] > image_size[0]:
        # Portrait image -> use the smaller side (width) as reference
        s1 = 2 * cam_K[0, 0] / image_size[0]
        u = image_size[1] / image_size[0]
        w1 = (image_size[0] - 2 * cam_K[0, 2]) / image_size[0]
        h1 = u * (image_size[1] - 2 * cam_K[1, 2]) / image_size[1]

    else:
        # Squared image -> average focal
        s1 = 2 * (cam_K[0, 0] + cam_K[1, 1]) / (image_size[0] + image_size[1])
        w1 = (image_size[0] - 2 * cam_K[0, 2]) / image_size[0]
        h1 = (image_size[1] - 2 * cam_K[1, 2]) / image_size[1]

    s2 = s1 # assume aspect_ratio=1 in the NDC coordinate system
    f1 =  zfar / (zfar - znear)
    f2 = -(zfar * znear) / (zfar - znear)
    
    return np.array([
        [s1, 0, w1, 0],
        [0, s2, h1, 0],
        [0, 0, f1, f2],
        [0, 0, 1, 0],
    ], np.float32)


def get_fov(w, f):
    theta_rad = 2 * np.arctan(0.5 * w / f)
    return 180. * theta_rad / np.pi


def get_focal(w, theta):
    theta_rad = np.pi * theta / 180.
    return 0.5 * w / np.tan(theta_rad / 2.0)


def disp_from_depth(depth, eps=1e-3):
    return 1.0 / torch.clamp(depth, eps)


def bounded_splus_exp(x, min_val, max_val):
    y = x - torch.log(max_val - min_val) / 2.0
    s = torch.log(1.0 / (max_val - min_val) + torch.exp(y))
    z = torch.exp(-s) + min_val
    return z

def bounded_splus_exp_np(x, min_val, max_val):
    y = x - np.log(max_val - min_val) / 2.0
    s = np.log(1.0 / (max_val - min_val) + np.exp(y))
    z = np.exp(-s) + min_val
    return z

def inverted_bounded_splus_exp(z, min_val, max_val):
    s = -torch.log(z - min_val)
    y = torch.log(torch.exp(s) - 1.0 / (max_val - min_val))
    x = y + torch.log(max_val - min_val) / 2.0
    return x

def inverted_bounded_splus_exp_np(z, min_val, max_val):
    s = -np.log(z - min_val)
    y = np.log(np.exp(s) - 1.0 / (max_val - min_val))
    x = y + np.log(max_val - min_val) / 2.0
    return x

def softplus(x):
    return torch.log(1.0 + torch.exp(x))

def softplus_np(x):
    return np.log(1.0 + np.exp(x))

def inverse_softplus(s):
    return torch.log(torch.exp(s) - 1.0)

def inverse_softplus_np(s):
    return np.log(np.exp(s) - 1.0)
