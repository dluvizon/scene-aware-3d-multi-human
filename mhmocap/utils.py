import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def automatic_image_crop(image, anchor_point, reference_point, reference_shape):
    """Automatic crop an input image given an image anchor point, a reference point
    located in a reference image (hypothetical), and the reference image shape.
    
    # Arguments
        image: tensor with shape (H, W, ...)
        anchor_point: tensor with shape (2,), where [anchor_row, anchor_col]
        reference_point: tensor with shape (2,), where [reference_row, reference_col]
        reference_shape: tensor with shape (2,), where [reference_num_rows, reference_num_cols]
    
    # Returns
        A cropped image with shape (new_H, new_W) that fits in the reference shape and where
        the anchor point is placed in the reference point.
        A new crop bounding box aligned in the reference shape in the format [x1, y1, x2, y2]
        A new crop bounding box corresponding to the original image placed in the reference shape
        (can extrapolate the reference shape limits).
    """
    img_rows, img_cols = image.shape[:2]
    ref_rows, ref_cols = reference_shape

    if isinstance(anchor_point, list) or isinstance(anchor_point, tuple):
        anchor_point = np.array(anchor_point)
    if isinstance(reference_point, list) or isinstance(reference_point, tuple):
        reference_point = np.array(reference_point)

    img_crop_pts = np.array([[0, 0], [img_rows, 0], [img_rows, img_cols], [0, img_cols]]) - anchor_point + reference_point
    
    row_min, col_min = img_crop_pts.min(axis=0)
    row_max, col_max = img_crop_pts.max(axis=0)
    org_crop = np.array([col_min, row_min, col_max, row_max])
    
    if row_min < 0:
        image = image[-row_min:]
        row_min = 0
    if col_min < 0:
        image = image[:, -col_min:]
        col_min = 0
    if row_max > ref_rows:
        image = image[:ref_rows - row_max]
        row_max = ref_rows
    if col_max > ref_cols:
        image = image[:, :ref_cols - col_max]
        col_max = ref_cols
    
    return image, np.array([col_min, row_min, col_max, row_max]), org_crop


def sample_average_depth(depth, mask, pos, win_size, metric='avg'):
    """Sample depth values from a given position and window size.

    # Arguments
        depth: tensor with shape [H, W]
        mask: boolean tensor with shape [H, W]
        pos: list or tuple with position [v|y|row, u|x|col]
        win_size: integer with window size (squared) in pixels
        metric: string with metric for sampling

    # Returns
        Sampled depth value
    """
    assert metric in ['avg', 'min', 'max'], (
        f'Error: sample_average_depth: invalid metric {metric}')
    assert depth.shape == mask.shape, (
        f'Error: sample_average_depth: invlaid depth and mask shapes '
        f'{depth.shape}, {mask.shape}')

    max_row, max_col = depth.shape
    pos_row, pos_col = pos
    assert (pos_col >= 0) and (pos_col < max_col) and (pos_row >= 0) and (pos_row < max_row), (
        f'Error: sample_average_depth: invalid position {pos}')

    r1 = max(pos_row - win_size // 2, 0)
    r2 = min(pos_row + win_size // 2, max_row)
    c1 = max(pos_col - win_size // 2, 0)
    c2 = min(pos_col + win_size // 2, max_col)
    de = depth[r1:r2, c1:c2][mask[r1:r2, c1:c2] > 0]

    if metric == 'avg':
        return de.mean(dtype=de.dtype)
    if metric == 'min':
        return de.min()
    if metric == 'max':
        return de.max()


def fillin_values(x, mask, filter_size, metric='median'):
    """For all the values marked as zero in the mask, fill-in based on the
    given metric on the neighbors pixels, considering the given filter size.
    If all neighbors of i are masked out, do not update x and mask i.

    # Arguments
        x: tensor with shape [H, W, ...]
        mask: tensor with shape [H, W]
        filter_size: integer size of a square filter
        metric: string

    # Returns
        A new x and mask tensors with updated values.
    """
    assert x.shape[0:2] == mask.shape, (
        f'Error: invalid x/mask shapes {x.shape}/{mask.shape}')
    assert filter_size > 1, f'Error: invalid filter size {filter_size}, must be > 1'
    valid_metrics = ['median', 'mean', 'max', 'min']
    assert metric in valid_metrics, (
        f'Error: invalid metric {metric}. Valid metrics are: ' + str(valid_metrics))

    fm = getattr(np, metric)
    nx = x.copy()
    nmask = mask.copy()

    num_rows, num_cols = nx.shape[0:2]
    k = filter_size // 2
    for row in range(num_rows):
        for col in range(num_cols):
            if mask[row, col]:
                continue

            min_r = max(0, row - k)
            max_r = min(num_rows, row + k + 1)
            min_c = max(0, col - k)
            max_c = min(num_cols, col + k + 1)
            v = nx[min_r : max_r, min_c : max_c]
            m = mask[min_r : max_r, min_c : max_c]
            midx = m > 0
            if midx.any():
                vlist = v[midx, ...]
                nx[row, col] = fm(vlist, axis=0)
                nmask[row, col] = 1

    return nx, nmask



def postprocess_dispmap(dispmap, mask=None, minz=1, maxz=20, fillin_ksize=7):
    """Post-process a disparity map to remove outliers and 'flying pixels'.

    # Arguments
        dispmap: numpy array with shape (H, W)
        maxz: maximum expected value in Z corresponding to this disparity map

    # Returns
        a new numpy array with a filtered disparity map with shape (H, W)
    """
    disp = np.clip(dispmap, 1.0 / maxz, 1)

    sobel_disp_grad_x = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    sobel_disp_grad_y = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    sobel_disp = np.abs(sobel_disp_grad_x) + np.abs(sobel_disp_grad_y)

    depth = 1.0 / (disp * (1.0 / minz - 1.0 / maxz) + 1.0 / maxz)
    sobel_depth_grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    sobel_depth_grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    sobel_depth = np.abs(sobel_depth_grad_x) + np.abs(sobel_depth_grad_y)

    sobel_grad = sobel_disp / sobel_disp.std() + sobel_depth / sobel_depth.std()
    sobel_edges = (sobel_grad > 3 * sobel_grad.mean()).astype(disp.dtype)
    dmask = cv2.erode((1 - sobel_edges), np.ones((3, 3)), iterations=2)
    if mask is not None:
        dmask = dmask * mask

    new_disp = disp
    new_mask = dmask
    while (new_mask.min() < 1):
        new_disp, new_mask = fillin_values(new_disp, new_mask, filter_size=fillin_ksize)

    return new_disp


def postprocess_depthmap(depth, mask=None, fillin_ksize=7, use_bilateral_filter=False):
    """Post-process a depth map to remove outliers and 'flying pixels'.

    # Arguments
        depth: numpy array with shape (H, W)
        depth: numpy array with shape (H, W)

    # Returns
        a new numpy array with a filtered depth map with shape (H, W)
    """
    if use_bilateral_filter:
        pred_disp = cv2.bilateralFilter(1.0 / np.clip(depth, 0.01, 100), 9, sigmaColor=0.05, sigmaSpace=25)
        depth = 1.0 / np.clip(pred_disp, 0.01, 100)

    disp = 1.0 / np.clip(depth, 0.1, 100)

    sobel_disp_grad_x = cv2.Sobel(disp, cv2.CV_32F, 1, 0, ksize=3)
    sobel_disp_grad_y = cv2.Sobel(disp, cv2.CV_32F, 0, 1, ksize=3)
    sobel_disp = np.abs(sobel_disp_grad_x) + np.abs(sobel_disp_grad_y)

    sobel_depth_grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    sobel_depth_grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    sobel_depth = np.abs(sobel_depth_grad_x) + np.abs(sobel_depth_grad_y)

    sobel_grad = sobel_disp / sobel_disp.std() + sobel_depth / sobel_depth.std()
    sobel_edges = (sobel_grad > 3 * sobel_grad.mean()).astype(disp.dtype)
    dmask = cv2.erode((1 - sobel_edges), np.ones((3, 3)), iterations=2)
    if mask is not None:
        dmask = dmask * mask

    new_depth = depth
    new_mask = dmask
    while (new_mask.min() < 1):
        new_depth, new_mask = fillin_values(new_depth, new_mask, filter_size=fillin_ksize)

    return new_depth


def get_effective_camera_intrinsics(actual_image_size, caminfo):
    """Returns the actual camera intrinsics (focal length and point center),
    considering the actual image size and the original camera intrinsics.

    # Arguments
        actual_image_size: tuple with current image size (width, height)
        caminfo: dictionary with
            'K': 3x3 matrix with camera intrinsics
            'image_size': tuple with the original image size (org_width, org_height)

    # Returns
        A new caminfo dictionary with the corrected values.
    """
    assert 'K' in caminfo.keys() and 'image_size' in caminfo.keys(), (
        f'Error: get_effective_camera_intrinsics: missing information in caminfo.')

    actual_image_size = np.array(actual_image_size)
    org_image_size = np.array(caminfo['image_size'])
    fx = caminfo['K'][0, 0] * actual_image_size[0] / org_image_size[0]
    fy = caminfo['K'][1, 1] * actual_image_size[1] / org_image_size[1]
    cx = caminfo['K'][0, 2] * actual_image_size[0] / org_image_size[0]
    cy = caminfo['K'][1, 2] * actual_image_size[1] / org_image_size[1]

    return {
        'K': np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]),
        'image_size': tuple(actual_image_size),
    }


def aggregate_kclosest_points(points, refidxs, k, num_iter=1):
    """Search for the K closests points in points[refidxs] and return
    a numpy array with the new found points indexes. Can work iteratively
    if num_iter > 1.
    
    # Arguments
        points: numpy array with shape (N, 3)
        refindxs: numpy array of index with shape (P,)
        k: integer
        num_iter: integer
    
    # Returns
        A numpy array of indexes with shape (Q,), where Q is the number
        of closest points found, already excluding replicates.
    """
    def __aggregate_kclosest(pts, ref):
        ps = pts[ref]
        idxs = np.array([], dtype=int)
        for p in ps:
            pd = np.sqrt(np.sum(np.square(pts - p), axis=-1))
            idxs = np.concatenate([idxs, np.argsort(pd)[1:k+1]], axis=0)

        return np.unique(idxs)

    new_vertices = refidxs
    aggregated = new_vertices
    for _ in range(num_iter):
        new_vertices = __aggregate_kclosest(points, new_vertices)
        aggregated = np.append(aggregated, new_vertices)

    return aggregated


def linear_kpts_assignment(pref, pred, thr=0.5):
    """Performs a linear assignment (Hungarian algorithm) for a set of
    reference and predicted poses/points.
    
    # Arguments
        pref: numpy (K, J, D+1), where K is the number of samples, J is
            the number of joints, and D is the dimmention. The last dimmention
            is a visibility score, from 0 to 1.
        pred: numpy (N, J, D+1), where N is the number of predictions.

    # Returns
        The indexes of `pref` that best match the samples in `pred`, as a
        tuple (pref_idx, pred_idx).
    """
    assert ((pred.ndim == 3) and (pred.ndim == 3)
            and (pred.shape[1:3] == pref.shape[1:3])), (
        f'Error: invalid input shapes {pref.shape} / {pred.shape}')
    K = pref.shape[0]
    N = pred.shape[0]
    pref = np.tile(pref[:, np.newaxis], (1, N, 1, 1))
    pred = np.tile(pred[np.newaxis], (K, 1, 1, 1))
    valid = ((pref[..., 2] > thr) * (pred[..., 2] > thr))
    dist = np.sqrt(np.sum(np.square(pref - pred), axis=-1))
    
    valid = valid.reshape((K * N, -1))
    dist = dist.reshape((K * N, -1))
    avg_dist = 1e6 * np.ones((K * N,), np.float32)
    for i, (d, v) in enumerate(zip(dist, valid)):
        if v.sum() > 0:
            avg_dist[i] = np.mean(d[v])

    pref_idx, pred_idx = linear_sum_assignment(avg_dist.reshape((K, N)))

    return pref_idx, pred_idx


def decouple_instance_segmentation_masks(instances, cls=None):
    """Given a batch of instance segmentation masks with cls classes, decouple
    them into cls binary masks. If cls is None, takes the maximum class index
    from the input.

    # Arguments
        instances: array int with shape (batch, H, W)
        cls: Number of classes, or None.

    # Returns
        A float array with one mask per class, with shape (batch, cls, H, W).
    """
    if cls is None:
        cls = np.unique(instances).max()

    seg_mask = np.zeros((instances.shape[0], cls) + instances.shape[1:], dtype=np.float32)
    for i, idx in enumerate(range(cls)):
        seg_mask[:, i] = (instances == idx + 1).astype(np.float32)

    return seg_mask


def angle_between_vectors(a, b):
    """Return the angle between two 3D vetors.
    
    # Arguments

    # Returns
    """
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.rad2deg(np.arccos(cos_theta))


def centered_boundingboxes(p2d, img_size, pix_size=7):
    w, h = img_size
    d = pix_size // 2
    c1 = np.clip(p2d[:, 0] - d, 0, w)
    c2 = np.clip(c1 + pix_size, 0, w)
    r1 = np.clip(p2d[:, 1] - d, 0, h)
    r2 = np.clip(r1 + pix_size, 0, h)
    bboxes = np.stack([r1, c1, r2, c2], axis=1)

    return bboxes


def sampling_boundingboxes(depth, bboxes, metric='mean'):
    N = len(bboxes)
    values = np.zeros((N,), np.float32)
    mask = np.zeros((N,), np.float32)
    metric_fn = getattr(np, metric)

    for i, b in enumerate(bboxes):
        r1, c1, r2, c2 = b
        if (r2 > r1) and (c2 > c1):
            values[i] = metric_fn(depth[r1:r2, c1:c2])
            mask[i] = 1.0

    return values, mask


def compute_points_inside_mesh(verts, faces, points):
    """Return the indices of points that are inside the given mesh.
    Assume that the normals of the vertices are pointing outside the mesh.
    
    # Arguments
        verts: Array with shape (V, 3)
        faces: Array with shape (F, 3)
        points: Array with shape (N, 3)
    
    # Returns
        A new array with the indices of points with shape (M,),
        where 0 <= M <= N.
    """
    # Compute all the triangle's centers
    faces = faces[..., np.newaxis] # (F, 3, 1)
    verts = verts[:, np.newaxis] # (V, 1, 3)
    triangles = np.take_along_axis(verts, faces, axis=0) # (F, 3, 3)
    centers = np.mean(triangles, axis=1)[np.newaxis] # (1, F, 3)
    
    # Define a bounding box around the mesh and filter to points inside it
    c_min = (np.min(centers, axis=(0, 1)) - 1e-3)[np.newaxis] # (1, 3)
    c_max = (np.max(centers, axis=(0, 1)) + 1e-3)[np.newaxis] # (1, 3)
    pts_mask = ((points > c_min) * (points < c_max)).all(axis=1)
    pts_idx = np.array(range(len(pts_mask)))[pts_mask]
    sel_points = points[pts_idx] # (S, 3)
    
    # Find the closest triangle for each selected point
    disp = np.sum(np.square(sel_points[:, np.newaxis] - centers), axis=2)
    closest_face_idx = np.argmin(disp, axis=1) # (S,)
    sel_triangles = triangles[closest_face_idx] # (S, 3, 3)
    sel_centers = centers[0, closest_face_idx] # (S, 3)

    # Compute the normals and the center-to-point vectors
    normals = np.cross(
        sel_triangles[:, 1] - sel_triangles[:, 0],
        sel_triangles[:, 2] - sel_triangles[:, 1],
        axis=1) # (F, 3)
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-3, None)
    c2p = sel_points - sel_centers
    c2p /= np.clip(np.linalg.norm(c2p, axis=1, keepdims=True), 1e-3, None)
    ip = np.sum(normals * c2p, axis=1)

    return pts_idx[ip < -0.01]
