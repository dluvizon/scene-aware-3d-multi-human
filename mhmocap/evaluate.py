
import numpy as np

from .utils import linear_kpts_assignment
from .transforms import batch_orthographic_projection
from .transforms import camera_projection


smpl24j_to_mupots_regression = [
    [[8/2, -6/2], [15, 12]], # 0
    [[1], [12]], # 1
    [[1], [17]], # 2
    [[1], [19]], # 3
    [[1], [21]], # 4
    [[1], [16]], # 5
    [[1], [18]], # 6
    [[1], [20]], # 7
    [[3/2, -1/2], [2, 1]], # 8
    [[1], [5]], # 9
    [[1], [8]], # 10
    [[3/2, -1/2], [1, 2]], # 11
    [[1], [4]], # 12
    [[1], [7]], # 13
    [[3/2, -1/2], [0, 3]], # 14
    [[1], [6]], # 15
    [[1], [15]], # 16
]

cmu_panoptic_to_mupots15j_map = [
    [[1], [1]], # 0 (MuPoTs)
    [[1], [0]], # 1
    [[1], [9]], # 2
    [[1], [10]], # 3
    [[1], [11]], # 4
    [[1], [3]], # 5
    [[1], [4]], # 6
    [[1], [5]], # 7
    [[1], [12]], # 8
    [[1], [13]], # 9
    [[1], [14]], # 10
    [[1], [6]], # 11
    [[1], [7]], # 12
    [[1], [8]], # 13
    [[1], [2]], # 14
]

alphapose_to_mupots15j_map = [
    [[1], [0]], # 0 (MuPoTs)
    [[1/2, 1/2], [5, 6]], # 1
    [[1], [6]], # 2
    [[1], [8]], # 3
    [[1], [10]], # 4
    [[1], [5]], # 5
    [[1], [7]], # 6
    [[1], [9]], # 7
    [[1], [12]], # 8
    [[1], [14]], # 9
    [[1], [16]], # 10
    [[1], [11]], # 11
    [[1], [13]], # 12
    [[1], [15]], # 13
    [[1/2, 1/2], [11, 12]], # 14
]


def _pose_map(x, mapping):
    """
    # Arguments
        x: numpy with shape (N, J_in, D)
        mapping: list of lists, where `len(mapping) = J_out`,
            and `mapping[j]=[[weights], [joints_in]]`.

    # Returns
        Numpy array `y` with shape (N, J_out, D)
    """
    assert x.ndim == 3, (
        f'Error: invalid input pose with shape {x.shape}'
    )
    N, J_in, D = x.shape
    J_out = len(mapping)
    
    y = np.zeros((N, J_out, D), np.float32)
    for j, (w, m) in enumerate(mapping):
        assert (np.sum(w) - 1.0) < 1e-6, (
            f'Error: invalid values at entry {j} with weights {w}'
        )
        npw = np.array(w, np.float32)[np.newaxis, :, np.newaxis]
        y[:, j] = (npw * x[:, np.array(m, int)]).sum(axis=1)

    return y


def map_cmu_panoptic_to_mupots15j(pose):
    return _pose_map(pose, cmu_panoptic_to_mupots15j_map)


def map_alphapose_to_mupots15j(pose):
    return _pose_map(pose, alphapose_to_mupots15j_map)


def compute_abs_rel_joint_distances(ref_pose3d, pred_pose3d, valid):
    """Compute metrics for one pair of GT/predict sample.
    # Arguments
        ref_pose3d: (17, 3)
        pred_pose3d: (17, 3)
        valid: (17, 1) or (17,)

    # Returns
        Two tuples (Abs-Dist, Root-Dist) with the joint (valid ones) distances
        between the reference and predicted poses.
    """
    root_ref = ref_pose3d[14:15]
    root_pred = pred_pose3d[14:15]

    ref_pose3d = ref_pose3d[:14]
    pred_pose3d = pred_pose3d[:14]
    if valid.ndim == 2:
        valid = valid[:14, 0]
    else:
        valid = valid[:14]
    
    abs_dist = np.sqrt(np.sum(np.square(ref_pose3d - pred_pose3d), axis=-1))
    abs_dist = abs_dist[valid > 0.5]
    
    root_ref_pose3d = ref_pose3d - root_ref
    root_pred_pose3d = pred_pose3d - root_pred
    rel_dist = np.sqrt(np.sum(np.square(root_ref_pose3d - root_pred_pose3d), axis=-1))
    rel_dist = rel_dist[valid > 0.5]

    return abs_dist, rel_dist


def compute_smpl_pred_error_ortho(joints_mupots17j, ref_poses3d, visibility, cam_smpl, cam_K, image_size):
    """ Compute 3D error distance from SMPL predictions to the ground truth poses,
    consedering the best matching on the 2D image plane. For obtaining the 2D poses,
    SMPL predictions are projected based on orthographic projection, and ground truth
    poses (in absolute 3D) are projected using a perspective camera.

    # Arguments
        joints_mupots17j: array with shape (T, N, 17, 3)
        ref_poses3d: array with shape (T, K, 17, 3)
        visibility: array with shape (T, K, 17, 1)
        cam_smpl: SMPL camera with shape (T, N, 3)
        cam_K: camera instrinsics as a (3, 3) matrix

    # Returns
        The relative joint distance (root centered) for a subset of joints
        and valid flags as arrays with shape (T, K, 14).
    """
    T, N = joints_mupots17j.shape[0:2]
    joints_2d = batch_orthographic_projection(joints_mupots17j.reshape((T * N, 17, 3)),
        cam_smpl.reshape((T * N, 3)), image_size).reshape((T, N, 17, 2))
    joints_2d = np.concatenate([joints_2d, np.ones_like(joints_2d[..., 0:1])], axis=-1)

    K = ref_poses3d.shape[1]
    ref_poses2d = camera_projection(ref_poses3d.reshape((T * K * 17, 3)), cam_K).reshape((T, K, 17, 2))
    ref_poses2d = np.concatenate([ref_poses2d, visibility], axis=-1)
    
    rel_dist = np.zeros((T, K, 14), np.float32)
    valid_joints = np.zeros((T, K, 14), np.float32)

    for t in range(T):
        pref_idx, pred_idx = linear_kpts_assignment(ref_poses2d[t], joints_2d[t])

        for k, (gt, pred, vis) in enumerate(zip(ref_poses3d[t, pref_idx], joints_mupots17j[t, pred_idx], visibility[t, pref_idx])):
            root_ref = gt[14:15]
            root_pred = pred[14:15]
            ref_pose3d = gt[:14]
            pred_pose3d = pred[:14]
            vis = vis[:14]

            root_ref_pose3d = ref_pose3d - root_ref
            root_pred_pose3d = pred_pose3d - root_pred
            rel_dist[t, k] = np.sqrt(np.sum(np.square(root_ref_pose3d - root_pred_pose3d), axis=-1))
            valid_joints[t, k] = (vis.squeeze() > 0.5).astype(np.float32)

    return rel_dist, valid_joints


def compute_smpl_pred_error_3dproj(output_data, ref_poses3d, visibility, SMPLPY, cam_K, Kd=None):
    """ Compute 3D error distance from SMPL predictions to the ground truth poses,
    considering the best matching strategy in the projected 2D poses.

    # Arguments
        output_data: dictionary with predictions. The required keys are:
            'poses_T': absolute pose per prediction in 3D (T, N, 1, 3)
            'poses_smpl': SMPL pose axis-angle parameters as (T, N, 72)
            'betas_smpl': SMPL beta parameters as (T, N, 10)
            'scale_factor': scale factor per prediction as (X, N, 1, 1),
                `X` can be either T, for per person and per frame scale,
                or 1, for per person scale (assuming the persons are the same
                in all the frames)
            'valid_smpl': array of flags of valid SMPL predictions (T, N, 1)
        ref_poses3d: numpy with GT poses in 3D as (T, K, J, 3), where K can be
            different from N.
        visibility: numpy with visibility for each keypoint as (T, K, J, 1)
        SMPLPY: SMPL wrapper object for recovering 3D vertices and sparse pose

    # Returns
        Arrays with the absolute and relative distances, and valid joints:
            abs_dist: (T, K, 14)
            rel_dist: (T, K, 14)
            valid_joints: (T, K, 14)
    """
    optim_poses_T = output_data['poses_T']
    optim_scale_factor = output_data['scale_factor']
    optim_poses_smpl = output_data['poses_smpl']
    optim_betas_smpl = output_data['betas_smpl']
    #optim_valid_smpl = output_data['valid_smpl'][..., 0] # (T, N)
    T, N = optim_poses_T.shape[0:2]
    if optim_scale_factor.shape[0] == 1:
        optim_scale_factor = np.tile(optim_scale_factor, (T, 1, 1, 1))

    K, J = ref_poses3d.shape[1:3]
    assert (J == 17) or (J == 19), (
        f'Invalid number of joints ({J}), only 17 (MuPoTs) or 19 (Panoptic) joints are supported, {J} joints given!'
        )
    if J == 19: # If CMU Panoptic layout, convert to 15J from MuPoTs-TS
        ref_poses3d = map_cmu_panoptic_to_mupots15j(ref_poses3d.reshape((T * K, -1, 3))).reshape((T, K, -1, 3))
        visibility = map_cmu_panoptic_to_mupots15j(visibility.reshape((T * K, -1, 1))).reshape((T, K, -1, 1))
    else:
        ref_poses3d = ref_poses3d[:, :, 0:15]
        visibility = visibility[:, :, 0:15]

    results = SMPLPY(betas=optim_betas_smpl.reshape((-1, 10)), poses=optim_poses_smpl.reshape((-1, 72)))

    if J == 19: # If CMU Panoptic
        joints_mupots15j = results['joints_alphapose'].cpu().numpy().reshape((T, N, -1, 3))
        joints_mupots15j = map_alphapose_to_mupots15j(joints_mupots15j.reshape((T * N, -1, 3))).reshape((T, N, -1, 3))
    else:
        joints_mupots15j = results['joints_mupots'].cpu().numpy().reshape((T, N, 17, 3))
        joints_mupots15j = joints_mupots15j[:, :, 0:15, :]

    ref_poses2d = camera_projection(ref_poses3d.reshape((-1, 3)), cam_K, Kd=Kd).reshape((T, K, -1, 2))
    ref_poses2d = np.concatenate([ref_poses2d, visibility], axis=-1)

    matched_ref3dpose = np.zeros((T, K, 14, 3), np.float32)
    matched_pred3dpose = np.zeros((T, K, 14, 3), np.float32)

    abs_root_pos_err = np.zeros((T, K), np.float32)
    valid_root = np.zeros((T, K), np.float32)

    abs_dist = np.zeros((T, K, 14), np.float32) # metrics are computed on the first 14 joints only
    rel_dist = np.zeros((T, K, 14), np.float32)

    valid_joints = np.zeros((T, K, 14), np.float32)

    for t in range(T):

        pref_p3d = ref_poses3d[t] # (K, J, 3)
        pref_vis = visibility[t] # (K, J, 1)
        #pref_p2d = camera_projection(pref_p3d.reshape((-1, 3)), cam_K).reshape(pref_p3d.shape[0:2] + (2,))
        pref_p2d = ref_poses2d[t]
        pred_pose17j_3d = optim_scale_factor[t] * joints_mupots15j[t] + optim_poses_T[t]
        
        pred_pose17j_2d = camera_projection(pred_pose17j_3d.reshape((-1, 3)), cam_K, Kd=Kd).reshape(pred_pose17j_3d.shape[0:2] + (2,))
        pred_pose17j_2d = np.concatenate([pred_pose17j_2d, np.ones_like(pred_pose17j_2d[..., 0:1])], axis=-1)

        #pref_idx, pred_idx = linear_kpts_assignment(np.concatenate([pref_p2d, pref_vis], axis=-1), pred_pose17j_2d[t])
        pref_idx, pred_idx = linear_kpts_assignment(pref_p2d, pred_pose17j_2d)

        for k, (gt, pred, vis) in enumerate(zip(pref_p3d[pref_idx], pred_pose17j_3d[pred_idx], pref_vis[pref_idx])):
            if vis[14, 0] > 0:
                valid_root[t, k] = 1
                abs_root_pos_err[t, k] = np.sqrt(np.sum(np.square(gt[14] - pred[14]), axis=-1))

            root_ref = gt[14:15]
            root_pred = pred[14:15]
            ref_pose3d = gt[:14]
            pred_pose3d = pred[:14]
            vis = vis[:14]

            matched_ref3dpose[t, k] = ref_pose3d
            matched_pred3dpose[t, k] = pred_pose3d

            abs_dist[t, k] = np.sqrt(np.sum(np.square(ref_pose3d - pred_pose3d), axis=-1))

            root_ref_pose3d = ref_pose3d - root_ref
            root_pred_pose3d = pred_pose3d - root_pred
            rel_dist[t, k] = np.sqrt(np.sum(np.square(root_ref_pose3d - root_pred_pose3d), axis=-1))
            valid_joints[t, k] = (vis.squeeze() > 0.49).astype(np.float32)

    abs_jitter = np.abs(
        np.sqrt(np.sum(np.square(matched_ref3dpose[1:] - matched_ref3dpose[:-1]), axis=-1))
        - np.sqrt(np.sum(np.square(matched_pred3dpose[1:] - matched_pred3dpose[:-1]), axis=-1))
    )
    abs_jitter = np.concatenate([abs_jitter[0:1], abs_jitter], axis=0)

    return {
        'abs_dist': abs_dist,
        'rel_dist': rel_dist,
        'valid_joints': valid_joints,
        'abs_root_pos_err': abs_root_pos_err,
        'valid_root': valid_root,
        'abs_jitter': abs_jitter,
    }


def match_pred_to_pref(ref_poses3d, visibility, dataset, poses_smpl, betas_smpl, cam_smpl):
    """Match predictions to reference poses based on the 2D joints projected from 3D poses.
    """
    T, N = poses_smpl.shape[0:2]
    K, J = ref_poses3d.shape[1:3]
    assert J == 17, (f'Invalid number of joints ({J}), only 17 joints are supported')
    
    results = dataset.SMPLPY(betas=betas_smpl.reshape((-1, 10)), poses=poses_smpl.reshape((-1, 72)))
    joints_mupots17j = results['joints_mupots'].cpu().numpy().reshape((T, N, 17, 3))
    pred_pose17j_2d = batch_orthographic_projection(
        joints_mupots17j.reshape((T * N, 17, 3)), cam_smpl.reshape((T * N, 3)),
        dataset.image_size).reshape((T, N, 17, 2))
    pred_pose17j_2d = np.concatenate([pred_pose17j_2d, np.ones_like(pred_pose17j_2d[..., 0:1])], axis=-1)
    
    ref_poses2d = camera_projection(ref_poses3d.reshape((T * K * 17, 3)), dataset.cam['K']).reshape((T, K, 17, 2))
    ref_poses2d = np.concatenate([ref_poses2d, visibility], axis=-1)

    match_list = []
    for t in range(T):
        pref_idx, pred_idx = linear_kpts_assignment(ref_poses2d[t], pred_pose17j_2d[t])
        match_list.append((pref_idx, pred_idx))

    return match_list


def compute_smpl_pred_error_3dproj_matched(optvar, ref_poses3d, visibility, SMPLPY, match_list):
    """ Compute 3D error distance from SMPL predictions to the ground truth poses,
    considering the best matching strategy.

    # Arguments
        optvar: dictionary with predictions. The required keys are:
            'poses_T': absolute pose per prediction in 3D (T, N, 1, 3)
            'poses_smpl': SMPL pose axis-angle parameters as (T, N, 72)
            'betas_smpl': SMPL beta parameters as (T, N, 10)
            'scale_factor': scale factor per prediction as (X, N, 1, 1),
                `X` can be either T, for per person and per frame scale,
                or 1, for per person scale (assuming the persons are the same
                in all the frames)
            'valid_smpl': array of flags of valid SMPL predictions (T, N, 1)
        ref_poses3d: numpy with GT poses in 3D as (T, K, J, 3), where K can be
            different from N.
        visibility: numpy with visibility for each keypoint as (T, K, J, 1)
        SMPLPY: SMPL wrapper object for recovering 3D vertices and sparse pose

    # Returns
        Arrays with the absolute and relative distances, and valid joints:
            abs_dist: (T, K, 14)
            rel_dist: (T, K, 14)
            valid_joints: (T, K, 14)
    """
    optim_poses_T = optvar['poses_T']
    optim_scale_factor = optvar['scale_factor']
    optim_poses_smpl = optvar['poses_smpl']
    optim_betas_smpl = optvar['betas_smpl']
    optim_valid_smpl = optvar['valid_smpl'][..., 0] # (T, N)
    T, N = optim_poses_T.shape[0:2]
    if optim_scale_factor.shape[0] == 1:
        optim_scale_factor = np.tile(optim_scale_factor, (T, 1, 1, 1))

    K, J = ref_poses3d.shape[1:3]
    assert J == 17, (f'Invalid number of joints ({J}), only 17 joints are supported')
    
    results = SMPLPY(betas=optim_betas_smpl.reshape((-1, 10)), poses=optim_poses_smpl.reshape((-1, 72)))
    joints_mupots17j = results['joints_mupots'].cpu().numpy().reshape((T, N, 17, 3))

    matched_ref3dpose = np.zeros((T, K, 14, 3), np.float32)
    matched_pred3dpose = np.zeros((T, K, 14, 3), np.float32)

    abs_dist = np.zeros((T, K, 14), np.float32) # metrics are computed on the first 14 joints only
    rel_dist = np.zeros((T, K, 14), np.float32)
    valid_joints = np.zeros((T, K, 14), np.float32)

    for t in range(T):
        pref_p3d = ref_poses3d[t] # (K, J, 3)
        pref_vis = visibility[t] # (K, J, 1)
        pred_pose17j_3d = optim_scale_factor[t] * joints_mupots17j[t] + optim_poses_T[t]
        pref_idx, pred_idx = match_list[t]

        for k, (gt, pred, vis) in enumerate(zip(pref_p3d[pref_idx], pred_pose17j_3d[pred_idx], pref_vis[pref_idx])):
            root_ref = gt[14:15]
            root_pred = pred[14:15]
            ref_pose3d = gt[:14]
            pred_pose3d = pred[:14]
            vis = vis[:14]

            matched_ref3dpose[t, k] = ref_pose3d
            matched_pred3dpose[t, k] = pred_pose3d

            abs_dist[t, k] = np.sqrt(np.sum(np.square(ref_pose3d - pred_pose3d), axis=-1))

            root_ref_pose3d = ref_pose3d - root_ref
            root_pred_pose3d = pred_pose3d - root_pred
            rel_dist[t, k] = np.sqrt(np.sum(np.square(root_ref_pose3d - root_pred_pose3d), axis=-1))
            valid_joints[t, k] = (vis.squeeze() > 0.5).astype(np.float32)

    s = np.mean((matched_ref3dpose * matched_pred3dpose)
                / np.clip(matched_pred3dpose * matched_pred3dpose, 1e-3, None))
    si_dist = np.sqrt(np.sum(np.square(matched_ref3dpose - s * matched_pred3dpose), axis=-1))

    return abs_dist, rel_dist, si_dist, valid_joints, s


def masked_average_error(dist, vis):
    """Given a distance array and a visibility array, compute the masked
    average error. Input arrays must have the SAME shape.
    
    # Arguments
        dist: float array with shape (...)
        vis: float array with shape (...)

    # Returns
        Scalar value of the masked average error.
    """
    assert dist.shape == vis.shape, (f'Invalid input shapes ({dist.shape}), ({vis.shape})')
    dist = dist.reshape((-1,)).astype(np.float32)
    vis = (vis > 0.5).reshape((-1,)).astype(np.float32)

    return np.sum(vis * dist) / np.clip(np.sum(vis), 1, None)


def masked_average_pck(dist, vis, thr):
    """Given a distance array, a visibility array and a threshold, compute the
    masked average PCK score. Inputs `dist` and `vis` must have the SAME shape.
    
    # Arguments
        dist: float array with shape (...)
        vis: float array with shape (...)
        thr: float (scalar)

    # Returns
        Scalar value of the masked average PCK acore.
    """
    assert dist.shape == vis.shape, (f'Invalid input shapes ({dist.shape}), ({vis.shape})')
    dist = dist.reshape((-1,)).astype(np.float32)
    vis = (vis > 0.5).reshape((-1,)).astype(np.float32)

    return np.sum(vis * (dist <= thr)) / np.clip(np.sum(vis), 1, None)