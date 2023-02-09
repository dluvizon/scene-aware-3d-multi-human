import os

import numpy as np
import scipy.io as sio
import cv2
import glob
from PIL import Image
import copy
from scipy import stats

import torch

from .smpl import SMPL
from .utils import postprocess_dispmap
from .utils import linear_kpts_assignment
from .utils import decouple_instance_segmentation_masks
from .transforms import get_fov
from .transforms import get_focal
from .transforms import batch_orthographic_projection
from .alphapose import load_alphapose_tracking_results
from .alphapose import update_pose_results
from .alphapose import preprocess_alphapose_predictions
from .alphapose import distance_poses_2d
from .one_euro_filter import OneEuroFilter


def load_mupots_sequence_metadata(samples_path):
    mat = sio.loadmat(os.path.join(samples_path, f'annot.mat'), squeeze_me=False)
    annot = mat['annotations']

    mat = sio.loadmat(os.path.join(samples_path, f'occlusion.mat'), squeeze_me=False)
    occlu = mat['occlusion_labels']

    assert occlu.shape[0] == annot.shape[0], (f'Error in the sequence length!')

    with open(os.path.join(samples_path, 'intrinsics.txt'), 'r') as fip:
        lines = fip.readlines()
        cam_K = np.array([[float(v) for v in r.strip().split()] for r in lines], dtype=np.float32)

    return annot, occlu, cam_K


def load_multiple_images(img_paths, resize_factor):
    images = []
    for fname in img_paths:
        image_pil = Image.open(fname)
        img_w, img_h = image_pil.size
        if np.abs(resize_factor - 1) > 1e-3:
            img_w = int(round(resize_factor * img_w))
            img_h = int(round(resize_factor * img_h))
            image_pil = image_pil.resize((img_w, img_h), resample=Image.BICUBIC)
        images.append(np.array(image_pil))

    return np.stack(images, axis=0) # array with shape (num_frames, H, W, 3)


def load_multiple_depthmaps(de_paths, image_size,
    renormalize=False,
    use_bilateral_filter=False,
    post_process=True):

    depths = []
    for fname in de_paths:
        disp_pil = Image.open(fname)
        img_w, img_h = disp_pil.size

        if (image_size[0] != img_w) or (image_size[1] != img_h):
            disp_pil = disp_pil.resize(image_size, resample=Image.BICUBIC)

        pred_disp = np.array(disp_pil, dtype=np.float32)
        pred_disp = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())
        if use_bilateral_filter:
            pred_disp = cv2.bilateralFilter(pred_disp, 15, sigmaColor=0.3, sigmaSpace=31)
        if post_process:
            pred_disp = postprocess_dispmap(pred_disp, minz=1, maxz=100, fillin_ksize=7)
        if renormalize:
            pred_disp = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())
        depths.append(pred_disp)

    return np.stack(depths, axis=0) # array with shape (num_frames, H, W)


def load_multiple_segmentation_maps(seg_paths, image_size, fg_erode_iters=3, bg_erode_iters=9):
    instances = []
    backgrounds = []
    for fname in seg_paths:
        inst_pil = Image.open(fname)
        img_w, img_h = inst_pil.size

        if (image_size[0] != img_w) or (image_size[1] != img_h):
            inst_pil = inst_pil.resize(image_size, resample=Image.NEAREST)

        inst = np.array(inst_pil)
        back = (inst == 0).astype(inst.dtype)
        if fg_erode_iters > 0:
            # This first iteration is to make sure that different instances
            # that are in contact to each other will not remain "glued" in th
            # erosion process, since it is performed for the composed instance image
            inst_e = cv2.erode(inst, np.ones((3, 3)), iterations=1)
            inst_d = cv2.dilate(inst, np.ones((3, 3)), iterations=1)
            maskout = (inst_e == inst_d).astype(inst.dtype)
            inst = maskout * inst
            if fg_erode_iters > 1:
                inst = cv2.erode(inst, np.ones((3, 3)), iterations=fg_erode_iters - 1)
        if bg_erode_iters > 0:
            back = cv2.erode(back, np.ones((3, 3)), iterations=bg_erode_iters)
        instances.append(inst)
        backgrounds.append(back)

    return np.stack(instances, axis=0), np.stack(backgrounds, axis=0) # arrays with shape (num_frames, H, W)


def load_multiple_bev_predictions(bev_paths):
    bev_preds = []

    print (f'DEBUG:: bev_paths: ', bev_paths)
    for fname in bev_paths:
        bev_pred = np.load(fname, allow_pickle=True)
        bev_pred = bev_pred['results'].item()
        bev_preds.append({
            'cam': bev_pred['cam'],
            'poses': bev_pred['smpl_thetas'],
            'betas': bev_pred['smpl_betas'][:, :10],
        })

    return bev_preds


def load_multiple_romp_predictions(romp_paths):
    romp_preds = []

    for fname in romp_paths:
        romp_pred = np.load(fname, allow_pickle=True)
        romp_pred = romp_pred['results']

        try:
            # This works for old versions or ROMP
            romp_cam = np.stack([romp_pred[k]['cam'] for k in range(len(romp_pred))], axis=0).astype(np.float32) # (s, tx, ty)
            romp_poses = np.stack([romp_pred[k]['poses'] for k in range(len(romp_pred))], axis=0).astype(np.float32)
            romp_betas = np.stack([romp_pred[k]['betas'] for k in range(len(romp_pred))], axis=0).astype(np.float32)
            romp_preds.append({
                'cam': romp_cam,
                'poses': romp_poses,
                'betas': romp_betas,
            })

        except:
            # This should work for newer versions (1.0.6 or later?)
            romp_pred = romp_pred.item()
            romp_preds.append({
                'cam': romp_pred['cam'],
                'poses': romp_pred['smpl_thetas'],
                'betas': romp_pred['smpl_betas'],
            })


    return romp_preds


def assign_instances_to_poses(instances, pose2d, thr=0.5):
    """Assign to each pose a corresponding segmentation mask, so that each
    pose index i has a mask with values (i+1).
    
    # Arguments
        instances: array with instance segmentation values, with shape (num_samples, H, W)
        pose2d: set of poses in the image plane, with shape (num_samples, num_people, J, D+1),
            where D is 2 (image plane) and the last dimmention holds the visibility score.

    # Returns
        A new array of instances, where the mask values match the 2D poses i by (i+1) values.
    """
    assert len(instances) == len(pose2d), (
        f'Error: invalid instances / poses shape {instances.shape} / {pose2d.shape}'
    )

    num_people = pose2d.shape[1]
    for f, (inst, p2d) in enumerate(zip(instances, pose2d)):
        segmentation_cls_reidx = {}
        for k in range(num_people):
            vis = p2d[k, :, 2]
            xlist = np.round(p2d[k, vis > thr, 0]).astype(int)
            ylist = np.round(p2d[k, vis > thr, 1]).astype(int)
            seg_pix = inst[ylist, xlist]
            if (seg_pix > 0).any():
                # we do have valid pixels in the segmentation maps  TODO
                #avg_cls = np.round(seg_pix[seg_pix > 0].mean()).astype(int)
                avg_cls = int(stats.mode(seg_pix[seg_pix > 0])[0].squeeze())
                if (avg_cls not in segmentation_cls_reidx) and (avg_cls != 0):
                    segmentation_cls_reidx[avg_cls] = k + 1

        # Remap segmentation indexes
        new_inst = np.zeros_like(inst, dtype=inst.dtype)
        for old_cls, new_cls in segmentation_cls_reidx.items():
            new_inst[inst == old_cls] = new_cls
        instances[f, ...] = new_inst

    return instances


def assign_instances_to_poses_v2(instances, pose2d, thr=0.5):
    """Assign to each pose a corresponding segmentation mask, so that each
    pose index i has a mask with values (i+1).
    
    # Arguments
        instances: array with instance segmentation values, with shape (num_samples, H, W)
        pose2d: set of poses in the image plane, with shape (num_samples, num_people, J, D+1),
            where D is 2 (image plane) and the last dimmention holds the visibility score.

    # Returns
        A new array of instances, where the mask values match the 2D poses i by (i+1) values.
    """
    assert len(instances) == len(pose2d), (
        f'Error: invalid instances / poses shape {instances.shape} / {pose2d.shape}'
    )
    H, W = instances.shape[1:]

    num_people = pose2d.shape[1]
    for f, (inst, p2d) in enumerate(zip(instances, pose2d)):
        cls_reidx = {}
        seg_cls = np.sort(np.unique(inst))[1:]
        if len(seg_cls) == 0:
            continue

        for c in seg_cls:
            mask = inst == c
            pix_cnt = np.zeros((num_people,))

            for k in range(num_people):
                vis = p2d[k, :, 2]
                xlist = np.round(np.clip(p2d[k, vis > thr, 0], 0, W - 1)).astype(int)
                ylist = np.round(np.clip(p2d[k, vis > thr, 1], 0, H - 1)).astype(int)
                pix_cnt[k] = mask[ylist, xlist].sum()

            cls_reidx[c] = np.argmax(pix_cnt) + 1
            if pix_cnt[cls_reidx[c] - 1] == 0:
                cls_reidx[c] = 0 # that was a mistake, erase that segment

        # Remap segmentation indexes
        new_inst = np.zeros_like(inst, dtype=inst.dtype)
        for old_cls, new_cls in cls_reidx.items():
            new_inst[inst == old_cls] = new_cls
        instances[f, ...] = new_inst

    return instances


def assign_smpl_to_poses(smpl_preds, pose2d, image_size, SMPLPY, sparse_joints_key='joints_alphapose'):
    """Assign to each pose a corresponding set of SMPL parameters.
    If there are more 2D poses than SMPL predictions, interpolate between
    previous and next SMPL predictions. If there are missing 2D pose
    predictions and SMPL models still not assigned, do an assignment based on
    the previous 2D keypoints, assigning to these new poses a confidence of 0.51
    
    # Arguments
        smpl_preds: a list of dictionaries with the following keys: cam, poses, betas
        pose2d: set of poses in the image plane, with shape (num_samples, num_people, J, D+1),
            where D is 2 (image plane) and the last dimmention holds the visibility score.
        SMPLPY: SMPL skinned model wrapper class for vertices regression
        sparse_joints_key: string, type of sparse joints to use,
            e.g., 'joints_alphapose' or 'joints_smpl24'

    # Returns
        A dictionary with SMPL samples arranged and filled, corresponding to pose2d.
    """
    smpl_preds = copy.deepcopy(smpl_preds)
    pose2d = copy.deepcopy(pose2d)

    for f in range(len(pose2d)):
        smpl = smpl_preds[f]
        p2d = pose2d[f] # (N, J, 3)

        smpl['valid'] = np.ones((len(smpl['poses']), 1))
        if len(smpl['poses']) < len(p2d):
            nmiss = len(p2d) - len(smpl['poses'])
            smpl['cam'] = np.concatenate([smpl['cam'], np.ones((nmiss,) + smpl['cam'].shape[1:])], axis=0)
            smpl['valid'] = np.concatenate([smpl['valid'], np.zeros((nmiss,) + smpl['valid'].shape[1:])], axis=0)
            smpl['poses'] = np.concatenate([smpl['poses'], np.zeros((nmiss,) + smpl['poses'].shape[1:])], axis=0)
            smpl['betas'] = np.concatenate([smpl['betas'], np.zeros((nmiss,) + smpl['betas'].shape[1:])], axis=0)
        results = SMPLPY(betas=smpl['betas'], poses=smpl['poses'])
        romp_pose2d_xyz = results[sparse_joints_key].cpu().numpy()
        romp_pose2d_2d = batch_orthographic_projection(romp_pose2d_xyz, smpl['cam'], image_size)

        # check if there are missing 2D poses at this frame.
        # If that is the case, copy the previous 2D pose here and set the maximum
        # confidence score to 0.255 (can be used during optimization, depending on the thr).
        # With this, we can still use an approximation of the
        # current prediction from the previous one as an initial pose.
        # This sample will be marked as `lagged_track`, and if there is a SMPL
        # matching to this one, SMPL keypoints will be used instead.
        lagged_track = np.zeros((len(p2d),)) # (N,)
        p2d_miss = np.sum(p2d[..., 2] >= 0.2, axis=1) < 2 # (N,)
        if (f > 0) and p2d_miss.any():
            prev_tmp = pose2d[f - 1][p2d_miss == True]
            prev_tmp[..., 2] = np.clip(prev_tmp[..., 2], 0, 0.502)
            p2d[p2d_miss == True] = prev_tmp
            lagged_track[p2d_miss == True] = 1
            #print ('prev_tmp', prev_tmp.shape, ' lagged_track', lagged_track)

        num_people, num_joints = romp_pose2d_2d.shape[0:2]
        romp_pose2d_2d = np.concatenate([romp_pose2d_2d, # with SMPL 2D pose, assign low conf. score
                0.502 * smpl['valid'][..., np.newaxis] * np.ones((num_people, num_joints, 1))
                ], axis=-1)
        pref_idx, pred_idx = linear_kpts_assignment(p2d, romp_pose2d_2d, thr=0.501)
        remap = pred_idx[pref_idx]

        for key in smpl.keys():
            smpl[key] = smpl[key][remap] # re-order smpl parameters

        # Now, if there are lagged track
        if lagged_track.any():
            romp_pose2d_2d = romp_pose2d_2d[remap]

            # checks the distance from p2d to smpl and assign if smaller than...
            for n in range(len(p2d)):
                if lagged_track[n]:
                    pose_dist = distance_poses_2d(p2d[n], romp_pose2d_2d[n], thr=0.501)
                    if pose_dist < 0.05 * max(image_size):
                        p2d[n] = romp_pose2d_2d[n]

        pose2d[f] = p2d
        smpl_preds[f] = smpl

    T = len(smpl_preds)
    N = len(smpl_preds[0]['poses'])

    # Perform a (lazy) filling of ROMP predictions to fill in missing predictions
    # Given a missing prediction, find the closest valid one (on time) and just copy it
    for f in range(T):
        for n in range(N):
            if smpl_preds[f]['valid'][n].squeeze() < 1e-4:
                # This is a missing ROMP prediction. Try to find the closest one.
                new_cam = None
                new_poses = None
                new_betas = None

                for k in range(1, T - 1):
                    f_minus = f - k
                    f_plus = f + k

                    if f_minus > 0:
                        if smpl_preds[f_minus]['valid'][n].squeeze() > 0.7:
                            new_cam = smpl_preds[f_minus]['cam'][n].copy()
                            new_poses = smpl_preds[f_minus]['poses'][n].copy()
                            new_betas = smpl_preds[f_minus]['betas'][n].copy()
                            break
                    if f_plus < T:
                        if smpl_preds[f_plus]['valid'][n].squeeze() > 0.7:
                            new_cam = smpl_preds[f_plus]['cam'][n].copy()
                            new_poses = smpl_preds[f_plus]['poses'][n].copy()
                            new_betas = smpl_preds[f_plus]['betas'][n].copy()
                            break
                    if (f_minus < 0) and (f_plus >= T):
                        break

                if new_cam is not None:
                    smpl_preds[f]['valid'][n] = 0.51
                    smpl_preds[f]['cam'][n] = new_cam
                    smpl_preds[f]['poses'][n] = new_poses
                    smpl_preds[f]['betas'][n] = new_betas

    return smpl_preds, pose2d


def load_and_assign_instances(
        frame_ids,
        max_num_people,
        resize_factor,
        SMPLPY,
        images_path,
        alphapose_path,
        hrnet_pose_path,
        use_hrnet_pose,
        depth_path,
        smpl_pred_path,
        segmentation_path,
        renormalize_depth=False,
        post_process_depth=True,
        erode_segmentation_iters=0,
        erode_backmask_iters=0,
        joint_coef_thr=0.49,
        filter_2dpose=True,
        filter_min_cutoff=0.01,
        filter_beta=25,
        verbose=True):

    print (f'DEBUG:: erode_segmentation_iters', erode_segmentation_iters)
    print (f'DEBUG:: erode_backmask_iters', erode_backmask_iters)
    print (f'DEBUG:: use_hrnet_pose', use_hrnet_pose)
    print (f'DEBUG:: joint_coef_thr', joint_coef_thr)
    print (f'DEBUG:: max_num_people', max_num_people)

    # Get the input file names, given the `frame_ids` selection
    img_names = [os.path.splitext(os.path.basename(s))[0] for s in glob.glob(os.path.join(images_path, "*.jpg"))]
    print ('Images_path:', images_path)
    img_names = sorted(img_names)
    if frame_ids is not None and len(frame_ids) > 0:
        img_names = [img_names[i] for i in frame_ids]
    else:
        frame_ids = range(len(img_names))

    img_paths = [os.path.join(images_path, s + '.jpg') for s in img_names]
    images = load_multiple_images(img_paths, resize_factor)
    image_size = images.shape[1:3][::-1]
    if verbose:
        print ('Image data:', images.shape, images.min(), images.max())

    de_paths = [os.path.join(depth_path, s + '.png') for s in img_names]
    depths = load_multiple_depthmaps(de_paths, image_size, renormalize=renormalize_depth, post_process=post_process_depth)
    if verbose:
        print ('Depth data:', depths.shape, depths.min(), depths.max())

    seg_paths = [os.path.join(segmentation_path, s + '.png') for s in img_names]
    instances, backmasks = load_multiple_segmentation_maps(seg_paths, image_size, erode_segmentation_iters, erode_backmask_iters)
    if verbose:
        print ('Segmentation data:', instances.shape, instances.min(), instances.max())
        print ('Background mask data:', backmasks.shape, backmasks.min(), backmasks.max())

    tracking_file = os.path.join(alphapose_path, 'alphapose-results.json')
    annot_alphapose = load_alphapose_tracking_results(tracking_file,
            coef_thr=joint_coef_thr,
            min_size=0.15 * min(image_size) / resize_factor)
    if use_hrnet_pose:
        hrnet_pose_file = os.path.join(hrnet_pose_path, 'hrnet-results.json')
        hrnet_pose = load_alphapose_tracking_results(hrnet_pose_file,
                coef_thr=0.2,
                min_size=0.15 * min(image_size) / resize_factor,
                ignore_tracking=True)
        annot_alphapose = update_pose_results(annot_alphapose, hrnet_pose)

    romp_paths = [os.path.join(smpl_pred_path, s + '.npz') for s in img_names]
    romp_preds = load_multiple_romp_predictions(romp_paths)
    if verbose:
        print ('ROMP predictions: ', len(romp_preds), romp_preds[0].keys())
    
    pose2d = preprocess_alphapose_predictions(annot_alphapose,
            frame_ids=frame_ids, max_num_people=max_num_people, verbose=verbose)
    pose2d[..., 0:2] *= resize_factor
    if verbose:
        print ('AlphaPose data: ', pose2d.shape) # (num_frames, num_people, 17, 3)

    # flag joints outside the image boundary as not visible
    pose2d[..., 2] *= (
        (pose2d[..., 0] >= 0)
        * (pose2d[..., 0] < image_size[0] - 1)
        * (pose2d[..., 1] >= 0)
        * (pose2d[..., 1] < image_size[1] - 1)
    )
    #pose2d[..., 2] = (pose2d[..., 2] > joint_coef_thr).astype(pose2d.dtype)

    # Here, we can remove poses that are not visible in this sequence
    # for more than a given percentage of the frames, since we want to
    # track people that are mostly visible in the entire sequence
    pvis = (pose2d[..., 2] > joint_coef_thr).max(axis=2).mean(axis=0) # (N,)
    pvis_thr = 1.0 / 8 # should appear in at least X% of the frames
    print ('DEBUG:: pvis', pvis, f" threshold is {pvis_thr}")
    pose2d = pose2d[:, pvis >= pvis_thr] # (T, new_N, J, 3)

    romp_preds, pose2d = assign_smpl_to_poses(romp_preds, pose2d, image_size, SMPLPY, sparse_joints_key='joints_alphapose')
    if verbose:
        print ('ROMP predictions (final): ', len(romp_preds), romp_preds[0].keys())

    # Assign instances and SMPL predictions to each person, based on its 2D pose
    instances = assign_instances_to_poses_v2(instances, pose2d)

    # Filter 2D poses to reduce quantization noise (e.g., from the argmax in AlphaPose)
    if filter_2dpose:
        if verbose:
            print ('Filtering 2D poses with One-Euro filter.')
        frame_rate = 25
        T, N = pose2d.shape[0:2]
        H, W = images.shape[1:3]
        p2d_norm = pose2d.reshape((T, -1, 3))
        p2d_norm[..., 0] /= W
        p2d_norm[..., 1] /= H
        p2d_norm_fw = p2d_norm[..., 0:2].copy()
        p2d_norm_bw = p2d_norm[..., 0:2].copy()
        time_i = np.zeros_like(p2d_norm[0, :, 0:2])
        oef_fw = OneEuroFilter(time_i, p2d_norm[0, :, 0:2], min_cutoff=filter_min_cutoff, beta=filter_beta)
        oef_bw = OneEuroFilter(time_i, p2d_norm[-1, :, 0:2], min_cutoff=filter_min_cutoff, beta=filter_beta)
        for i in range(1, T):
            j = T - i
            time_i = time_i + (i / frame_rate)
            mask = np.tile(p2d_norm[i, :, 2:] > joint_coef_thr, (1, 1, 2)).astype(np.float32)
            p2d_norm_fw[i] = oef_fw(time_i, p2d_norm[i, :, 0:2].copy(), mask=mask)
            mask = np.tile(p2d_norm[j, :, 2:] > joint_coef_thr, (1, 1, 2)).astype(np.float32)
            p2d_norm_bw[j] = oef_bw(time_i, p2d_norm[j, :, 0:2].copy(), mask=mask)

        p2d_norm_f = (p2d_norm_fw + p2d_norm_bw) / 2.0
        p2d_norm_f[..., 0] *= W
        p2d_norm_f[..., 1] *= H
        p2d_norm_f = p2d_norm_f.reshape((T, N, -1, 2))
        pose2d[..., 0:2] = p2d_norm_f[..., 0:2]

    # Unwrap the dictionary with romp predictions to a set of numpy arrays
    cam_smpl = np.array([d['cam'] for d in romp_preds], np.float32)
    poses_smpl = np.array([d['poses'] for d in romp_preds], np.float32)
    betas_smpl = np.array([d['betas'] for d in romp_preds], np.float32)
    valid_smpl = np.array([d['valid'] for d in romp_preds], np.float32)

    return {
        'images': images,
        'depths': depths,
        'instances': instances,
        'backmasks': backmasks,
        'pose2d': pose2d,
        'cam_smpl': cam_smpl,
        'poses_smpl': poses_smpl,
        'betas_smpl': betas_smpl,
        'valid_smpl': valid_smpl,
        'romp_preds': romp_preds,
        'frame_ids': np.array(frame_ids, int),
    }


class H3DHCustomSequenceData(torch.utils.data.Dataset):
    def __init__(self, data_root,
                 cam_K=None,
                 cam_dist_coef=None,
                 fov=60, # if cam_K is given, fov is recomputed
                 frame_ids=None, # If None, use all the frames from the folder
                 max_num_people=None,
                 resize_factor=1/4,
                 images_path='images',
                 alphapose_path='AlphaPose',
                 hrnet_pose_path='HRNet2DPose',
                 use_hrnet_pose=True,
                 joint_confidence_thr=0.5,
                 depth_path='DPT_midas21_monodepth',
                 smpl_pred_path='ROMP_Predictions',
                 segmentation_path='Mask2Former_Instances',
                 getitem_keys=[
                     'images',
                     'depths',
                     'seg_mask',
                     'backmasks',
                     'pose2d',
                     'poses_smpl',
                     'betas_smpl',
                     'valid_smpl',
                     'cam_smpl',
                     'frame_ids',
                     'idxs'],
                 smpl_model_parameters_path='../model_data/parameters',
                 erode_segmentation_iters=0,
                 erode_backmask_iters=0,
                 renormalize_depth=True,
                 post_process_depth=True,
                 filter_2dpose=False,
                 filter_min_cutoff=0.004,
                 filter_beta=30,
                 device_name='cuda', # or 'cpu'
                ):

        print (f'DEBUG:: H3DHCustomSequenceData')

        # Creates a wrapper for SMPL inference and project the SMPL parameters to the image plane
        smpl_J_reg_extra_path = os.path.join(smpl_model_parameters_path, 'J_regressor_extra.npy')
        smpl_J_reg_h37m_path = os.path.join(smpl_model_parameters_path, 'J_regressor_h36m.npy')
        smpl_J_reg_alphapose_path = os.path.join(smpl_model_parameters_path, 'SMPL_AlphaPose_Regressor_RMSprop_6.npy')
        smpl_J_reg_mupots_path = os.path.join(smpl_model_parameters_path, 'SMPL_MuPoTs_Regressor_v1.npy')
        device = torch.device(device_name)
        SMPLPY = SMPL(
            smpl_model_parameters_path,
            J_reg_extra9_path=smpl_J_reg_extra_path,
            J_reg_h36m17_path=smpl_J_reg_h37m_path,
            J_reg_alphapose_path=smpl_J_reg_alphapose_path,
            J_reg_mupots_path=smpl_J_reg_mupots_path).to(device)

        images_path = os.path.join(data_root, images_path)
        alphapose_path = os.path.join(data_root, alphapose_path)
        hrnet_pose_path = os.path.join(data_root, hrnet_pose_path)
        depth_path = os.path.join(data_root, depth_path)
        smpl_pred_path = os.path.join(data_root, smpl_pred_path)
        segmentation_path = os.path.join(data_root, segmentation_path)

        # Pre-load data, it could take a while
        data = load_and_assign_instances(frame_ids,
                max_num_people,
                resize_factor,
                SMPLPY,
                images_path,
                alphapose_path,
                hrnet_pose_path,
                use_hrnet_pose,
                depth_path,
                smpl_pred_path,
                segmentation_path,
                renormalize_depth=renormalize_depth,
                post_process_depth=post_process_depth,
                erode_segmentation_iters=erode_segmentation_iters,
                erode_backmask_iters=erode_backmask_iters,
                joint_coef_thr=joint_confidence_thr,
                filter_2dpose=filter_2dpose,
                filter_min_cutoff=filter_min_cutoff,
                filter_beta=filter_beta,
                verbose=True)

        data['seg_mask'] = decouple_instance_segmentation_masks(data['instances'], cls=data['pose2d'].shape[1])
        data['idxs'] = np.array(range(len(data['frame_ids']))) # Indexes of the current loaded samples

        image_size = data['images'].shape[1:3][::-1]
        if cam_K is not None:
            cam_K = resize_factor * cam_K
            fov = get_fov(min(image_size), min(cam_K[0, 0], cam_K[1, 1]))
        else:
            f = get_focal(min(image_size), fov)
            cam_K = np.array([
                [f, 0, image_size[0] / 2],
                [0, f, image_size[1] / 2],
                [0, 0, 1],
            ], np.float32)

        cam = {
            'K': cam_K,
            'fov': fov,
            'Kd': cam_dist_coef,
            'image_size': image_size,
            }
        print (f'DEBUG:: H3DHCustomSequenceData: using cam:', cam)

        self.data = data
        self.image_size = image_size
        self.cam = cam
        self.getitem_keys = getitem_keys
        self.SMPLPY = SMPLPY

    def __len__(self):
        return len(self.data['frame_ids'])

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        ret = {}
        for key in self.getitem_keys:
            try:
                ret[key] = self.data[key][idx]
            except Exception as e:
                print (f'Error in key "{key}, idx {idx}"', str(e))
                raise StopIteration

        return ret
