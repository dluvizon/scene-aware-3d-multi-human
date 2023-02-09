import json
import numpy as np
import copy

from .utils import linear_kpts_assignment


def load_alphapose_tracking_results(track_file,
        image_ids=None,
        coef_thr=0.5,
        min_size=None,
        ignore_tracking=False):
    try:
        with open(track_file, 'r') as fip:
            data = json.load(fip)
    except Exception as e:
        print ('load_alphapose_tracking_results: Error', e)
        raise

    if ignore_tracking:
        person_idx = 0

    annot_alphapose = {}
    for d in data:
        if d['category_id'] != 1:
            continue
        if (image_ids is not None) and (d['image_id'] not in image_ids):
            continue

        if d['image_id'] not in annot_alphapose.keys():
            annot_alphapose[d['image_id']] = {}

        pose = np.array(d['keypoints'], np.float32).reshape((-1, 3))

        add_pose = False
        if (np.sum(pose[:, -1] > coef_thr) >= 2): # check at least two valid joints
            if (min_size is not None): # check minimum size
                valp = pose[pose[:, -1] > coef_thr]
                psize = max(np.max(valp.T[0]) - np.min(valp.T[0]), np.max(valp.T[1]) - np.min(valp.T[1]))
                if psize > min_size:
                    add_pose = True
            else:
                add_pose = True

        if add_pose:
            if ignore_tracking:
                #idx = 1 + (
                #    max(annot_alphapose[d['image_id']]) if len(annot_alphapose[d['image_id']]) else 0
                #    )
                annot_alphapose[d['image_id']][person_idx] = pose
                person_idx += 1
            else:
                annot_alphapose[d['image_id']][d['idx']] = pose

    return annot_alphapose


def update_pose_results(annot, new_annot):
    # print (f'DEBUG:: update_pose_results, len(annot)', len(annot))
    # print (f'DEBUG:: update_pose_results, len(annot)', len(new_annot))
    for img_key in annot.keys():
        if img_key in new_annot:
            # print (f'\nDEBUG:: found img_key old', img_key, len(annot[img_key]), annot[img_key].keys())
            # print (f'DEBUG:: found img_key new', img_key, len(new_annot[img_key]), new_annot[img_key].keys())
            annot_keys = list(annot[img_key].keys())
            pref = np.array([annot[img_key][k] for k in annot_keys])
            pnew = np.array([new_annot[img_key][k] for k in new_annot[img_key].keys()])
            # print (f'DEBUG:: pref', pref.shape)
            # print (f'DEBUG:: pnew', pnew.shape)
            pref_idx, pred_idx = linear_kpts_assignment(pref, pnew, thr=0.25)
            # print (f'DEBUG:: annot_keys, pref_idx, pred_idx', annot_keys, pref_idx, pred_idx)
            for i in range(len(pref_idx)):
                annot[img_key][annot_keys[pref_idx[i]]] = pnew[pred_idx[i]]

    return annot


def update_pose_velocity_2d(curr_pose, prev_pose, velocity, decay=0.9, momentum=0.5):
    """Given current and previous 2D poses, update the pose velocity.
    If there is no matching between poses (given the visibility), returns the
    current velocity after decaying.

    # Arguments
        curr_pose: array (J, 3) including visibility
        prev_pose: array (J, 3) including visibility
        velocity: array (2,)

    # Returns
        Updated velocity (2,)
    """
    diff = curr_pose[:, 0:2] - prev_pose[:, 0:2]
    mask = ((curr_pose[:, 2:] > 0.5) * (prev_pose[:, 2:] > 0.5)).astype(np.float32)
    if mask.sum() > 0:
        new_vel = np.sum(mask * diff, axis=0) / np.clip(np.sum(mask, axis=0), 1, None)
        return momentum * velocity + (1.0 - momentum) * new_vel

    else:
        return decay * velocity


def distance_poses_2d(pose1, pose2, thr=0.5):
    """ Compute the 2D distance between two poses.
    
    # Arguments
        pose1 and pose2: array (J, 3) including visibility

    # Returns
        Scalar value of the average distance between the two poses,
        considering only visible joints.
    """
    diff = pose1[:, 0:2] - pose2[:, 0:2]
    mask = ((pose1[:, 2:] > thr) * (pose2[:, 2:] > thr)).astype(np.float32)
    if np.sum(mask) >= 1:
        return np.sum(np.sqrt(np.sum(np.square(mask * diff), axis=0))) / np.sum(mask)
    else:
        return 99999


def preprocess_alphapose_predictions(annot_alphapose, frame_ids=None, max_num_people=None, verbose=False):
    """Given a dictionary of predictions from alphapose, arrange them into
    a single array with a fixed number of frames and detections.

    # Arguments
        annot_alphapose: dictionary with alphapose predictions, see 'load_alphapose_tracking_results()'
        max_num_people: integer or None. If given, try to match the predictions for each frame
            to the maximum number of people, given the previous frame. This can be useful if the
            number of people in one recording is previously known. If None, use the maximum
            number of detected people in the sequence.

    # Returns
        An array with shape (T, N, 17, 3), where 'T=len(annot_alphapose)' and N is
        the minimum between number of tracked people and 'max_num_people'.
    """
    annot_alphapose = copy.deepcopy(annot_alphapose)
    images_set = set()
    idx_set = set()

    for img_key in annot_alphapose.keys():
        images_set.add(img_key)
        for person_id in annot_alphapose[img_key].keys():
            idx_set.add(person_id)

    images_set = sorted(images_set)
    idx_set = sorted(idx_set)

    if verbose:
        print (f'Found {len(images_set)} images with predictions from AlphaPose with idx:', idx_set)

    T = len(images_set)
    if max_num_people is not None:
        N = min(len(idx_set), max_num_people)
    else:
        max_N = 0
        for t in range(T):
            max_N = max(max_N, len(annot_alphapose[list(annot_alphapose.keys())[t]].keys()))
        N = max_N
        print (f'AlphaPose:: found max {N} predictions per frame from AlphaPose!')

    pose2d = np.zeros((T, N, 17, 3), np.float32)
    prev_valid_pose_tidx = np.zeros((N,), int)
    curr_pose_velocity = np.zeros((N, 2), np.float32)
    #already_tracked = np.zeros((N,), int)
    mapping_person_id_to_idx = {}

    for t in range(T):
        img_key = images_set[t]
        n_tracked = np.zeros((N,), int)

        if t == 0:
            # In the first frame, assign all possible detections to one index
            # and initialize the buffer for previous poses
            keylist = sorted(list(annot_alphapose[img_key].keys()))
            ik = -1
            for ik in range(min(N, len(keylist))):
                mapping_person_id_to_idx[keylist[ik]] = ik
                pose2d[t, ik] = annot_alphapose[img_key][keylist[ik]]
                prev_valid_pose_tidx[ik] = t
                n_tracked[ik] = 1
                #already_tracked[ik] = 1

            # In the case we have less predictions in the first frame than in the full video, init them
            idx2 = keylist[ik] + 1
            for ik2 in range(ik+1, N):
                mapping_person_id_to_idx[idx2] = ik2
                idx2 += 1
            
        else:
            # After the first frame:
            #  1. For each person id already being tracked, check if there is
            #     a new prediction in the current frame. If so, copy it and
            #     remove it from the dictionary of annotations.
            for person_id in mapping_person_id_to_idx.keys():
                if person_id in annot_alphapose[img_key].keys():
                    n = mapping_person_id_to_idx[person_id]
                    pose2d[t, n] = annot_alphapose[img_key][person_id]
                    curr_pose_velocity[n] = update_pose_velocity_2d(pose2d[t, n], pose2d[t - 1, n], curr_pose_velocity[n])
                    prev_valid_pose_tidx[n] = t
                    n_tracked[n] = 1
                    del annot_alphapose[img_key][person_id]

            #  2.1 Check if there are no more predictions to be assigned AND if
            #     there are missing poses in the buffer -> missing pose!
            if (len(annot_alphapose[img_key]) == 0) and (n_tracked.min() == 0):
                # Here we need to remove this tracklet from the assignment
                # `mapping_person_id_to_idx`, otherwise, this person ID can appear
                # in the next frames in a different person instance
                # (which unfortunately happens with AlphaPose)
                # The exception to this is when the nth instance was NEVER tracked before, from `already_tracked`
                inv_mapping = {v: k for k, v in mapping_person_id_to_idx.items()}
                for ik in np.arange(N)[n_tracked == 0]:
                    #if already_tracked[ik]:
                    inv_mapping[ik] = -999 # reset tracking id (key in the fw mapping)
                mapping_person_id_to_idx = {v: k for k, v in inv_mapping.items()}
            
            #  2.2 Check if there are still predictions to be assigned AND if
            #     there are missing poses in the buffer
            elif n_tracked.min() == 0:
                #  3. In that case, get the indices of people not tracked and
                #     the corresponding previous valid pose(s).
                n_indices = np.arange(N)[n_tracked == 0]
                t_indices = prev_valid_pose_tidx[n_indices]

                #  4. Build two arrays, one with the previous valid poses
                #     (already tracked), and another one with the predictions
                #     still not assigned
                prev_poses_ref = np.stack([pose2d[t_i, n_i] for t_i, n_i in zip(t_indices, n_indices)], axis=0)

                curr_keylist = list(annot_alphapose[img_key].keys())
                pred_poses = np.stack([annot_alphapose[img_key][p_id] for p_id in curr_keylist], axis=0)

                #  5. Match both arrays with the Hungarian Algorithm and
                #     assign the closest ones to the poses buffer.
                #     Only assign the new pose if it is inside a `tracking region`.
                #     This assignment only happens if the `pose_candidate` (closest one)
                #     is lying inside a `tracking region` that is defined by the previous
                #     velocity of that person times the number of passed frames since the
                #     person was last tracked times a factor of 3.
                #     In the case of new assignment, also updates the list of previous tracked poses.
                prev_idx, pred_idx = linear_kpts_assignment(prev_poses_ref, pred_poses)
                for ref_i, pre_i in zip(prev_idx, pred_idx):
                    n = n_indices[ref_i]

                    pose_candidate = annot_alphapose[img_key][curr_keylist[pre_i]]
                    dist_to_prev_pose = distance_poses_2d(pose_candidate, prev_poses_ref[ref_i])
                    dt = t - prev_valid_pose_tidx[n]
                    ref_thr = 3 * dt * np.sqrt(np.sum(np.square(curr_pose_velocity[n])))

                    if dist_to_prev_pose < ref_thr:
                        pose2d[t, n] = pose_candidate
                        prev_valid_pose_tidx[n] = t

                        keys_to_remove = list(filter(lambda k: mapping_person_id_to_idx[k] == n, mapping_person_id_to_idx.keys()))
                        for kr in keys_to_remove:
                            del mapping_person_id_to_idx[kr]
                        mapping_person_id_to_idx[curr_keylist[pre_i]] = n
                        
                        del annot_alphapose[img_key][curr_keylist[pre_i]]

    if frame_ids is not None:
        pose2d = pose2d[frame_ids]

    return pose2d


def format_annotations_in_array(annot_alphapose, frame_ids=None, max_num_people=None, verbose=False):
    """Given a dictionary of predictions from alphapose, arrange all predictions into a single
    array with a fixed number of frames and detections.

    # Arguments
        annot_alphapose: dictionary with alphapose predictions, see 'load_alphapose_tracking_results()'
        frame_ids: indices to be considered, or None to use all
        max_num_people: integer or None. If given, try to match the predictions for each frame
            to the maximum number of people, given the previous frame. This can be useful if the
            number of people in one recording is previously known. If None, use the maximum
            number of detected people in the sequence.

    # Returns
        An array with shape (T, N, 17, 3), where 'T=len(frame_ids)' and N is
        the minimum between number of tracked people and 'max_num_people'.
    """
    images_set = set()
    idx_set = set()
    annot_alphapose = copy.deepcopy(annot_alphapose)

    for img_key in annot_alphapose.keys():
        images_set.add(img_key)
        for person_id in annot_alphapose[img_key].keys():
            idx_set.add(person_id)

    images_set = sorted(images_set)
    idx_set = sorted(idx_set)

    if verbose:
        print (f'Found {len(images_set)} images with predictions from AlphaPose with idx:', idx_set)


    T = len(images_set)
    if max_num_people is not None:
        N = min(len(idx_set), max_num_people)
    else:
        max_N = 0
        for t in range(T):
            max_N = max(max_N, len(annot_alphapose[list(annot_alphapose.keys())[t]].keys()))
        N = max_N
        print (f'AlphaPose:: found max {N} predictions per frame from AlphaPose!')

    pose2d = np.zeros((T, N, 17, 3), np.float32)
    prev_valid_pose_tidx = np.zeros((N,), int)
    curr_pose_velocity = np.zeros((N, 2), np.float32)
    mapping_person_id_to_idx = {}

    for t in range(T):
        img_key = images_set[t]
        if t == 0:
            # In the first frame, assign all possible detections to one index
            # and initialize the buffer for previous poses
            keylist = sorted(list(annot_alphapose[img_key].keys()))
            for ik in range(min(N, len(keylist))):
                mapping_person_id_to_idx[keylist[ik]] = ik
                pose2d[t, ik] = annot_alphapose[img_key][keylist[ik]]
                prev_valid_pose_tidx[ik] = t
        else:
            # After the first frame:
            #  1. For each person id already being tracked, check if there is
            #     a new prediction in the current frame. If so, copy it and
            #     remove it from the dictionary of annotations.
            n_tracked = np.zeros((N,), int)
            for person_id in mapping_person_id_to_idx.keys():
                if person_id in annot_alphapose[img_key].keys():
                    n = mapping_person_id_to_idx[person_id]
                    pose2d[t, n] = annot_alphapose[img_key][person_id]
                    curr_pose_velocity[n] = update_pose_velocity_2d(pose2d[t, n], pose2d[t - 1, n], curr_pose_velocity[n])
                    prev_valid_pose_tidx[n] = t
                    n_tracked[n] = 1
                    del annot_alphapose[img_key][person_id]
            #  2. Check if there are still predictions to be assigned AND if
            #     there are missing poses in the buffer
            if (len(annot_alphapose[img_key]) > 0) and (n_tracked.min() == 0):
                #  3. In that case, get the indices of people not tracked and
                #     the corresponding previous valid pose(s).
                n_indices = np.array(range(N))[n_tracked == 0]
                t_indices = prev_valid_pose_tidx[n_indices]

                #  4. Build two arrays, one with the previous valid poses
                #     (already tracked), and another one with the predictions
                #     still not assigned
                prev_poses_ref = np.stack([pose2d[t_i, n_i] for t_i, n_i in zip(t_indices, n_indices)], axis=0)

                curr_keylist = list(annot_alphapose[img_key].keys())
                pred_poses = np.stack([annot_alphapose[img_key][p_id] for p_id in curr_keylist], axis=0)

                #  5. Match both arrays with the Hungarian Algorithm and
                #     assign the closest ones to the poses buffer.
                #     Only assign the new pose if it is inside a `tracking region`.
                #     This assignment only happens if the `pose_candidate` (closest one)
                #     is lying inside a `tracking region` that is defined by the previous
                #     velocity of that person times the number of passed frames since the
                #     person was lasted tracked times a factor of 3.
                #     In the case of new assignment, also updates the list of previous tracked poses.
                prev_idx, pred_idx = linear_kpts_assignment(prev_poses_ref, pred_poses)
                for ref_i, pre_i in zip(prev_idx, pred_idx):
                    n = n_indices[ref_i]
                    pose_candidate = annot_alphapose[img_key][curr_keylist[pre_i]]
                    dist_to_prev_pose = distance_poses_2d(pose_candidate, prev_poses_ref[ref_i])
                    dt = t - prev_valid_pose_tidx[n]

                    if dist_to_prev_pose < 3 * dt * np.max(curr_pose_velocity[n]):
                        pose2d[t, n] = pose_candidate
                        prev_valid_pose_tidx[n] = t

                        keys_to_remove = list(filter(lambda k: mapping_person_id_to_idx[k] == n, mapping_person_id_to_idx.keys()))
                        for kr in keys_to_remove:
                            del mapping_person_id_to_idx[kr]
                        mapping_person_id_to_idx[curr_keylist[pre_i]] = n
                        
                        del annot_alphapose[img_key][curr_keylist[pre_i]]

    if frame_ids is not None:
        pose2d = pose2d[frame_ids]

    return pose2d

