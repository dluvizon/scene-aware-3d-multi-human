import os
import sys

import numpy as np
import pickle

from .config import ConfigContext
from .config import parse_args

from .datautils import H3DHCustomSequenceData
from .datautils import load_mupots_sequence_metadata
from .predict import Predictor


def build_mupots_dataloader(data_path, ts_id, smpl_model_parameters_path,
        resize_factor=1, start_frame=0, end_frame=-1, step_frame=1,
        cam_K=None,
        use_hrnet_pose=True,
        joint_confidence_thr=0.49,
        depth_path='DPT_midas21_monodepth',
        erode_segmentation_iters=0,
        erode_backmask_iters=0,
        renormalize_depth=True,
        post_process_depth=True,
        filter_2dpose=True,
        filter_min_cutoff=0.01,
        filter_beta=25,
        ):

    print (f'DEBUG:: joint_confidence_thr>> ', joint_confidence_thr)
    print (f'DEBUG:: erode_segmentation_iters>> ', erode_segmentation_iters)
    print (f'DEBUG:: erode_backmask_iters>> ', erode_backmask_iters)
    print (f'DEBUG:: renormalize_depth>> ', renormalize_depth)
    print (f'DEBUG:: post_process_depth>> ', post_process_depth)

    data_path = os.path.join(data_path, f'TS{ts_id}')
    annot, occlu, cam_K_ts = load_mupots_sequence_metadata(os.path.join(data_path, 'images'))
    if cam_K is None:
        cam_K = cam_K_ts
    
    if end_frame > -1:
        frame_ids = range(start_frame, end_frame, step_frame)
    else:
        frame_ids = range(start_frame, annot.shape[0], step_frame)

    dataset = H3DHCustomSequenceData(
        data_root=data_path,
        cam_K=cam_K,
        frame_ids=frame_ids,
        use_hrnet_pose=use_hrnet_pose,
        joint_confidence_thr=joint_confidence_thr,
        depth_path=depth_path,
        resize_factor=resize_factor,
        smpl_model_parameters_path=smpl_model_parameters_path,
        erode_segmentation_iters=erode_segmentation_iters,
        erode_backmask_iters=erode_backmask_iters,
        renormalize_depth=renormalize_depth,
        post_process_depth=post_process_depth,
        filter_2dpose=filter_2dpose,
        filter_min_cutoff=filter_min_cutoff,
        filter_beta=filter_beta,
    )

    num_frames = len(frame_ids)
    num_people = annot.shape[1]
    pose3d_gt = np.zeros((num_frames, num_people, 17, 3), np.float32)
    pose3d_univ_gt = np.zeros((num_frames, num_people, 17, 3), np.float32)
    visibility = np.zeros((num_frames, num_people, 17, 1), np.float32)
    for f in range(num_frames):
        for i in range(num_people):
            pose3d_gt[f, i] = annot[frame_ids[f], i]['annot3'][0, 0].T / 1000.
            pose3d_univ_gt[f, i] = annot[frame_ids[f], i]['univ_annot3'][0, 0].T / 1000.
            visibility[f, i] = (occlu[frame_ids[f], i].T == 0).astype(np.float32)

    return dataset, pose3d_gt, pose3d_univ_gt, visibility


if __name__ == '__main__':
    parsed_args = parse_args(sys.argv[1:]) if len(sys.argv) > 1 else None
    with ConfigContext(parsed_args):
        kargs = {}
        for key, value in parsed_args.smpl.items():
            kargs[key] = value

        for key, value in parsed_args.data.items():
            kargs[key] = value
        output_path = os.path.join(parsed_args.output_path, f"TS{parsed_args.ts_id}")
        print ("Info: writing output to ", output_path)

        dataset, pose3d_gt, pose3d_univ_gt, visibility = build_mupots_dataloader(
            ts_id=parsed_args.ts_id,
            resize_factor=parsed_args.resize_factor,
            erode_segmentation_iters=parsed_args.erode_segmentation_iters,
            erode_backmask_iters=parsed_args.erode_backmask_iters,
            renormalize_depth=parsed_args.renormalize_depth,
            post_process_depth=parsed_args.post_process_depth,
            **kargs)

        predictor = Predictor(dataset, output_path=output_path, parsed_args=parsed_args, **kargs)
        log = predictor.run()

        # Save ground truth annotation from MuPoTs
        annot_mupots = {
            'pose3d_gt': pose3d_gt,
            'pose3d_univ_gt': pose3d_univ_gt,
            'visibility': visibility,
        }
        with open(os.path.join(output_path, 'mupots_annot.pkl'), 'wb') as fip:
            pickle.dump(annot_mupots, fip)
