import sys

from .config import ConfigContext
from .config import parse_args

from .datautils import H3DHCustomSequenceData
from .predict import Predictor


def build_internet_dataloader(data_path, smpl_model_parameters_path, fov=60,
        resize_factor=1, start_frame=0, end_frame=-1, step_frame=1,
        use_hrnet_pose=True,
        joint_confidence_thr=0.49,
        depth_path='DPT_midas21_monodepth',
        smpl_pred_path = "ROMP_Predictions",
        erode_segmentation_iters=0,
        erode_backmask_iters=0,
        renormalize_depth=True,
        post_process_depth=True,
        filter_2dpose=True,
        filter_min_cutoff=0.01,
        filter_beta=25,
        ):
    frame_ids = range(start_frame, end_frame, step_frame)

    print (f'DEBUG:: use_hrnet_pose>> ', use_hrnet_pose)
    print (f'DEBUG:: joint_confidence_thr>> ', joint_confidence_thr)
    print (f'DEBUG:: erode_segmentation_iters>> ', erode_segmentation_iters)
    print (f'DEBUG:: erode_backmask_iters>> ', erode_backmask_iters)
    print (f'DEBUG:: renormalize_depth>> ', renormalize_depth)
    print (f'DEBUG:: post_process_depth>> ', post_process_depth)
    print (f'DEBUG:: fov>> ', fov)

    # W = 1920 # TODO: read intrinsics from input (add option)
    # import numpy as np
    # focal=(1178.827, 1134.778)
    # center=(960.0, 540.0)
    # cam_K = np.array([
    #    [focal[0], 0, center[0]],
    #    [0, focal[1], center[1]],
    #    [0, 0, 1],
    # ], np.float32)

    dataset = H3DHCustomSequenceData(
        data_root=data_path,
        cam_K=None, fov=fov,
        #cam_K=cam_K,
        frame_ids=frame_ids,
        use_hrnet_pose=use_hrnet_pose,
        joint_confidence_thr=joint_confidence_thr,
        depth_path=depth_path,
        smpl_pred_path=smpl_pred_path,
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

    return dataset

if __name__ == '__main__':
    parsed_args = parse_args(sys.argv[1:]) if len(sys.argv) > 1 else None
    with ConfigContext(parsed_args):
        kargs = {}
        for key, value in parsed_args.smpl.items():
            kargs[key] = value

        for key, value in parsed_args.data.items():
            kargs[key] = value
        print ("Info: writing output to ", parsed_args.output_path)

        dataset = build_internet_dataloader(
            resize_factor=parsed_args.resize_factor,
            erode_segmentation_iters=parsed_args.erode_segmentation_iters,
            erode_backmask_iters=parsed_args.erode_backmask_iters,
            renormalize_depth=parsed_args.renormalize_depth,
            post_process_depth=parsed_args.post_process_depth,
            **kargs)

        print (f'DEBUG:: parsed_args', parsed_args)
        predictor = Predictor(dataset, output_path=parsed_args.output_path,
                parsed_args=parsed_args, **kargs)
        log = predictor.run()

        print ('scale_factor', log['stage1_optvar']['scale_factor'].squeeze())
        print ('min_z', log['stage1_optvar']['min_z'].squeeze())
        print ('max_z', log['stage1_optvar']['max_z'].squeeze())
