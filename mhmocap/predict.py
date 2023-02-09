import os

import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from .io import io_mkdir
from .datautils import H3DHCustomSequenceData
from .optimizer import SMPLDepthSequenceOptimizer
from .transforms import camera_projection

output_plots_ext = "png" # "pdf"
plot_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'gold', 'olive', 'deeppink', 'darkorange', 'navy']


# {0,  "Nose"},
# {1,  "LEye"},
# {2,  "REye"},
# {3,  "LEar"},
# {4,  "REar"},
# {5,  "LShoulder"},
# {6,  "RShoulder"},
# {7,  "LElbow"},
# {8,  "RElbow"},
# {9,  "LWrist"},
# {10, "RWrist"},
# {11, "LHip"},
# {12, "RHip"},
# {13, "LKnee"},
# {14, "Rknee"},
# {15, "LAnkle"},
# {16, "RAnkle"},
alphapose_links = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [5, 7],
    [7, 9],
    [6, 8],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
]


def save_visualization_init_data(output_path, dataset, init_optvar, loss_2d, joints_thr=0.5):
    import matplotlib.colors as mcolors
    plt.rc('font', size=16)
    plt.rcParams["figure.figsize"] = (16, 6)
    
    fig, axs = plt.subplots(1, 1)
    axs.plot(np.log(loss_2d), c='r', label='Pose 2D loss')
    plt.ylabel('log(loss)')
    fig.legend()
    axs.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, 'fig_optim_curves_init.' + output_plots_ext), pad_inches=0, dpi=300)
    plt.close(fig)

    vis_path = os.path.join(output_path, 'vis_init')
    Path(vis_path).mkdir(parents=True, exist_ok=True)

    plt.rcParams["figure.figsize"] = (24, 8)
    for i in tqdm(range(len(dataset))):
        if i % 25 == 0: #95:
           break

        output = dataset[i]
        image = output['images']
        seg_mask = output['seg_mask'] # (N, H, W)
        backmask = output['backmasks'] # (H, W)
        scale_factor = init_optvar['scale_factor'][0]
        poses_T = init_optvar['poses_T'][i]
        poses_smpl = init_optvar['poses_smpl'][i] # (N, 72)
        betas_smpl = init_optvar['betas_smpl'][0] # (N, 10)
        valid_smpl = init_optvar['valid_smpl'][0] # (N, 1)
        ref_pose2d = init_optvar['pose2d'][i] # (N, J, 3)

        # Check validity for each data modality
        h, w = seg_mask.shape[1:]
        #valid_seg_mask = np.sum(seg_mask, axis=(1, 2)) > (0.005 * h * w)
        valid_smpl = valid_smpl[:, 0] > 0.7

        #avg_joint_conf = np.mean(ref_pose2d[..., 2], axis=1)
        #num_valid_joints = np.sum(ref_pose2d[..., 2] > joints_thr, axis=1)
        valid_2dpose = np.sum(ref_pose2d[..., 2] > joints_thr, axis=1) >= 3

        #print (f'{i}: ', list(valid_seg_mask), list(valid_smpl), list(valid_2dpose), list(avg_joint_conf), list(num_valid_joints))

        results = dataset.SMPLPY(betas=betas_smpl, poses=poses_smpl)
        verts = results['verts'].cpu().numpy()
        joints_smpl_alphapose = results['joints_alphapose'].cpu().numpy()
        optim_verts3d = scale_factor * verts  + poses_T
        optim_joints3d = scale_factor * joints_smpl_alphapose  + poses_T

        N = joints_smpl_alphapose.shape[0]
        optim_verts2d = camera_projection(optim_verts3d.reshape((-1, 3)), dataset.cam['K']).reshape((N, -1, 2))
        optim_joints2d = camera_projection(optim_joints3d.reshape((-1, 3)), dataset.cam['K']).reshape((N, -1, 2))

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(image)
        axs[1].imshow(image)
        for n in range(len(ref_pose2d)):
            rgb_color = np.array([[mcolors.to_rgb(plot_colors[n])]])
            if n == 0:
                seg_img = seg_mask[n, :, :, np.newaxis] * rgb_color
            else:
                seg_img += seg_mask[n, :, :, np.newaxis] * rgb_color

            if valid_smpl[n]:
                axs[1].scatter(optim_verts2d[n, :, 0], optim_verts2d[n, :, 1], marker='o', color=plot_colors[n], alpha=0.05)

            for smpl_p2d, p2d in zip(optim_joints2d[n], ref_pose2d[n]):
                if valid_2dpose[n] and (p2d[2] > joints_thr):
                    axs[0].scatter(p2d[0], p2d[1], marker='x', color=plot_colors[n], alpha=1.0)
                if valid_smpl[n]:
                    axs[1].scatter(smpl_p2d[0], smpl_p2d[1], marker='^', color='w', alpha=1.0)

            for link in alphapose_links:
                p1 = ref_pose2d[n, link[0]]
                p2 = ref_pose2d[n, link[1]]
                if (p1[2] > joints_thr) and (p2[2] > joints_thr):
                    axs[0].plot([p1[0], p2[0]], [p1[1], p2[1]], color=plot_colors[n], lw=3)

        seg_img = (np.clip(backmask[..., np.newaxis], 0.3, 1) * image + 0.7 * 255 * seg_img).astype(np.uint8)
        axs[2].imshow(seg_img)

        fig.tight_layout()
        fig.savefig(os.path.join(vis_path, f'vis_{i:04d}.png'), pad_inches=0, dpi=240)
        plt.close(fig)


def save_visualization_stage1(output_path, dataset, stage1_optvar, poses2d, log):
    plt.rc('font', size=16)
    plt.rcParams["figure.figsize"] = (16, 6)

    loss_pose24j = np.stack([v['loss_pose24j'] for v in log], axis=0)
    loss_depth = np.stack([v['loss_depth'] for v in log], axis=0)
    loss_silhouette = np.stack([v['loss_silhouette'] for v in log], axis=0)
    reg_vel = np.stack([v['reg_vel'] for v in log], axis=0)
    reg_filter_verts = np.stack([v['reg_filter_verts'] for v in log], axis=0)
    reg_ref_poses = np.stack([v['reg_ref_poses'] for v in log], axis=0)
    reg_scale = np.stack([v['reg_scale'] for v in log], axis=0)
    reg_contact = np.stack([v['reg_contact'] for v in log], axis=0)
    reg_foot_sliding = np.stack([v['reg_foot_sliding'] for v in log], axis=0)

    fig, axs = plt.subplots(1, 1)
    axs.plot(np.log(loss_pose24j), c='r', label='Pose 2D loss')
    axs.plot(np.log(loss_depth), c='b', label='Depth loss')
    axs.plot(np.log(loss_silhouette), c='g', label='Silhouette loss')
    axs.plot(np.log(reg_vel), c='darkorange', label='Reg. 3D Pose Velocity')
    axs.plot(np.log(reg_filter_verts), c='darkgreen', label='Reg. 3D Vert. Smooth')
    axs.plot(np.log(reg_ref_poses), c='m', label='Reg. Ref. Poses')
    axs.plot(np.log(reg_scale), c='y', label='Reg. Scale')
    axs.plot(np.log(reg_contact), c='k', label='Reg. Contact')
    axs.plot(np.log(reg_foot_sliding), c='gold', label='Reg. Food Slid.')
    
    plt.ylabel('log(loss)')
    fig.legend()
    axs.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, 'fig_optim_curves_stage1.' + output_plots_ext), pad_inches=0, dpi=300)
    plt.close(fig)
    
    vis_path = os.path.join(output_path, 'vis_stage1')
    Path(vis_path).mkdir(parents=True, exist_ok=True)

    images = []
    depths = []
    backmasks = []
    cam_smpl = []
    optim_verts3d_all = []
    valid_smpl = []

    scene_depth_min = 9999
    scene_depth_max = 0
    for i in range(len(dataset)):
        scene_depth_min = min(scene_depth_min, stage1_optvar['min_z'][i])
        scene_depth_max = max(scene_depth_max, stage1_optvar['max_z'][i])

    scale_factor = stage1_optvar['scale_factor'][0]
    print (f'\nDEBUG:: >> stage1_optvar:')
    print (f'        >> scale_factor:', scale_factor)
    print (f'        >> scene_depth_min / scene_depth_max:', scene_depth_min, scene_depth_max)

    plt.rcParams["figure.figsize"] = (24, 8)
    for i in tqdm(range(len(dataset))):
        output = dataset[i]
        image = output['images']
        poses_T = stage1_optvar['poses_T'][i]
        poses_smpl = stage1_optvar['poses_smpl'][i]
        ref_pose2d = poses2d[i]
        min_z = stage1_optvar['min_z'][i]
        max_z = stage1_optvar['max_z'][i]
        depth = 1.0 / (output['depths'] * (1.0 / min_z - 1.0 / max_z) + 1.0 / max_z)

        results = dataset.SMPLPY(betas=stage1_optvar['betas_smpl'][0], poses=poses_smpl)
        verts = results['verts'].cpu().numpy()
        joints_smpl_alphapose = results['joints_alphapose'].cpu().numpy()
        optim_verts3d = scale_factor * verts  + poses_T
        optim_joints3d = scale_factor * joints_smpl_alphapose  + poses_T

        N = joints_smpl_alphapose.shape[0]
        optim_verts2d = camera_projection(optim_verts3d.reshape((-1, 3)), dataset.cam['K']).reshape((N, -1, 2))
        optim_joints2d = camera_projection(optim_joints3d.reshape((-1, 3)), dataset.cam['K']).reshape((N, -1, 2))

        images.append(image)
        depths.append(depth)
        backmasks.append(output['backmasks'])
        cam_smpl.append(output['cam_smpl'])
        optim_verts3d_all.append(optim_verts3d)
        valid_smpl.append(output['valid_smpl'])

        if i < 20:
            fig, axs = plt.subplots(1, 3)

            axs[0].imshow(image)
            axs[1].imshow(image)
            axs[2].imshow(np.log(depth), vmin=float(np.log(scene_depth_min)), vmax=float(np.log(scene_depth_max)))
            for n in range(len(ref_pose2d)):
                for link in alphapose_links:
                    p1 = optim_joints2d[n, link[0]]
                    p2 = optim_joints2d[n, link[1]]
                    axs[0].plot([p1[0], p2[0]], [p1[1], p2[1]], color=plot_colors[n], lw=3)

                for ps, p2d in zip(optim_joints2d[n], ref_pose2d[n]):
                    axs[0].scatter(ps[0], ps[1], marker='v', color=plot_colors[n], alpha=1.0)
                    if p2d[2] > 0.5:
                        axs[0].scatter(p2d[0], p2d[1], marker='^', color='w', alpha=1.0)
                        axs[0].plot([ps[0], p2d[0]], [ps[1], p2d[1]], color='w')
                
                axs[1].scatter(optim_verts2d[n, :, 0], optim_verts2d[n, :, 1], marker='.', color=plot_colors[n], alpha=0.05)

            fig.tight_layout()
            fig.savefig(os.path.join(vis_path, f'vis_{i:04d}.png'), pad_inches=0, dpi=240)
            plt.close(fig)

    vis_data_stage1 = {
        'images': np.stack(images, axis=0),
        'depths': np.stack(depths, axis=0),
        'backmasks': np.stack(backmasks, axis=0),
        'cam_smpl': np.stack(cam_smpl, axis=0),
        'cam': dataset.cam,
        'verts': np.stack(optim_verts3d_all, axis=0),
        'valid': np.stack(valid_smpl, axis=0),
        'pose2d': poses2d.copy(),
    }
    with open(os.path.join(output_path, 'visualization_data_stage1.pkl'), 'wb') as fip:
        pickle.dump(vis_data_stage1, fip)


class Predictor(object):
    def __init__(self, dataset, smpl_model_parameters_path, output_path, parsed_args,
            joint_confidence_thr=0.5, **kargs):
        io_mkdir(output_path)
        self.debug = True if (hasattr(parsed_args, 'debug') and parsed_args.debug) else False

        # Set the device
        if torch.cuda.is_available() and parsed_args.gpu >= 0:
            self.device_name = f"cuda:{parsed_args.gpu}"
        else:
            self.device_name = "cpu"
            print("WARNING: CPU only, this will be slow!")

        self.dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=parsed_args.batch_size,
            shuffle=parsed_args.shuffle,
            num_workers=parsed_args.num_workers,
            pin_memory=False)

        if len(self.dataloader) != (len(dataset) / parsed_args.batch_size):
            print (
                f"WARNING: Variable number of images in the batches. "
                f"len(dataset)={len(dataset)}, batch_size={parsed_args.batch_size}"
                )

        if 'Kd' in dataset.cam:
            cam_dist_coef = dataset.cam['Kd']
        else:
            cam_dist_coef = None

        self.optim_smpl = SMPLDepthSequenceOptimizer(
            device=self.device_name,
            image_size=dataset.image_size,
            num_frames=len(dataset),
            fov=dataset.cam['fov'],
            cam_K=dataset.cam['K'],
            cam_dist_coef=cam_dist_coef,
            proj2d_loss_coef=parsed_args.proj2d_loss_coef,
            depth_loss_coef=parsed_args.depth_loss_coef,
            silhouette_loss_coef=parsed_args.silhouette_loss_coef,
            reg_velocity_coef=parsed_args.reg_velocity_coef,
            reg_verts_filter_coef=parsed_args.reg_verts_filter_coef,
            reg_poses_coef=parsed_args.reg_poses_coef,
            reg_scales_coef=parsed_args.reg_scales_coef,
            reg_contact_coef=parsed_args.reg_contact_coef,
            reg_foot_sliding_coef=parsed_args.reg_foot_sliding_coef,
            smpl_model_parameters_path=smpl_model_parameters_path)

        self.dataset = dataset
        self.num_iter = parsed_args.num_iter
        self.output_path = output_path
        self.save_visualizations = parsed_args.save_visualizations
        self.joint_confidence_thr = joint_confidence_thr


    def run(self):
        # Pre-load the initial SMPL predictions and 2D keypoints
        init_keys = ['pose2d', 'poses_smpl', 'betas_smpl', 'valid_smpl']
        dataset = self.dataset

        init_data = {}
        for k in init_keys:
            init_data[k] = []

        for i, spl in enumerate(dataset):
            for k in init_keys:
                if i == 0:
                    init_data[k] = []
                init_data[k].append(spl[k])
                if i == len(dataset) - 1:
                    init_data[k] = np.stack(init_data[k], axis=0)

        init_log_loss = self.optim_smpl.init_optimized_variables(**init_data)
        init_optvar = self.optim_smpl.get_optimized_variables()
        init_optvar['pose2d'] = init_data['pose2d']
        with open(os.path.join(self.output_path, 'optvar_init.pkl'), 'wb') as fip:
            pickle.dump(init_optvar, fip)

        if self.save_visualizations:
            loss_2d = np.stack([v['loss_2d'] for v in init_log_loss], axis=0)
            save_visualization_init_data(self.output_path, dataset, init_optvar, loss_2d,
                    joints_thr=self.joint_confidence_thr)

        log = self.optim_smpl.fit(self.dataloader, num_iter=self.num_iter, verbose=True)
        stage1_optvar = self.optim_smpl.get_optimized_variables()

        with open(os.path.join(self.output_path, 'optvar_stage1.pkl'), 'wb') as fip:
            pickle.dump(stage1_optvar, fip)

        if self.save_visualizations:
            save_visualization_stage1(self.output_path, dataset, stage1_optvar, init_optvar['pose2d'], log)

        return {
            'init_log_loss': init_log_loss,
            'init_optvar': init_optvar,
            'stage1_log': log,
            'stage1_optvar': stage1_optvar,
        }


def build_studio_dataloader(data_path, ts_id, cam_id, smpl_model_parameters_path,
        resize_factor=1, start_frame=0, end_frame=-1, step_frame=1,
        depth_path='DPT_midas21_monodepth',
        erode_segmentation_iters=1,
        erode_backmask_iters=2,
        renormalize_depth=True,
        post_process_depth=True,
        ):

    if cam_id is not None:
        data_path = os.path.join(data_path, f'seq{ts_id}', f'cam{cam_id}')
    else:
        data_path = os.path.join(data_path, f'seq{ts_id}')
    frame_ids = range(start_frame, end_frame, step_frame)

    W = 1028
    focal=(0.582952201 * W, 0.582485139 * W)
    center=(0.501329839 * W, 0.349481702 * W)
    cam_K = np.array([
        [focal[0], 0, center[0]],
        [0, focal[1], center[1]],
        [0, 0, 1],
    ], np.float32)

    dataset = H3DHCustomSequenceData(
        data_root=data_path,
        cam_K=cam_K,
        frame_ids=frame_ids,
        depth_path=depth_path,
        resize_factor=resize_factor,
        smpl_model_parameters_path=smpl_model_parameters_path,
        erode_segmentation_iters=erode_segmentation_iters,
        erode_backmask_iters=erode_backmask_iters,
        renormalize_depth=renormalize_depth,
        post_process_depth=post_process_depth,
    )

    return dataset

