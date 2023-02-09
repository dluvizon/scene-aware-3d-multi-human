import os

import numpy as np
from tqdm import tqdm

import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes

from .smpl import SMPL
from .losses import build_avg_depth_loss_fn
from .losses import build_masked_mse_loss_fn
from .one_euro_filter import OneEuroFilter
from .transforms import camera_projection_torch
from .transforms import camera_inverse_projection_torch
from .transforms import softplus
from .transforms import compute_calibration_matrix
from .transforms import get_focal
from .io import save_image
from .morphology import Erode2D
from .fhsog import aggegrate_scene_geometry_median
from .utils import postprocess_depthmap
from .utils import fillin_values


class SMPLOptimizerBase(object):
    """A base class for SMPL model optimization.
    """
    def __init__(self, device=None,
                 smpl_model_parameters_path='model_data/parameters',
                 smpl_J_reg_extra_path='J_regressor_extra.npy',
                 smpl_J_reg_h37m_path='J_regressor_h36m.npy',
                 smpl_J_reg_alphapose_path='SMPL_AlphaPose_Regressor_RMSprop_6.npy',
                 smpl_sparse_joints_key='joints_alphapose',
                 pose24j_weights=None,
                 pose17j_weights=None,
                 ):
        """
        # Arguments
            device: torch device or None for automatic selection
            smpl_model_parameters_path: string with the path to the SMPL parameters
            smpl_J_reg_extra_path: relative path of the regressor weights to extra joints
            smpl_J_reg_h37m_path: relative path of the regressor weights to the h36m layout
            pose24j_weights: list or 1-D numpy array with the weights for the 3D joints
                in the J joints layout, used during optimization. If none, assign a pre-defined
                weights to each joint.
        """

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
                print("WARNING: CPU only, this will be slow!")
        else:
            self.device = device

        # Setup SMPL model
        self.smpl_model_parameters_path = os.path.abspath(smpl_model_parameters_path)
        smpl_J_reg_extra_path = os.path.join(smpl_model_parameters_path, smpl_J_reg_extra_path)
        smpl_J_reg_h37m_path = os.path.join(smpl_model_parameters_path, smpl_J_reg_h37m_path)
        smpl_J_reg_alphapose_path = os.path.join(smpl_model_parameters_path, smpl_J_reg_alphapose_path)
        self.SMPLPY = SMPL(smpl_model_parameters_path,
            J_reg_extra9_path=smpl_J_reg_extra_path,
            J_reg_h36m17_path=smpl_J_reg_h37m_path,
            J_reg_alphapose_path=smpl_J_reg_alphapose_path).to(device)
        self.faces_smpl = torch.tensor(
            self.SMPLPY.faces[np.newaxis, :].astype(np.int32), device=self.device)
        self.smpl_sparse_joints_key = smpl_sparse_joints_key

        if pose24j_weights is None:
            pose24j_weights = [
                1, #1.0, # 0
                1, #1.0, # 1
                1, #1.0, # 2
                1, #1.0, # 3
                1, #3.0, # 4
                1, #3.0, # 5
                1, #1.0, # 6
                1, #5.0, # 7
                1, #5.0, # 8
                1, #1.0, # 9
                1, #2.0, # 10
                1, #2.0, # 11
                1, #1.0, # 12
                1, #1.0, # 13
                1, #1.0, # 14
                1, #3.0, # 15
                1, #1.0, # 16
                1, #1.0, # 17
                1, #5.0, # 18
                1, #5.0, # 19
                1, #3.0, # 20
                1, #3.0, # 21
                1, #2.0, # 22
                1, #2.0, # 23
            ]
        pose24j_weights = np.array(pose24j_weights, np.float32)
        pose24j_weights = len(pose24j_weights) * pose24j_weights / np.sum(pose24j_weights)
        self.pose24j_weights = torch.tensor(pose24j_weights[np.newaxis, :, np.newaxis], device=self.device)

        if pose17j_weights is None:
            pose17j_weights = [
                1.0, # 0
                1.0, # 1
                1.0, # 2
                1.0, # 3
                1.0, # 4
                1.0, # 5
                1.0, # 6
                1.0, # 7
                1.0, # 8
                1.0, # 9
                1.0, # 10
                1.0, # 11
                1.0, # 12
                1.0, # 13
                1.0, # 14
                1.0, # 15
                1.0, # 16
            ]
        pose17j_weights = np.array(pose17j_weights, np.float32)
        pose17j_weights = len(pose17j_weights) * pose17j_weights / np.sum(pose17j_weights)
        self.pose17j_weights = torch.tensor(pose17j_weights[np.newaxis, :, np.newaxis], device=self.device)


    def predict(self, poses_T, poses_smpl, betas_smpl, scale_factor):
        betas_smpl = torch.tensor(betas_smpl).to(self.device)
        poses_smpl = torch.tensor(poses_smpl).to(self.device)

        results = self.SMPLPY(betas=betas_smpl, poses=poses_smpl)
        pred_verts = results['verts'].cpu().detach().numpy()
        pred_joints = results[self.smpl_sparse_joints_key].cpu().detach().numpy()
        verts_smpl = scale_factor * pred_verts + poses_T
        pose24j_smpl = scale_factor * pred_joints + poses_T

        return verts_smpl, pose24j_smpl


class SMPLDepthSequenceOptimizer(SMPLOptimizerBase):
    """Implements an optimized for a full sequence by sampling frames in a
    temporal sequence.
    """
    def __init__(self,
                 image_size,
                 num_frames,
                 fov=60,
                 focal_length=None,
                 znear=1.0,
                 zfar=100.0,
                 cam_K=None,
                 cam_dist_coef=None,
                 proj2d_loss_coef=1.0,
                 depth_loss_coef=1.0,
                 silhouette_loss_coef=1.0,
                 reg_velocity_coef=1.0,
                 reg_verts_filter_coef=1.0,
                 reg_poses_coef=1.0,
                 reg_scales_coef=1.0,
                 reg_contact_coef=1.0,
                 reg_foot_sliding_coef=1.0,
                 joint_confidence_thr=0.5,
                 eps=1e-3,
                 **kargs):
        """Instantiate an optimized for SMPL.

        # Arguments
            image_size: tuple of integers (H, W)
            num_frames: integer, number of frames of the full sequence. If
                `num_frames=1`, proceed with a single frame approach, where
                the temporal losses are not computed.
            fov: float
            znear: float
            zfar: float
            cam_K: camera intrinsics in the form of a 3x3 matrix
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]. If none, infer the values
                from `image_size` and `focal_length`/`fov`.
            cam_dist_coef: Kp distortion coefficients, if given, are only used in the camera perspective projection.
            kargs: dictionary with base class arguments. See `SMPLOptimizerBase`.

        """
        super().__init__(**kargs)

        if focal_length is None:
            focal_length = get_focal(min(image_size), fov)

        if cam_K is None:
            self.cam_K = np.array([
                [focal_length, 0, image_size[1] / 2.0],
                [0, focal_length, image_size[0] / 2.0],
                [0, 0, 1]], dtype=np.float32)
        else:
            self.cam_K = cam_K.astype(np.float32)
        self.cam_dist_coef = cam_dist_coef

        self.znear = znear
        self.zfar = zfar
        self.R = torch.Tensor([[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]]).to(self.device)
        self.T = torch.Tensor([[0., 0., 0.]]).to(self.device)
        K = compute_calibration_matrix(self.znear, self.zfar, self.cam_K, image_size)
        K = torch.Tensor(K[np.newaxis]).to(self.device)

        # Setup Pytorch3D for differentiable raster and render
        self.cameras = FoVPerspectiveCameras(R=self.R, T=self.T, K=K, device=self.device)
        self.raster_settings = RasterizationSettings(
                image_size=image_size[::-1],
                blur_radius=1e-4,
                faces_per_pixel=8,
                perspective_correct=False)
        self.rasterizer = MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings)

        # Silhouette renderer
        silhouette_raster_settings = RasterizationSettings(
                image_size=image_size[::-1],
                blur_radius=2e-5,
                faces_per_pixel=4,
                perspective_correct=False)
        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=silhouette_raster_settings,
                ),
            shader=SoftSilhouetteShader(),
        )

        self.proj2d_loss_coef = proj2d_loss_coef
        self.depth_loss_coef = depth_loss_coef
        self.silhouette_loss_coef = silhouette_loss_coef
        self.reg_velocity_coef = reg_velocity_coef
        self.reg_verts_filter_coef = reg_verts_filter_coef
        self.reg_poses_coef = reg_poses_coef
        self.reg_scales_coef = reg_scales_coef
        self.reg_contact_coef = reg_contact_coef
        self.reg_foot_sliding_coef = reg_foot_sliding_coef
        self.joint_confidence_thr = joint_confidence_thr
        self.eps = eps

        #---------------------------------------------------------------------
        # Define the auxiliary variables
        #---------------------------------------------------------------------

        ## Constant values
        self.num_frames = num_frames
        self.img_w, self.img_h = image_size
        self.cam_intrinsics = torch.tensor(
            np.tile(self.cam_K[np.newaxis], (num_frames, 1, 1, 1)),
            device=self.device)
        self.min_delta_z = torch.tensor(1.0, device=self.device)

        # Adjust pose weights to temporal sequences
        self.pose_weights = self.pose17j_weights.unsqueeze(0)


    def init_optimized_variables(self, pose2d, poses_smpl, betas_smpl, valid_smpl, scale_factor=None, num_iter=100):
        """Initialize the optimized variables based on initial 2D keypoints
        and SMPL parameters estimated per frame.

        # Arguments
            pose2d: 2D poses array with shape (T, N, 24, 3), including visibility
            poses_smpl: array with shape (T, N, 72)
            betas_smpl: array with shape (T, N, 10)
            valid_smpl: array with shape (T, N, 1)
        """
        assert (pose2d.shape[:2] == poses_smpl.shape[:2] == betas_smpl.shape[:2] == valid_smpl.shape[:2]), (
            f'Error: invalid inputs {pose2d.shape}, {poses_smpl.shape}, {betas_smpl.shape}, {valid_smpl.shape}'
        )
        T, N = pose2d.shape[0:2]
        self.num_people = N

        ## Scale factor (pre-activated)
        if scale_factor is not None:
            xscale_factor = np.log(scale_factor) / np.log(1.1)
            self.xscale_factor = torch.tensor(xscale_factor[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32), device=self.device)
            self.optim_scale_factor = False
        else:
            self.xscale_factor = torch.tensor(np.zeros((1, self.num_people, 1, 1), dtype=np.float32), device=self.device, requires_grad=True)
            self.optim_scale_factor = True

        init_output = self.__init_global_poses(pose2d, poses_smpl, betas_smpl, num_iter)
        #print (f'DEBUG:: init poses_T.z:', init_output['poses_T'][..., -1])

        ## Global position in 3D for each person (initially estimated for scale=1)
        self.poses_T = torch.tensor(init_output['poses_T'], device=self.device, requires_grad=True)
        max_z = np.clip(np.max(init_output['poses_T'][..., 2:], axis=1), 2, None)

        ## SMPL parameters
        self.poses_smpl = torch.tensor(poses_smpl, device=self.device, requires_grad=True) # (T, N, 72)
        avg_betas = np.mean(betas_smpl, axis=0, keepdims=True)
        self.betas_smpl = torch.tensor(avg_betas, device=self.device, requires_grad=True) # (1, N, 10)
        self.betas_smpl_ref = torch.tensor(avg_betas, device=self.device) # (1, N, 10)
        self.valid_smpl = torch.tensor((valid_smpl > 0.7).astype(np.float32), device=self.device) # (T, N, 1)

        ## Scene parameters
        self.zmin_lin = torch.tensor(np.ones_like(max_z), device=self.device, requires_grad=True)
        self.zmax_lin = torch.tensor(2.0 * max_z, device=self.device, requires_grad=True)

        ## Auxiliary functions
        self.erode = torch.nn.Sequential(
            Erode2D(kernel_size=3),
            Erode2D(kernel_size=3),
        ).to(self.device)

        # Define auxiliary variables
        grid_x = np.linspace(0.5, self.img_w - 0.5, self.img_w)
        grid_y = np.linspace(0.5, self.img_h - 0.5, self.img_h)
        grid_xy = np.stack(np.meshgrid(grid_x, grid_y, indexing='xy'), axis=-1)
        self.grid_xy = torch.tensor(grid_xy[np.newaxis].astype(np.float32), device=self.device) # (1, H, W, 2)
        self.scene_depth = None
        self.scene_pcd = None
        self.poses_T_filtered = None
        self.verts_filtered = None

        return init_output['optim_log']


    def fit(self, dataloader,
            num_iter=250,
            min_cutoff1=0.01, # 0.001
            min_cutoff2=0.001, # 0.0001
            beta1=0.02, # 0.1,
            beta2=0.5, #0.7,
            update_filters_every=25,
            verbose=False):
        """Fit in the frame sequence provided by dataloader.
        In this optimization process, the losses are computed for each batch and
        accumulated for a full sequence of frames. Only after all the samples in
        the dataset are computed, the optimizer is evaluated and the weights are
        updated. For more information about how this works in pytorch, check:
        https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350/2

        # Arguments
            dataloader: An iterable object for loading the data.
            TODO
        """
        optim_variables_stage1 = [
            self.poses_T,
            self.poses_smpl,
            self.betas_smpl,
            self.zmin_lin,
            self.zmax_lin,
        ]
        if self.optim_scale_factor:
            optim_variables_stage1.append(self.xscale_factor)
        else:
            print (f'WARNING!!! Not optimizing scale_factor!')

        optimizer_stage1 = torch.optim.RMSprop(optim_variables_stage1, lr=0.01, alpha=0.5, momentum=0.9)
        scheduler_stage1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_stage1, gamma=0.99)

        lossfn_mse = torch.nn.MSELoss(reduction='sum')
        lossfn_reg = torch.nn.L1Loss(reduction='sum')
        lossfn_depth = build_avg_depth_loss_fn()
        lossfn_silhouette = build_masked_mse_loss_fn()
        optim_log = []

        pose_norm_fact = torch.tensor(
            np.array([[[[self.img_w, self.img_h]]]], np.float32), device=self.device) # (1, 1, 1, 2)

        def pose2d_loss_fn(pred, gt, mask):
            return lossfn_mse(mask * pred / pose_norm_fact, mask * gt / pose_norm_fact)

        cycles = range(num_iter)
        if verbose:
            cycles = tqdm(cycles)

        # Main optimization loop
        for cycle in cycles:
            optimizer_stage1.zero_grad()
            cache_log = []
            images_scene_list = []
            depths_scene_list = []
            backmasks_list = []

            # Update the filtered variables for temporal smoothness
            if (cycle >= 30) and (cycle % update_filters_every == 0):
                self.poses_T_filtered = self.one_euro_filter(self.poses_T, min_cutoff=min_cutoff1, beta=beta1).detach().clone()
                results = self.SMPLPY(
                        betas=self.betas_smpl.tile((self.num_frames, 1, 1)).view(-1, 10),
                        poses=self.poses_smpl.view(-1, 72))
                scale_factor = torch.pow(1.1, self.xscale_factor)
                local_verts_full = scale_factor * results['verts'].view(self.num_frames, self.num_people, -1, 3) # (T, N, V, 3)
                self.verts_filtered = self.one_euro_filter(
                        local_verts_full + self.poses_T,
                        min_cutoff=min_cutoff2, beta=beta2).detach().clone() # (T, N, V, 3)

            for data in dataloader: # Start an optimization cycle over the full video
                idx_data = {} # Indexed data variables
                for k in data.keys():
                    idx_data[k] = data[k].to(self.device)

                images_scene_list.append(idx_data['images'].cpu().detach().numpy())
                backmasks_list.append((idx_data['backmasks'] / 1.0).cpu().detach().numpy())
                idx_var = self.__eval_batch_optimized_variables(idx_data['idxs'])
                batch_size, N = idx_var['poses_smpl'].shape[0:2]

                pose2d_thr_scores = torch.ge(idx_data['pose2d'][..., 2:3], self.joint_confidence_thr).float() # (batch, N, J, 1)
                pose2d_valid = torch.ge(torch.sum(pose2d_thr_scores, dim=(2, 3)), 2).float() # (batch, N)
                smpl_valid = idx_var['valid_smpl'].float() # (batch, N, 1)
                mask_valid = torch.ge(
                    torch.sum(idx_data['seg_mask'], dim=(2, 3)),
                    0.005 * self.img_h * self.img_w).float() # (batch, N)

                ###-----------------------------------------------------------
                ### Compute 2D keypoints loss
                ###-----------------------------------------------------------
                joints_smpl_2d = camera_projection_torch(
                    idx_var['joints_smpl_abs'].view(batch_size * self.num_people, -1, 3),
                    idx_var['intrinsics'],
                    Kd=self.cam_dist_coef).view(batch_size, self.num_people, -1, 2)

                loss_pose24j = pose2d_loss_fn(joints_smpl_2d, idx_data['pose2d'][..., 0:2],
                        self.pose_weights * pose2d_thr_scores)

                ###-----------------------------------------------------------
                ### Compute Raster Depth loss
                ###-----------------------------------------------------------
                target_disp = idx_data['depths'] * (1.0 / idx_var['min_z'] - 1.0 / idx_var['max_z']) + 1.0 / idx_var['max_z']
                depths_scene_list.append((1.0 / target_disp).cpu().detach().numpy())
                faces_smpl = torch.tile(self.faces_smpl, (batch_size * self.num_people, 1, 1))
                meshes = Meshes(idx_var['verts_smpl_abs'].view(batch_size * self.num_people, -1, 3), faces_smpl)
                fragments = self.rasterizer(meshes)
                zbuf = fragments.zbuf[..., 0] # In the depth domain
                zbuf = zbuf.view(batch_size, self.num_people, self.img_h, self.img_w)
                zbuf_valid_mask = torch.gt(zbuf, 0) # (batch, N, h, w)

                eroded_seg = self.erode(idx_data['seg_mask'].view(batch_size * self.num_people, 1, self.img_h, self.img_w))
                eroded_seg = eroded_seg.view(batch_size, self.num_people, self.img_h, self.img_w)
                zbuf_supervision_mask = zbuf_valid_mask * eroded_seg # (batch, N, h, w)
                # Do not apply depth loss if we don't have a valid 2D pose
                zbuf_supervision_mask *= pose2d_valid.unsqueeze(-1).unsqueeze(-1)

                zbuf_disp = 1.0 / torch.clamp(zbuf + 0.2, self.eps)
                target_disp = target_disp.unsqueeze(1)
                loss_depth = lossfn_depth(zbuf_disp, target_disp, zbuf_supervision_mask)

                ###-----------------------------------------------------------
                ### Compute silhouette loss
                ###-----------------------------------------------------------
                silhouette_images = self.renderer_silhouette(meshes)
                silhouette_images = silhouette_images[..., 3].view(batch_size, self.num_people, self.img_h, self.img_w)
                # Order the persons from close to far
                pT_idx = torch.argsort(idx_var['poses_T'][..., 0, 2], dim=1) # Z component

                loss_silhouette = 0
                for j in range(batch_size):
                    #fname = os.path.join("./test", f'image_{j:03d}.png')
                    #save_image(idx_data['images'][j], fname)

                    acc_mask = torch.zeros_like(silhouette_images[0, 0], device=self.device)
                    for nj in range(N):
                        nj_s = pT_idx[j, nj]
                        sil_img = silhouette_images[j, nj_s]
                        seg_img = idx_data['seg_mask'][j, nj_s]

                        # fname = os.path.join("./test", f'silhouette_{j:03d}_{nj}.png')
                        # save_image(sil_img, fname)
                        #fname = os.path.join("./test", f'seg_{j:03d}_{nj}.png')
                        #save_image(seg_img, fname)
                        # diff = (1 - acc_mask) * ((sil_img - seg_img) ** 2)
                        # fname = os.path.join("./test", f'diff_{j:03d}_{nj}.png')
                        # save_image(diff, fname)

                        # Only apply silhouette loss if this sample has mask and 2d poses
                        apply_sil_loss = (mask_valid[j, nj] * pose2d_valid[j, nj]).cpu().detach().numpy()
                        if apply_sil_loss > 0:
                            loss_silhouette += lossfn_silhouette(sil_img, seg_img, 1 - acc_mask)
                        acc_mask = torch.gt(acc_mask + seg_img, 0).float()
                        #fname = os.path.join("./test", f'acc_mask_{j:03d}_{nj}.png')
                        #save_image(acc_mask, fname)

                ###-----------------------------------------------------------
                ### Compute contact and foot sliding loss terms
                ###-----------------------------------------------------------
                # Y axis points down, verts_idx_lower_y as (batch, N, 1, 3)
                reg_contact = 0
                reg_foot_sliding = 0
                if (self.scene_depth is not None) and (self.scene_pcd is not None):
                    global_verts = idx_var['verts_smpl_abs'] # (T, N, V, 3)
                    verts_idx_lower_y = torch.argmax(global_verts[..., 1:2], dim=2, keepdim=True).tile((1, 1, 1, 3)).long()
                    #verts_idx_lower_y = torch.argsort(global_verts[..., 1:2], dim=2, descending=True)[:, :, 0:10] # (T, N, X, 1)
                    low_verts = torch.gather(global_verts, 2, verts_idx_lower_y) # (T, N, X, 3)

                    # Compute the closest point in the scene point cloud
                    vert_pcd_dist = torch.sum(torch.pow(self.scene_pcd - low_verts, 2), -1) # (T, N, M)
                    #pcd_idx_closest_dist = torch.argmin(vert_pcd_dist, dim=-1) # (T, N)
                    num_points = 32
                    pcd_idx_closest_dist = torch.argsort(vert_pcd_dist, dim=-1)[..., :num_points]
                    pcd_idx_closest_dist = pcd_idx_closest_dist.view(
                                batch_size, self.num_people, num_points, 1).tile(1, 1, 1, 3) # (T, N, num_points, 3)
                    pcd_closest_pts = torch.gather(
                                self.scene_pcd.tile((batch_size, self.num_people, 1, 1)), 2, pcd_idx_closest_dist) # (T, N, num_points, 3)
                    pcd_closest_pts = torch.mean(pcd_closest_pts, dim=2, keepdim=True)

                    contact_dist_vertical = (pcd_closest_pts - low_verts)[..., 1:2] # (T, N, 1, 1)
                    #in_thr_contact_region = torch.gt(contact_dist_vertical, 0) * torch.le(contact_dist_vertical, contach_thr) # (T, N, 1, 1)
                    target_poses_T = idx_var['poses_T'].detach().clone()
                    target_poses_T[..., 1:2] += contact_dist_vertical + 0.02
                    reg_contact = lossfn_reg(idx_var['poses_T'], target_poses_T.detach().clone())
                    #reg_contact = contact_loss(idx_var['poses_T'] - target_poses_T.detach().clone())

                    # Takes the lowest verts at time t also from the previous frame t-1
                    contach_thr = 0.20 # x100 cm
                    in_thr_contact_region = torch.gt(contact_dist_vertical, -contach_thr)
                    low_verts_t = low_verts[1:] # (T-1, N, 1, 3)
                    in_thr_contact_region_t = in_thr_contact_region[1:] # (T-1, N, 1, 1)
                    low_verts_tm1 = torch.gather(global_verts[:-1], 2, verts_idx_lower_y[1:]) # (T-1, N, 1, 3)
                    reg_foot_sliding = lossfn_reg(
                            in_thr_contact_region_t * low_verts_t,
                            in_thr_contact_region_t * low_verts_tm1) \
                            / torch.clamp(torch.sum(in_thr_contact_region_t), 1)

                ###-----------------------------------------------------------
                ### SMPL parameters regularization
                ###-----------------------------------------------------------
                reg_ref_poses = lossfn_reg(
                        smpl_valid * idx_data['poses_smpl'],
                        smpl_valid * idx_var['poses_smpl'])
                reg_ref_poses += batch_size * lossfn_reg(self.betas_smpl, self.betas_smpl_ref)

                ###-----------------------------------------------------------
                ### Person scale regularization
                ###-----------------------------------------------------------
                reg_scale_avg = torch.square(torch.sum(idx_var['scale_factor'] - 1.0))
                reg_scale_person = torch.mean(torch.square(idx_var['scale_factor'] - 1.0))

                # Accumulate all the energy terms with coefficients
                loss_output = (self.proj2d_loss_coef * loss_pose24j
                        + self.depth_loss_coef * loss_depth
                        + self.silhouette_loss_coef * loss_silhouette
                        + self.reg_poses_coef * reg_ref_poses
                        + self.reg_scales_coef * reg_scale_person + (self.reg_scales_coef > 0) * reg_scale_avg
                        + self.reg_contact_coef * reg_contact
                        + self.reg_foot_sliding_coef * reg_foot_sliding
                )
                #print (f'DEBUG:: self.reg_foot_sliding_coef', self.reg_foot_sliding_coef)
                loss_output.backward()

                cache_log.append({
                    'loss_pose24j': loss_pose24j.cpu().detach().numpy(),
                    'loss_depth': loss_depth.cpu().detach().numpy(),
                    'loss_silhouette': loss_silhouette if isinstance(loss_silhouette, int) else loss_silhouette.cpu().detach().numpy(),
                    'reg_ref_poses': reg_ref_poses.cpu().detach().numpy(),
                    'reg_scale': (reg_scale_avg + reg_scale_person).cpu().detach().numpy(),
                    'reg_contact': reg_contact if isinstance(reg_contact, int) else reg_contact.cpu().detach().numpy(),
                    'reg_foot_sliding': reg_foot_sliding if isinstance(reg_foot_sliding, int) else reg_foot_sliding.cpu().detach().numpy(),
                })
                # End of a cycle

            ###-----------------------------------------------------------
            ### Temporal optimization
            ###-----------------------------------------------------------
            reg_vel = lossfn_mse(self.poses_T[1:], self.poses_T[:-1])
            loss_temp = self.reg_velocity_coef * reg_vel

            reg_filter_verts = 0
            if (self.poses_T_filtered is not None) and (self.verts_filtered is not None):
                results = self.SMPLPY(
                        betas=self.betas_smpl.tile((self.num_frames, 1, 1)).view(-1, 10),
                        poses=self.poses_smpl.view(-1, 72))
                scale_factor = torch.pow(1.1, self.xscale_factor)
                local_verts_full = scale_factor * results['verts'].view(self.num_frames, self.num_people, -1, 3) # (T, N, V, 3)
                global_verts_full = local_verts_full + self.poses_T
                reg_filter_verts = lossfn_mse(
                    global_verts_full[1:] - global_verts_full[:-1],
                    self.verts_filtered[1:] - self.verts_filtered[:-1])
                loss_temp += self.reg_verts_filter_coef * reg_filter_verts
            loss_temp.backward()

            ## Update the scene geometry (scene point cloud)
            if cycle >= 30:
                depths = np.concatenate(depths_scene_list, axis=0) # (T, H, W)
                images = np.concatenate(images_scene_list, axis=0) # (T, H, W, 3)
                backmasks = np.concatenate(backmasks_list, axis=0) # (T, H, W)
                ma_image, ma_depth, ma_mask = aggegrate_scene_geometry_median(depths, images, backmasks, 'median')
                self.scene_depth = postprocess_depthmap(ma_depth, ma_mask, use_bilateral_filter=True) # (H, W)
                self.update_scene_pointcloud(self.scene_depth, ma_mask)

            optimizer_stage1.step()
            scheduler_stage1.step()
            if len(cache_log) > 0:
                optim_log.append({})
                for k in cache_log[0].keys():
                    optim_log[-1][k] = np.mean([cache_log[i][k] for i in range(len(cache_log))])
                optim_log[-1]['reg_vel'] = reg_vel if isinstance(reg_vel, int) else reg_vel.cpu().detach().numpy()
                optim_log[-1]['reg_filter_verts'] = reg_filter_verts if isinstance(reg_filter_verts, int) else reg_filter_verts.cpu().detach().numpy()

        scene_mask = ma_mask.copy()
        scene_img = ma_image.copy()
        while (scene_mask.min() == 0):
            scene_img, scene_mask = fillin_values(scene_img, scene_mask, filter_size=11)
        self.scene_img = scene_img
        self.scene_mask = scene_mask

        return optim_log


    def update_scene_pointcloud(self, scene_depth, scene_mask):
        scene_depth = torch.tensor(scene_depth, device=self.device) # (H, W)
        scene_mask = torch.tensor(scene_mask, device=self.device)

        scene_pcd_uvd = torch.cat((self.grid_xy, scene_depth.unsqueeze(-1).unsqueeze(0)), dim=-1) # (1, H, W, 3)
        scene_pcd_uvd = scene_pcd_uvd.view(1, -1, 3) # (1, H * W, 3)
        scene_pcd = camera_inverse_projection_torch(scene_pcd_uvd,
            torch.tensor(self.cam_K[np.newaxis], device=self.device)).squeeze(0) # (H * W, 3)
        scene_pcd = scene_pcd[torch.gt(scene_mask.view(-1), 0.5), :]
        if self.scene_pcd is not None:
            del self.scene_pcd
        self.scene_pcd = scene_pcd.unsqueeze(0).unsqueeze(0) # (1, 1, M, 3)


    def get_optimized_variables(self):
        scale_factor = torch.pow(1.1, self.xscale_factor)
        min_z = softplus(self.zmin_lin)
        max_z = min_z.detach().clone() + self.min_delta_z + softplus(self.zmax_lin)

        return {
            'scale_factor': scale_factor.cpu().detach().numpy(),
            'poses_T': self.poses_T.cpu().detach().numpy(),
            'poses_smpl': self.poses_smpl.cpu().detach().numpy(),
            'betas_smpl': self.betas_smpl.cpu().detach().numpy(),
            'valid_smpl': self.valid_smpl.cpu().detach().numpy(),
            'min_z': min_z.cpu().detach().numpy(),
            'max_z': max_z.cpu().detach().numpy(),
            'scene_depth': self.scene_depth if hasattr(self, 'scene_depth') else None,
            'scene_img': self.scene_img if hasattr(self, 'scene_img') else None,
            'scene_mask': self.scene_mask if hasattr(self, 'scene_mask') else None,
            #'verts_filtered': self.verts_filtered.cpu().detach().numpy() if self.verts_filtered is not None else None,
        }


    def get_filtered_vertices_by_smpl(self, min_cutoff_T=0.004, min_cutoff_angles=0.1, beta_T=0.7, beta_angles=0.1, frame_rate=25):
        poses_T = self.poses_T.cpu().detach().numpy()
        smpl_pose = self.poses_smpl.cpu().detach().numpy()

        oef_T = OneEuroFilter(0, poses_T[0], dx0=0, min_cutoff=min_cutoff_T, beta=beta_T, d_cutoff=1.0)
        oef_pose = OneEuroFilter(0, smpl_pose[0], dx0=0, min_cutoff=min_cutoff_angles, beta=beta_angles, d_cutoff=1.0)

        for i in range(1, len(smpl_pose)):
            poses_T[i] = oef_T(i / frame_rate, poses_T[i])
            smpl_pose[i] = oef_pose(i / frame_rate, smpl_pose[i])

        poses_T = torch.tensor(poses_T, device=self.device)
        smpl_pose = torch.tensor(smpl_pose, device=self.device)

        results = self.SMPLPY(
            betas=self.betas_smpl.tile((self.num_frames, 1, 1)).view(-1, 10),
            poses=smpl_pose.view(-1, 72)
        )
        verts = results['verts'].view(self.num_frames, self.num_people, -1, 3)
        scale_factor = torch.pow(1.1, self.xscale_factor)
        verts_abs_filtered = scale_factor * verts + poses_T

        return verts_abs_filtered


    def one_euro_filter(self, x, min_cutoff=0.1, beta=0.02, frame_rate=25):
        y = x.cpu().detach().numpy()
        time_i = np.zeros_like(y[0])
        oef = OneEuroFilter(time_i, y[0], min_cutoff=min_cutoff, beta=beta)

        T = len(y)
        for i in range(1, T):
            time_i = time_i + (i / frame_rate)
            y[i] = oef(time_i, y[i].copy())
        y = torch.tensor(y.astype(np.float32), device=self.device, requires_grad=False)

        return y


    def __eval_batch_optimized_variables(self, idxs):
        idx_var = {} # Indexed variables
        batch_size = idxs.shape[0]
        idx_var['scale_factor'] = torch.pow(1.1, self.xscale_factor)

        idx_var['min_z'] = softplus(self.zmin_lin[idxs])
        idx_var['max_z'] = (
            idx_var['min_z'].detach().clone()
            + self.min_delta_z
            + softplus(self.zmax_lin[idxs])
         ) # (batch, 1, 1)

        idx_var['poses_smpl'] = self.poses_smpl[idxs].view(-1, 72)
        idx_var['betas_smpl'] = self.betas_smpl.tile((batch_size, 1, 1)).view(-1, 10)
        idx_var['valid_smpl'] = self.valid_smpl[idxs]

        results = self.SMPLPY(betas=idx_var['betas_smpl'], poses=idx_var['poses_smpl'])
        verts = results['verts'].view(batch_size, self.num_people, -1, 3)
        joints_smpl24 = results[self.smpl_sparse_joints_key].view(batch_size, self.num_people, -1, 3)

        idx_var['poses_smpl'] = idx_var['poses_smpl'].view(batch_size, self.num_people, 72)
        idx_var['betas_smpl'] = idx_var['betas_smpl'].view(batch_size, self.num_people, 10)
        idx_var['poses_T'] = self.poses_T[idxs]

        idx_var['verts_smpl_abs'] = idx_var['scale_factor'] * verts + idx_var['poses_T'] # (batch, N, V, 3)
        idx_var['joints_smpl_abs'] = idx_var['scale_factor'] * joints_smpl24 + idx_var['poses_T'] # (batch, N, J, 3)

        idx_var['intrinsics'] = torch.tile(self.cam_intrinsics[idxs], (1, self.num_people, 1, 1)).view(-1, 3, 3)

        return idx_var


    def __init_global_poses(self, pose2d_24j, poses_smpl, betas_smpl, num_iter, joints_thr=0.15):
        """Initilize poses_T array and refine poses_smpl with an optimization
        based approach, where the goal is to minimize the projection loss
        between the 3D poses and given 2D reference projection in the image
        plane.

        # Arguments
            pose2d_24j: numpy array of 2D pose in the image plane with shape
                (T, N, 24, 3), including the vibility flag as in last dimension
            poses_smpl: initial SMPL pose parameters, with shape (T, N, 72)
            betas_smpl: initial SMPL beta parameters, with shape (T, N, 10)
            num_iter: integer

        # Returns
            Returns a tuple of optimized values (poses_T, poses_smpl) as numpy.
        """
        T, N = pose2d_24j.shape[0:2]
        assert N == self.num_people, (
            f'Unexpected numper of people ({N} instead of {self.num_people}) in `pose2d_24j`')
        poses_T = np.tile(np.array([[[[0, 0, 1]]]], dtype=np.float32), (T, N, 1, 1))
        poses_T = torch.tensor(poses_T, device=self.device, requires_grad=True)

        poses_smpl = torch.tensor(poses_smpl.astype(np.float32), device=self.device)
        betas_smpl = torch.tensor(betas_smpl.astype(np.float32), device=self.device)

        vis_pose2d_24j = torch.tensor((pose2d_24j[..., 2:] > joints_thr).astype(np.float32), device=self.device)
        pose2d_24j = torch.tensor(pose2d_24j[..., 0:2].astype(np.float32), device=self.device)

        optimizer = torch.optim.Adam([poses_T], lr=0.5, betas=(0.5, 0.5), eps=1e-6)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        lossfn_pose2d = torch.nn.MSELoss(reduction='mean')
        lossfn_speed = torch.nn.MSELoss(reduction='sum')
        optim_log = []

        # Optimization loop
        for i in range(num_iter):
            optimizer.zero_grad()

            results = self.SMPLPY(betas=betas_smpl.view(T * N, -1), poses=poses_smpl.view(T * N, -1))
            scale_factor = torch.pow(1.1, self.xscale_factor)
            global_pose3d = scale_factor * results[self.smpl_sparse_joints_key].view(T, N, -1, 3) + poses_T
            reshaped_cam = torch.tile(self.cam_intrinsics, (1, N, 1, 1)).view(T * N, 3, 3)
            proj_2d = camera_projection_torch(global_pose3d.view(T * N, -1, 3), reshaped_cam, Kd=self.cam_dist_coef)

            loss_2d = lossfn_pose2d(
                self.pose_weights * vis_pose2d_24j * proj_2d.view(T, N, -1, 2),
                self.pose_weights * vis_pose2d_24j * pose2d_24j)

            poses_T_speed = lossfn_speed(poses_T[1:], poses_T[:-1])
            loss = (self.proj2d_loss_coef * loss_2d
                    +self.reg_velocity_coef * poses_T_speed)
            optim_log.append({'loss_2d': loss_2d.cpu().detach().numpy(),})

            loss.backward()
            optimizer.step()
            scheduler.step()

        return {
            'optim_log': optim_log,
            'poses_T': poses_T.cpu().detach().numpy(),
        }
