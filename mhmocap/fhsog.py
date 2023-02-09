import numpy as np
from tqdm import tqdm
import math
from scipy.ndimage.morphology import distance_transform_edt

import torch
from pytorch3d.structures import Meshes

from .transforms import camera_inverse_projection


def over_composite_from_fog(fog_alpha, near_z=1, far_z=100):
    D, H, W = fog_alpha.shape
    dval = np.linspace(np.log(near_z), np.log(far_z), D, dtype=np.float32)[:, np.newaxis, np.newaxis]

    blend = [np.ones((H, W), np.float32)]
    for d in range(D - 1):
        cur_blend = (1 - fog_alpha[d]) * blend[-1]
        blend.append(cur_blend)
    blend = np.stack(blend, axis=0)
    logdepth = np.sum(dval * fog_alpha * blend, axis=0)
    depth = np.exp(logdepth)
    
    return depth


def compute_points_inside_mesh(verts, faces, points, max_num_points=2**16, eps=1e-2):
    """Compute the points that are inside a given mesh.
    
    # Arguments
        verts: array with shape (V, 3)
        faces: array with shape (F, 3)
        points: array with shape (N, 3)
        max_num_points: integer, this is the maximum number of points that are
            processed in a single call, this limits the maximum amount of
            memory that this function tries to allocate.

    # Returns
        A new array with shape (M,) containing the indices of the points that
        are inside the mesh.
    """
    max_num_points = int(max_num_points)
    points = points.astype(np.float32) # (N, 3)

    # compute the center of each triangle
    face_verts = verts[faces].astype(np.float32) # (F, 3, 3)
    centers = np.mean(face_verts, axis=1, keepdims=True) # (F, 1, 3)

    # check if the input needs to be splitted to avoid memory crash
    closest_faces_idx = []
    idx = 0
    while (idx < len(points)):
        # find the closest faces in the mesh for a subset of points
        if (len(points) - idx) > max_num_points:
            clst = np.argmin(
                np.sum(np.square(centers - points[np.newaxis, idx:idx + max_num_points]), axis=2), # (Ni,)
                axis=0)
        else:
            clst = np.argmin(
                np.sum(np.square(centers - points[np.newaxis, idx:]), axis=2), # (Ni,)
                axis=0)
        idx += len(clst)
        closest_faces_idx.append(clst)
    closest_faces_idx = np.concatenate(closest_faces_idx, axis=0) # (N,)
    
    face2point = points - centers[closest_faces_idx, 0]
    face2point /= np.clip(np.linalg.norm(face2point, axis=1, keepdims=True), eps, None) # (N, 3)

    closest_faces = face_verts[closest_faces_idx]
    normals = np.cross(
        closest_faces[:, 1] - closest_faces[:, 0],
        closest_faces[:, 2] - closest_faces[:, 1],
        axis=1)
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), eps, None) # (N, 3)
    dots = np.sum(face2point * normals, axis=1) # (N,)

    return np.argwhere(dots < -eps / 10)[:, 0]


def build_fhs_occupancy_grid(dataset, min_z, max_z, num_depth_bins=128):
    """This function build a Frustum Human-Scene Occupancy Grid from a set of
    depth maps predictions and SMPL models predictions (if given).
    
    # Arguments
        dataset: iterable instance of a class that has the following attributes:
            cam: dictionary with the key 'K', that gives the camera intrinsics
            SMPLPY: instance of the SMPL class
            image_size: tuple (W, H)
            In the itterations, dataset should provide the following data (as a dictionary):
                'images': array with shape (H, W, 3)
                'depths': array with shape (H, W)
                'backmasks': array with shape (H, W)
        min_z: array with scalars for de-normalizing disparity maps, with shape (T,), where `T = len(dataset)`
        max_z: array with shape (T,)
    """
    T = len(dataset)
    W, H = dataset.image_size
    D = num_depth_bins

    assert (len(min_z) == T) and (len(max_z) == T), (
        f'Invalid dataset length ({T}) and min_z/max_z lengths ({len(min_z)})/({len(max_z)})')
    
    near_z = 0.999 * np.median(min_z.squeeze()) #min_z.min()
    far_z = 1.001 * np.median(max_z.squeeze()) #max_z.max()

    fhsog_alpha = np.zeros((D + 1, H, W), np.uint)
    texture_map = np.zeros((3, H, W), np.uint)

    for t, data in enumerate(dataset):
        de = 1.0 / (data['depths'] * (1.0 / min_z[t] - 1.0 / max_z[t]) + 1.0 / max_z[t]) # (H, W)
        mask = (data['backmasks'] > 0.5) * ((de >= near_z) * (de <= far_z)).astype(np.uint) # (H, W)

        vlog_bins = (np.log(np.clip(de, near_z, far_z)) - np.log(near_z)) / (np.log(far_z) - np.log(near_z))
        idx_bins = (mask * (1 + D * vlog_bins)).astype(np.uint)[np.newaxis] # (1, H, W)

        values = np.take_along_axis(fhsog_alpha, idx_bins, axis=0)
        np.put_along_axis(fhsog_alpha, idx_bins, values + 1, axis=0)

        img = np.transpose(data['images'], (2, 0, 1)).astype(np.uint) # (3, H, W)
        texture_map += mask[np.newaxis] * img

    mask_acc = T - fhsog_alpha[0] # This is equivalent to a mask accumulator
    texture_map = (texture_map / np.clip(mask_acc, 1, None)).astype(np.uint8) # TODO: for now, this is a simple masked average
    back_mask = (mask_acc > 0).astype(np.uint)
    texture_map += 255 * (1 - back_mask[np.newaxis])
    
    fhsog_alpha = fhsog_alpha[1:]
    amax = np.argmax(fhsog_alpha, axis=0) # (H, W)
    for d in range(D):
        fhsog_alpha[d] = back_mask * (amax <= d).astype(np.uint)
        if (fhsog_alpha[d].sum() / back_mask.sum()) > 0.95:
            fhsog_alpha[d:] = 1
            break

    return fhsog_alpha, back_mask, texture_map, near_z, far_z


def carve_fog_with_meshes(fog, verts, faces, near_z, far_z, cam_k, carving_thr=0, carving_value=0, verbose=False):
    D, H, W = fog.shape

    # compute the frustum coordinates (in camera space) of all voxels in our grid
    uu = np.linspace(0.5, W - 0.5, W)
    vv = np.linspace(0.5, H - 0.5, H)
    dd = np.exp(np.linspace(np.log(near_z), np.log(far_z), D))
    frust_grid = np.stack(np.meshgrid(uu, vv, dd, indexing='ij'), axis=-1).reshape((-1, 3))

    solid_mask = fog > 0
    fhsog_solid_idx = np.argwhere(solid_mask.T.reshape((-1,)))[:, 0]
    solid_3d_grid = camera_inverse_projection(frust_grid[fhsog_solid_idx], cam_k) # (num_solid_pts, )
    human_grid_counter = np.zeros_like(fhsog_solid_idx)

    vT, N = verts.shape[0:2]
    verts = verts.reshape((vT * N, -1, 3)) # (vT * N, V, 3)
    if verbose:
        iter_fn = tqdm(verts)
    else:
        iter_fn = verts

    for vt in iter_fn:
        min_xyz = np.min(vt, axis=0, keepdims=True) - 1e-3
        max_xyz = np.max(vt, axis=0, keepdims=True) + 1e-3
        bbox_pts_idx = np.argwhere(
            (solid_3d_grid > min_xyz).all(axis=1)
            * (solid_3d_grid < max_xyz).all(axis=1))[:, 0]
        if len(bbox_pts_idx) > 0:
            pts_inside_mesh_idx = compute_points_inside_mesh(vt, faces, solid_3d_grid[bbox_pts_idx])
            if len(pts_inside_mesh_idx) > 0:
                human_grid_counter[bbox_pts_idx[pts_inside_mesh_idx]] += 1

    #print ('human_grid_counter', human_grid_counter.min(), human_grid_counter.max())

    solid_carvind_idx = np.argwhere(human_grid_counter > carving_thr)[:, 0]
    carving_idx = fhsog_solid_idx[solid_carvind_idx]
    #if len(carving_idx) > 0:
    #    fog.T.reshape((-1,))[carving_idx] = carving_value

    return carving_idx


def aggegrate_scene_geometry_median(depths, images, backmasks, depth_metric='median'):
    """Aggregates multiple depth maps, images, and segmentations masks to
    build a static RGB-D image of the scene.

    # Arguments
        depths: shape (T, H, W)
        images: shape (T, H, W, 3)
        backmasks: shape (T, H, W)
    """
    if images is not None:
        ma_bkg_img = np.ma.array(images, mask=np.tile(backmasks[..., np.newaxis] == 0, (1, 1, 1, 3)))
        ma_bkg_img = np.ma.median(ma_bkg_img, axis=0)
        bkg_img = ma_bkg_img.data.astype(np.uint8)
    else:
        bkg_img = None

    metric_fn = getattr(np.ma, depth_metric)
    ma_depth = np.ma.array(depths, mask=backmasks == 0)
    ma_bkg_depth = metric_fn(ma_depth, axis=0)
    bkg_depth = ma_bkg_depth.data.astype(np.float32)
    mask = ma_bkg_depth.mask == 0

    return bkg_img, bkg_depth, mask


def compute_gaussian_distance_field_1d(x, sampling=None, sigma=1.0):
    N = x.shape[0]
    if sampling is None:
        sampling = (1.0 / N,)
    edt = distance_transform_edt(x, sampling=sampling)
    gdf = (1.0 / (sigma * np.sqrt(2 * math.pi))) * np.exp(-np.square(edt) / (np.square(sigma)))

    return gdf.astype(np.float32)


def compute_gaussian_distance_field_2d(omap, sampling=None, sqclip=0.01, sigma=1.0):
    """Compute a Gaussian Distance Field from a set of binary occupancy maps.
    
    # Arguments
        omap: shape (H, W)

    # Returns
        A new tensor with shape (T, H, W) where each time t contais a 2D GDF map.
    """
    H, W = omap.shape
    if sampling is None:
        sampling = (1.0 / H, 1.0 / W)

    edt = distance_transform_edt(omap, sampling=sampling)
    gdf = (1 / (sigma * np.sqrt(2 * math.pi))) * np.exp(-np.square(edt) / (np.square(sigma)))

    return gdf.astype(np.float32) * (edt > 0).astype(np.float32)


def build_fhsog_from_smpl_2(depths, images, backmasks, verts, faces,
        rasterizer, near_z, far_z, num_depth_bins, device,
        sqclip=0.01, sigma=0.25):

    T, N = verts.shape[0:2]
    D = num_depth_bins

    faces = np.tile(faces[np.newaxis].astype(np.int32), (N, 1, 1))
    backseg_mask = (backmasks > 0.5) * ((depths > near_z) * (depths < far_z)).astype(np.uint) # (T, H, W)

    H, W = images.shape[1:3]
    fhsog_alpha = np.zeros((D, H, W), np.float32)
    texture_map = np.zeros((H, W, 3), np.float32)
    
    img_smpl_mask_all = []
    img_gdf_all = []

    for t in tqdm(range(T)):
        meshes = Meshes(
            torch.tensor(verts[t], device=device),
            torch.tensor(faces, device=device))
        fragments = rasterizer(meshes)
        zbuf = fragments.zbuf[..., 0].cpu().numpy()
        H, W = zbuf.shape[1:]
        
        zbuf_mask = zbuf > 0
        img_smpl_mask = np.max(zbuf_mask, axis=0)
        img_smpl_mask_all.append(img_smpl_mask)

        avg_person_depth = np.sum(zbuf_mask * zbuf, axis=(1, 2)) / np.clip(zbuf_mask.sum(axis=(1, 2)), 1, None)
        avg_person_depth = np.clip(avg_person_depth, near_z, far_z)
        img_gdf = compute_gaussian_distance_field_2d(backseg_mask[t] * (1 - img_smpl_mask), sqclip=sqclip, sigma=sigma)
        img_gdf_all.append(img_gdf)
        
        vlog_person_depth_bins = (np.log(np.clip(avg_person_depth, near_z, far_z)) - np.log(near_z)) / (np.log(far_z) - np.log(near_z)) # (N,)
        idx_person_depth_bins = (D * vlog_person_depth_bins - 0.5).astype(np.uint) # (N,)
    
        vlog_bins = (np.log(np.clip(depths[t], near_z, far_z)) - np.log(near_z)) / (np.log(far_z) - np.log(near_z)) # (H, W)
        idx_bins = (D * vlog_bins - 0.5).astype(np.uint)[np.newaxis] # (1, H, W)
        
        person_depth_vec = np.ones((D, 1, 1))
        person_depth_vec[idx_person_depth_bins] = 0
        person_depth_gdf = compute_gaussian_distance_field_1d(person_depth_vec.squeeze(), sigma=5.0)
        person_depth_gdf = person_depth_gdf[..., np.newaxis, np.newaxis] # (D, 1, 1)
        #print (person_depth_vec.squeeze())
        #print (person_depth_gdf)

        values = np.take_along_axis(fhsog_alpha, idx_bins, axis=0)
        
        gval = img_gdf * person_depth_gdf # (D, H, W)
        gval = np.take_along_axis(gval, idx_bins, axis=0) # (1, H, W)

        texture_map += gval.squeeze()[..., np.newaxis] * images[t]
        
        new_values = values + gval
        np.put_along_axis(fhsog_alpha, idx_bins, new_values, axis=0)

    outmask = np.sum(fhsog_alpha, axis=0)
    texture_map = texture_map / np.clip(outmask[..., np.newaxis], 0.1, None)
    texture_map = np.clip(texture_map, 0, 255).astype(images.dtype)
    #outmask = outmask > 0.0 * np.log(T) #(sigma * T / 2)

    return fhsog_alpha, texture_map, outmask, np.stack(img_smpl_mask_all, axis=0), np.stack(img_gdf_all, axis=0)