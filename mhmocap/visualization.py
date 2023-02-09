import os
import numpy as np
import open3d as o3d
import copy
from PIL import Image
from pathlib import Path

default_vis_color_list = [
    np.array([[240, 112, 96]], np.float32) / 255.,
    np.array([[96, 240, 112]], np.float32) / 255.,
    np.array([[96, 112, 240]], np.float32) / 255.,
    np.array([[196, 96, 192]], np.float32) / 255.,
    np.array([[232, 224, 44]], np.float32) / 255.,
    np.array([[44, 232, 192]], np.float32) / 255.,
    np.array([[32, 220, 255]], np.float32) / 255.,
    np.array([[64, 128, 128]], np.float32) / 255.,
    np.array([[220, 190, 190]], np.float32) / 255.,
    np.array([[190, 244, 150]], np.float32) / 255.,
    np.array([[180, 32, 32]], np.float32) / 255.,
    np.array([[32, 120, 150]], np.float32) / 255.,
    np.array([[96, 192, 244]], np.float32) / 255.,
    np.array([[220, 108, 64]], np.float32) / 255.,
    np.array([[128, 208, 64]], np.float32) / 255.,
]

class BaseVisualizer(object):
    def __init__(self, camera, renderoption_filename="../renderoption.json"):
        self.renderoption_filename = renderoption_filename
        self.window_size = camera['image_size']
        self.K = camera['K']


def load_render_option_callback(renderoption_filename):
    def load_render_option(vis):
        print (f'Loading render options from ', renderoption_filename)
        vis.get_render_option().load_from_json(renderoption_filename)
        return False
    return load_render_option

def update_camera_callback(window_size, K):
    def update_camera(vis):
        ctr = vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        win_w = init_param.intrinsic.width
        win_h = init_param.intrinsic.height
        cam_w = window_size[0]
        cam_h = window_size[1]
        init_param.intrinsic.set_intrinsics(
                win_w, win_h,
                win_w * K[0, 0] / cam_w, win_h * K[1, 1] / cam_h,
                win_w / 2 - 0.5, win_h / 2 - 0.5,
        )
        init_param.extrinsic = np.array([
                [1, 0, 0, 0],
                [0,-1, 0, 0],
                [0, 0,-1, 0],
                [0, 0, 0, 1],
            ], dtype=init_param.extrinsic.dtype)
        ctr.convert_from_pinhole_camera_parameters(init_param)
        return False
    return update_camera

class SceneHumansVisualizer(BaseVisualizer):
    """This class implements a visualizer for multiple frames point cloud
    scene with multiple humans. The implementation is based on Open3D.

    # Arguments
        images: numpy array uint8 with shape (N, H, W, 3), for one scene per
            N frames, or (H, W, 3) for a unique scene
        depths: numpy array float32 with shape (N, H, W), for one scene per
            frame, or (H, W) for a unique scene
        camera: dictionary with the following info:
            'image_size': tuple (H, W)
            'K': camera intrinsics as a 3x3 matrix
        vertices: vertices of each person in each frame. Can be an array
            with shape (T, N, V, 3), or a list (len = T) of lists (len <= N)
            of arrays with shape (V^t_n, 3).
        valid_vertices: flag for valid vertices, as array (T, N) or as a list
            (len = T) of lists (len <= N) with booleans.
            If None, all vertices are shown.
        output_path: string with path to save recorded frames.
        capture_rendered_color: boolean, if True, save a PNG color frame after each event.
        capture_rendered_depth: boolean, if True, save a PNG depth frame after each event.
    """
    def __init__(self, images, depths, camera,
            vertices=None,
            faces=None,
            valid_vertices=None,
            # pose3d=None, # (T, N, J, 3)
            output_path=None,
            capture_rendered_color=False,
            capture_rendered_depth=False,
            show_coordinate_axis=True,
            # radius=1.0,
            # resolution=5,
            vis_color_list=default_vis_color_list,
            y1=-1, y2=-1, x1=-1, x2=1, z1=0, z2=4,
            show_floor=False,
            floor_color=[0.2, 0.2, 0.2],
            verbose=True,
            **kargs):
        super().__init__(camera, **kargs)

        self.verbose = verbose
        if (images.ndim == 4) and (depths.ndim == 3):
            self.pcd_scene_list = build_point_clouds_from_rgbd_frames(camera, images, depths)
            self.multiple_scenes = True
        elif (images.ndim == 3) and (depths.ndim == 2):
            self.pcd_scene_list = [
                build_single_point_cloud_from_rgbd(camera, images, depths)
            ]
            self.multiple_scenes = False
        else:
            raise ValueError(f'Invalid images and depths shape: {images.shape}, {depths.shape}')

        self.vertices = vertices
        self.faces = faces
        if self.vertices is not None:
            if self.faces is not None:
                self.show_meshes = True
                self.mesh_list = build_meshes(self.vertices[0], self.faces, vis_color_list=vis_color_list)
                self.valid_vertices = valid_vertices
            else:
                self.show_meshes = False
                self.pcd_obj_list = build_point_clouds_from_vertices(vertices,
                        valid_vertices=valid_vertices, vis_color_list=vis_color_list)

        # self.pose3d = pose3d
        # if self.pose3d is not None:
        #     num_joints = pose3d.shape[1]
        #     self.mesh_pose = build_meshes_from_pose(num_joints, vis_color_list=default_vis_color_list, radius=radius, resolution=resolution)
        
        if vertices is not None:
            self.max_frames = len(vertices)
        else:
            self.max_frames = len(self.pcd_scene_list)

        if output_path is None:
            self.output_path = os.path.realpath('./')
        else:
            self.output_path = os.path.realpath(output_path)
            Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.capture_rendered_color = capture_rendered_color
        self.capture_rendered_depth = capture_rendered_depth
        self.show_coordinate_axis = show_coordinate_axis

        if show_floor:
            vertices = o3d.utility.Vector3dVector(np.array(
                [
                    [x1, y1, z1],
                    [x1, y2, z2],
                    [x2, y2, z2],
                    [x2, y1, z1]
                ]))
            triangles = o3d.utility.Vector3iVector(np.array(
                [
                    [0, 3, 1],
                    [1, 3, 2],
                ]))
            self.mesh_floor = o3d.geometry.TriangleMesh(vertices, triangles)
            #self.mesh_floor.compute_vertex_normals()
            self.mesh_floor.paint_uniform_color(floor_color)
            self.mesh_floor.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        else:
            self.mesh_floor = None


    def run(self):
        self.pcd_base = copy.deepcopy(self.pcd_scene_list[0])
        if self.vertices is not None:
            if self.show_meshes:
                self.objs = copy.deepcopy(self.mesh_list)
            else:
                self.objs = copy.deepcopy(self.pcd_obj_list[0])
        self.curr_frame = 0
        self.curr_image_index = 0
        self.vis = o3d.visualization.Visualizer()

        def capture_frame(vis):
            try:
                if self.capture_rendered_color:
                    image = vis.capture_screen_float_buffer()
                    image = np.array(image)
                    image_pil = Image.fromarray((255 * image).astype(np.uint8))

                    fname = os.path.join(self.output_path, f'img_{self.curr_image_index:06d}.png')
                    image_pil.save(fname)
                
                if self.capture_rendered_depth:
                    depth = vis.capture_depth_float_buffer()
                    depth = (1000 * np.array(depth))
                    depth_pil = Image.fromarray(depth.astype(np.uint16))

                    fname = os.path.join(self.output_path, f'depth_{self.curr_image_index:06d}.png')
                    depth_pil.save(fname)

                self.curr_image_index += 1
                return False

            except Exception as e:
                print ('Error: SceneHumansVisualizer::capture_frame ' + str(e))
                return True

        def rotate_and_save(angles):
            def _custum_rotate_and_save(vis):
                ctr = vis.get_view_control()
                ctr.rotate(angles[0], angles[1])
                # t = self.curr_frame + 1
                # if t >= self.max_frames:
                #     t = 0
                # self.curr_frame = t
                
                #capture_frame(vis)

                return False
            return _custum_rotate_and_save

        def camera_translate(trans):
            def _custum_camera_translate(vis):
                ctr = vis.get_view_control()
                ctr.camera_local_translate(
                    forward=trans[0],
                    right=trans[1],
                    up=trans[2],
                )
                
                #capture_frame(vis)

                return False
            return _custum_camera_translate

        def show_next_frame(vis):
            try:
                t = self.curr_frame + 1
                if t >= self.max_frames:
                    t = 0
                self.curr_frame = t

                if self.verbose:
                    print (f'frame {t}')

                if self.multiple_scenes:
                    self.pcd_base.points = self.pcd_scene_list[t].points
                    self.pcd_base.colors = self.pcd_scene_list[t].colors
                    vis.update_geometry(self.pcd_base)

                if self.vertices is not None:
                    if self.show_meshes:
                        for n in range(len(self.objs)):
                            if self.valid_vertices[t, n, 0]:
                                self.objs[n].vertices = o3d.utility.Vector3dVector(self.vertices[t][n])
                            else:
                                dummy_v = np.tile(np.array([[0, 0, -1]]), (len(self.vertices[t][n]), 1))
                                self.objs[n].vertices = o3d.utility.Vector3dVector(dummy_v)
                            self.objs[n].compute_vertex_normals()
                            self.objs[n].transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                            vis.update_geometry(self.objs[n])
                    else:
                        self.objs.points = self.pcd_obj_list[t].points
                        self.objs.colors = self.pcd_obj_list[t].colors               
                        vis.update_geometry(self.objs)

                capture_frame(vis)

                return False

            except Exception as e:
                print ('Error: SceneHumansVisualizer::show_next_frame ' + str(e))
                return True

        key_to_callback = {}
        key_to_callback[ord("C")] = capture_frame
        key_to_callback[ord("J")] = rotate_and_save((1, 0))
        key_to_callback[ord("L")] = rotate_and_save((-1, 0))
        key_to_callback[ord("I")] = rotate_and_save((0, 1))
        key_to_callback[ord("K")] = rotate_and_save((0, -1))
        key_to_callback[ord("W")] = camera_translate((0.01, 0, 0))
        key_to_callback[ord("S")] = camera_translate((-0.01, 0, 0))
        key_to_callback[ord("A")] = camera_translate((0.0, 0.01, 0))
        key_to_callback[ord("D")] = camera_translate((0.0, -0.01, 0))
        key_to_callback[ord("X")] = camera_translate((0.0, 0.0, 0.01))
        key_to_callback[ord("C")] = camera_translate((0.0, 0.0, -0.01))
        key_to_callback[ord("N")] = show_next_frame
        key_to_callback[ord("U")] = update_camera_callback(self.window_size, self.K)
        key_to_callback[ord("R")] = load_render_option_callback(self.renderoption_filename)

        obj_list = [self.pcd_base]
        if self.vertices is not None:
            if self.show_meshes:
                obj_list += self.objs
            else:
                obj_list.append(self.objs)

        # Draw open3d Coordinate system
        if self.show_coordinate_axis:
            axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            axis_mesh = axis_mesh.translate((0.0, 0.0, 0.0))
            axis_mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            obj_list.append(axis_mesh)

        if self.mesh_floor is not None:
            obj_list.append(self.mesh_floor)

        o3d.visualization.draw_geometries_with_key_callbacks(obj_list, key_to_callback)


def build_3d_scene_and_humans(camera, image=None, depth=None, verts_list=None,
        vis_color_list=default_vis_color_list):
    """Build a 3D point cloud with Open3D of a given scene geometry, given as
    an RGB-D mapping, and the vertices / keypoints in 3D from a set of objects
    in the scene.

    # Arguments
        camera: dictionary with the following info:
            'image_size': tuple (H, W)
            'K': camera intrinsics as a 3x3 matrix
        image: numpy array uint8 with shape (H, W, 3)
        depth: numpy array float32 with shape (H, W)
        verts_list: list of numpy arrays of 3D points, each with shape (Q, 3),
            where Q can change from one instance to another.
    """
    o3d_cam = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters(0))
    o3d_cam.width = camera['image_size'][0]
    o3d_cam.height = camera['image_size'][1]
    o3d_cam.intrinsic_matrix = camera['K']
    pcd_list = []

    if (image is not None) and (depth is not None):
        o3d_color_image = o3d.geometry.Image(image)
        o3d_depth_image = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color_image, o3d_depth_image,
            depth_scale=1.0, depth_trunc=depth.max(),
            convert_rgb_to_intensity=False)
        pcd_scene = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_cam)
        pcd_scene.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd_list.append(pcd_scene)

    if verts_list is not None:
        all_objs = []
        all_colors = []
        for i, obj in enumerate(verts_list):
            obj_color = np.ones_like(obj) * vis_color_list[i % (len(vis_color_list))]
            all_objs.append(obj)
            all_colors.append(obj_color)

        all_objs = np.concatenate(all_objs, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        pcd_objs = o3d.geometry.PointCloud()
        pcd_objs.points = o3d.utility.Vector3dVector(all_objs)
        pcd_objs.colors = o3d.utility.Vector3dVector(all_colors)
        pcd_objs.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd_list.append(pcd_objs)

    return pcd_list


def draw_geometry_with_key_callback(pcd_list):
    """ Draw a 3D geometry with Open3D with customized callbacks.

    # Arguments
        pcd_list: list of o3d.geometry.PointCloud objects to be rendered.
    """
    print (
        '-- customized draw geometry: --\n'
        '    `k`: change background to black\n'
        '    `w`: change background to white\n'
        '    `i`: capture image buffer\n'
        '    `k`: capture depth buffer\n'
    )
    draw_geometry_with_key_callback.index = -1
    if not os.path.exists("./frames/"):
        os.makedirs("./frames/")

    def build_change_background_to_color(color):
        def _change_bkg(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray(color)
            return False
        return _change_bkg

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "/CT/depth-3dhp/work/git/depth-multiperson/renderoption.json")
        return False

    def capture_image(vis, idx=None):
        if idx is None:
            draw_geometry_with_key_callback.index += 1
            idx = draw_geometry_with_key_callback.index
        image = vis.capture_screen_float_buffer()
        image = np.array(image)
        image_pil = Image.fromarray((255 * image).astype(np.uint8))
        image_pil.save(f'./frames/img_{idx:06d}.png')
        return False

    def capture_depth(vis, idx=None):
        depth = vis.capture_depth_float_buffer()
        depth = (1000 * np.array(depth))
        depth_pil = Image.fromarray(depth.astype(np.uint16))
        depth_pil.save(f'./frames/depth_{idx:06d}.png')
        return False
    
    def build_rotate_and_save(x, y):
        def _rotate_and_save(vis):
            ctr = vis.get_view_control()
            ctr.rotate(x, y)
            draw_geometry_with_key_callback.index += 1
            capture_image(vis, draw_geometry_with_key_callback.index)
            #capture_depth(vis, draw_geometry_with_key_callback.index)
            return False
        return _rotate_and_save

    key_to_callback = {}
    key_to_callback[ord("K")] = build_change_background_to_color([0, 0, 0])
    key_to_callback[ord("W")] = build_change_background_to_color([255, 255, 255])
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_image
    key_to_callback[ord(".")] = capture_depth
    key_to_callback[ord('C')] = build_rotate_and_save(10, 0)
    key_to_callback[ord('Z')] = build_rotate_and_save(-10, 0)
    key_to_callback[ord('S')] = build_rotate_and_save(0, 10)
    key_to_callback[ord('X')] = build_rotate_and_save(0, -10)
    o3d.visualization.draw_geometries_with_key_callbacks(pcd_list, key_to_callback)


def custom_draw_geometry_with_camera_trajectory(pcd_list, camera_trajectory, output_path,
        save_depth=False, window_size=(1920, 1080)):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()

    Path(output_path).mkdir(parents=True, exist_ok=True)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        try:
            ctr = vis.get_view_control()
            glb = custom_draw_geometry_with_camera_trajectory
            if glb.index >= 0:
                
                image = np.array(vis.capture_screen_float_buffer(False))
                image_pil = Image.fromarray((255 * image).astype(np.uint8))
                image_pil.save(os.path.join(output_path, f'img_{glb.index:06d}.png'))

                if save_depth:
                    depth = np.array(vis.capture_depth_float_buffer(False))
                    depth_pil = Image.fromarray((1000 * depth).astype(np.uint16))
                    depth_pil.save(os.path.join(output_path, f'depth_{glb.index:06d}.png'))

            glb.index = glb.index + 1
            if glb.index < len(camera_trajectory):
                x, y = camera_trajectory[glb.index]
                ctr.rotate(x, y)
            else:
                custom_draw_geometry_with_camera_trajectory.vis.register_animation_callback(None)
            return False

        except:
            return True

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window(width=window_size[0], height=window_size[1], left=0, top=0)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    #vis.get_render_option().load_from_json("/CT/depth-3dhp/work/git/Open3D/examples/test_data/renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


def build_single_point_cloud_from_rgbd(camera, image, depth):
    o3d_cam = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters(0)
    )
    o3d_cam.width = camera['image_size'][0]
    o3d_cam.height = camera['image_size'][1]
    o3d_cam.intrinsic_matrix = camera['K']

    o3d_color_image = o3d.geometry.Image(image)
    o3d_depth_image = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color_image, o3d_depth_image,
        depth_scale=1.0, depth_trunc=depth.max() + 1,
        convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_cam)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


def build_point_clouds_from_rgbd_frames(camera, images, depths):
    """Builds a list of 3D point cloud with Open3D of a given sequence of
    scene geometry, given as an RGB-D mappings.

    # Arguments
        camera: dictionary with the following info:
            'image_size': tuple (H, W)
            'K': camera intrinsics as a 3x3 matrix
        image: numpy array uint8 with shape (T, H, W, 3),
            where T if the number of frames
        depth: numpy array float32 with shape (T, H, W)

    # Returns
        A list of point clouds, one per frame T.
    """
    assert (images.shape[0:3] == depths.shape), (
            f'Error: invalid image / depth buffers {images.shape} / {depths.shape}'
    )
    assert ((images.shape[2] == camera['image_size'][0])
            and (images.shape[1] == camera['image_size'][1])), (
        f'Error: invalid image / camera info {images.shape} / {camera}'
    )

    pcd_list = []
    for i in range(len(images)):
        pcd = build_single_point_cloud_from_rgbd(camera, images[i], depths[i])
        pcd_list.append(pcd)

    return pcd_list



def build_point_clouds_from_vertices(verts, valid_vertices=None, vis_color_list=default_vis_color_list):
    """Builds a list of 3D point cloud with Open3D of a given sequence of
    vertices.

    # Arguments
        verts: array of vertices with shape (T, N, V, 3), where T is the
            number of frames, N is the number of objects, and V is the
            number of vertices.
        valid_vertices: flag for valid vertices, as array (T, N) or as a list
            (len = T) of lists (len <= N) with booleans.
            If None, all vertices are shown.

    # Returns
        A list of point clouds, one per frame T.
    """
    pcd_list = []
    for f in range(len(verts)):
        all_objs = []
        all_colors = []
        for i, obj in enumerate(verts[f]):
            if (valid_vertices is None) or ((valid_vertices is not None) and valid_vertices[f][i]):
                obj_color = np.ones_like(obj) * vis_color_list[i % (len(vis_color_list))]
                all_objs.append(obj)
                all_colors.append(obj_color)

        if len(all_objs) > 0:
            all_objs = np.concatenate(all_objs, axis=0)
            all_colors = np.concatenate(all_colors, axis=0)

            pcd_objs = o3d.geometry.PointCloud()
            pcd_objs.points = o3d.utility.Vector3dVector(all_objs)
            pcd_objs.colors = o3d.utility.Vector3dVector(all_colors)
            pcd_objs.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pcd_list.append(pcd_objs)

    return pcd_list

def build_meshes(verts, faces, vis_color_list=default_vis_color_list):
    """Builds a list of 3D meshes Open3D of a given sequence of vertices.

    # Arguments
        verts: array of vertices with shape (N, V, 3), where N is the number
            of objects with V vertices.
        faces: array of faces with shape (F, 3), assumed to be the same for all objects

    # Returns
        A list of meshes, one per frame T.
    """
    meshes = []
    for n in range(len(verts)):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts[n])
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(vis_color_list[n][0])
        mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        meshes.append(mesh)

    return meshes



def build_meshes_from_pose(num_joints, vis_color_list=default_vis_color_list, radius=1.0, resolution=5):
    mesh_list = []
    for j in range(num_joints):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        mesh_sphere.paint_uniform_color(vis_color_list[j][0])
        mesh_list.append(mesh_sphere)

    return mesh_list


def make_pose_mesh_mupots(pose3d, vis, radius=0.02, resolution=5, color=[0.5, 0.5, 0.5], thr=0.5):
    joint_links = [
        1, #0
        2, #1
        3, #2
        4, #3
        -1, #4
        1, #5
        5, #6
        6, #7
        14, #8
        8, #9
        9, #10
        14, #11
        11, #12
        12, #13
        1, #14
    ]
    points = np.zeros_like(pose3d[0:15])
    point_links = []
    line_colors = [color for i in range(len(points))]

    objts = []
    for i in range(len(points)):
        if vis[i] > thr:
            points[i] = pose3d[i]
            used_radius = radius
        else:
            points[i] = np.zeros_like(pose3d[i])
            used_radius = 1e-6
        
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=used_radius, resolution=resolution)
        mesh_sphere = mesh_sphere.translate(points[i])
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(color)
        mesh_sphere.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        objts.append(mesh_sphere)

        if (vis[i] > 0.5) and (joint_links[i] >= 0) and (vis[joint_links[i]] > 0.5):
            point_links.append([i, joint_links[i]])
        else:
            point_links.append([0, 0])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(point_links)
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    line_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return objts, line_set


class SkeletonVisualizer(BaseVisualizer):
    def __init__(self, camera, pred3d, pred_vis,
            pose_gt=None,
            gt_vis=None,
            match_list=None,
            output_path=None,
            capture_rendered_color=False,
            show_coordinate_axis=False,
            backimages=None,
            backdist=6,
            vis_color_list=default_vis_color_list,
            y1=-1, y2=-1, x1=-1, x2=1, z1=0, z2=4,
            show_floor=True,
            floor_color=[0.2, 0.2, 0.2],
            radius=0.03,
            **kargs):
        super().__init__(camera, **kargs)

        assert ((pred3d.shape[0:2] == pred_vis.shape[0:2])
                and (pred3d.shape[2:] == (15, 3))
                and (pred_vis.shape[2:] == (15, 1))
        )
        if (pose_gt is not None) and (gt_vis is not None):
            assert ((pose_gt.shape[0:2] == gt_vis.shape[0:2])
                    and (pose_gt.shape[2:] == (15, 3))
                    and (gt_vis.shape[2:] == (15, 1))
                    and (match_list is not None)
            )

        self.max_frames = len(pred3d)

        self.sphere_list = []
        self.lines_list = []
        self.pcd_scene_list = []
        if backimages is not None:
            if backimages.ndim == 4:
                T, H, W = backimages.shape[0:3]
                self.pcd_scene_list = build_point_clouds_from_rgbd_frames(camera, backimages, backdist * np.ones((T, H, W), np.float32))
                self.multiple_scenes = True
            elif backimages.ndim == 3:
                H, W = backimages.shape[0:2]
                self.pcd_scene_list = [
                    build_single_point_cloud_from_rgbd(camera, backimages, backdist * np.ones((T, H, W), np.float32))
                ]
                self.multiple_scenes = False
            else:
                raise ValueError(f'Invalid backimages shape: {backimages.shape}')

        for t in range(self.max_frames):
            self.sphere_list.append([])
            self.lines_list.append([])

            if (pose_gt is not None) and (gt_vis is not None):
                pref_idx, pred_idx = match_list[t]
                for n, (p1, v1, p2, v2) in enumerate(
                    zip(pred3d[t, pred_idx, :15],
                        pred_vis[t, pred_idx, :15],
                        pose_gt[t, pref_idx, :15],
                        gt_vis[t, pref_idx, :15],
                        )):

                    obj, lines = make_pose_mesh_mupots(p1, v1, radius=radius, color=vis_color_list[n][0], thr=0.5)
                    self.sphere_list[t] += obj
                    self.lines_list[t].append(lines)

                    obj, lines = make_pose_mesh_mupots(p2, v2, radius=radius, color=[0.25, 0.25, 0.25], thr=0.1)
                    self.sphere_list[t] += obj
                    self.lines_list[t].append(lines)

            else:
                for n, (p1, v1) in enumerate(zip(pred3d[t], pred_vis[t])):
                    obj, lines = make_pose_mesh_mupots(p1, v1, radius=radius, color=vis_color_list[n][0], thr=0.5)
                    self.sphere_list[t] += obj
                    self.lines_list[t].append(lines)

        if show_floor:
            vertices = o3d.utility.Vector3dVector(np.array(
                [
                    [x1, y1, z1],
                    [x1, y2, z2],
                    [x2, y2, z2],
                    [x2, y1, z1]
                ]))
            triangles = o3d.utility.Vector3iVector(np.array(
                [
                    [0, 3, 1],
                    [1, 3, 2],
                ]))
            self.mesh_floor = o3d.geometry.TriangleMesh(vertices, triangles)
            #self.mesh_floor.compute_vertex_normals()
            self.mesh_floor.paint_uniform_color(floor_color)
            self.mesh_floor.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        else:
            self.mesh_floor = None
        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2
        self.z1 = z1
        self.z2 = z2
        self.output_path = output_path

        self.capture_rendered_color = capture_rendered_color
        self.show_coordinate_axis = show_coordinate_axis


    def run(self):
        if len(self.pcd_scene_list) > 0:
            self.pcd_base = copy.deepcopy(self.pcd_scene_list[0])

        self.curr_frame = 0
        self.curr_image_index = 0
        self.vis = o3d.visualization.Visualizer()

        def capture_frame(vis):
            try:
                if self.capture_rendered_color:
                    image = vis.capture_screen_float_buffer()
                    image = np.array(image)
                    image_pil = Image.fromarray((255 * image).astype(np.uint8))

                    fname = os.path.join(self.output_path, f'img_{self.curr_image_index:06d}.png')
                    image_pil.save(fname)

                self.curr_image_index += 1
                return False

            except Exception as e:
                print ('Error: SkeletonVisualizer::capture_frame ' + str(e))
                return True

        def show_next_frame(vis):
            try:
                t = self.curr_frame + 1
                if t >= self.max_frames:
                    t = 0
                self.curr_frame = t
                print (f'curr frame: ', t)

                if len(self.pcd_scene_list) > 0 and self.multiple_scenes:
                    self.pcd_base.points = self.pcd_scene_list[t].points
                    self.pcd_base.colors = self.pcd_scene_list[t].colors
                    vis.update_geometry(self.pcd_base)

                for n in range(len(self.obj_list)):
                    self.obj_list[n].vertices = copy.deepcopy(self.sphere_list[t][n].vertices)
                    self.obj_list[n].compute_vertex_normals()
                    vis.update_geometry(self.obj_list[n])
                
                for n in range(len(self.edges)):
                    self.edges[n].points = copy.deepcopy(self.lines_list[t][n].points)
                    self.edges[n].lines = copy.deepcopy(self.lines_list[t][n].lines)
                    vis.update_geometry(self.edges[n])

                capture_frame(vis)

                return False

            except Exception as e:
                print ('Error: SkeletonVisualizer::show_next_frame ' + str(e))
                return True

        key_to_callback = {}
        key_to_callback[ord("C")] = capture_frame
        key_to_callback[ord("N")] = show_next_frame
        key_to_callback[ord("U")] = update_camera_callback(self.window_size, self.K)
        key_to_callback[ord("R")] = load_render_option_callback(self.renderoption_filename)

        self.obj_list = []
        for n in range(len(self.sphere_list[0])):
            self.obj_list.append(copy.deepcopy(self.sphere_list[0][n]))
        self.edges = []
        for n in range(len(self.lines_list[0])):
            self.edges.append(copy.deepcopy(self.lines_list[0][n]))

        # Draw open3d Coordinate system
        self.axis_obj = []
        if self.show_coordinate_axis:
            axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            axis_mesh = axis_mesh.translate((2*self.x1, 2*self.y1 - 1, 2*self.z1))
            axis_mesh.transform([[0.5, 0, 0, 0], [0.0, -0.5, 0, 0], [0, 0, -0.5, 0], [0, 0, 0, 1]])
            self.axis_obj = [axis_mesh]

        all_objs = self.obj_list + self.edges + self.axis_obj
        if len(self.pcd_scene_list) > 0:
            all_objs.append(self.pcd_base)
        if self.mesh_floor is not None:
            all_objs.append(self.mesh_floor)

        o3d.visualization.draw_geometries_with_key_callbacks(all_objs, key_to_callback)


if __name__ == '__main__':
    import sys
    import cv2
    import pickle
    import torch

    from .config import ConfigContext
    from .config import parse_args
    from .smpl import SMPL

    parsed_args = parse_args(sys.argv[1:]) if len(sys.argv) > 1 else None
    with ConfigContext(parsed_args):
        kargs = {}
        for key, value in parsed_args.smpl.items():
            kargs[key] = value

        for key, value in parsed_args.data.items():
            kargs[key] = value
        print ("Info: writing output to ", parsed_args.output_path)

        smpl_J_reg_extra_path = os.path.join(kargs['smpl_model_parameters_path'], 'J_regressor_extra.npy')
        smpl_J_reg_h37m_path = os.path.join(kargs['smpl_model_parameters_path'], 'J_regressor_h36m.npy')
        smpl_J_reg_alphapose_path = os.path.join(kargs['smpl_model_parameters_path'], 'SMPL_AlphaPose_Regressor_RMSprop_6.npy')
        smpl_J_reg_mupots_path = os.path.join(kargs['smpl_model_parameters_path'], 'SMPL_MuPoTs_Regressor_v1.npy')
        # if parsed_args.gpu >= 0:
        #     device = torch.device(f"cuda:{parsed_args.gpu}")
        # else:
        device = torch.device(f"cpu")

        SMPLPY = SMPL(
            kargs['smpl_model_parameters_path'],
            J_reg_extra9_path=smpl_J_reg_extra_path,
            J_reg_h36m17_path=smpl_J_reg_h37m_path,
            J_reg_alphapose_path=smpl_J_reg_alphapose_path,
            J_reg_mupots_path=smpl_J_reg_mupots_path).to(device)

        with open(os.path.join(parsed_args.input_path, 'optvar_stage1.pkl'), 'rb') as fip:
            optvar_stage1 = pickle.load(fip)

        if optvar_stage1['betas_smpl'].shape[0] == 1:
            T = len(optvar_stage1['poses_T'])
            optvar_stage1['betas_smpl'] = np.tile(optvar_stage1['betas_smpl'], (T, 1, 1))

        scene_depth = optvar_stage1['scene_depth']
        scene_img = optvar_stage1['scene_img']
        scene_mask = optvar_stage1['scene_mask']

        with open(os.path.join(parsed_args.input_path, 'visualization_data_stage1.pkl'), 'rb') as fip:
            vis_data_stage1 = pickle.load(fip)

        images = vis_data_stage1['images']
        depths = vis_data_stage1['depths']
        backmasks = vis_data_stage1['backmasks']
        cam_smpl = vis_data_stage1['cam_smpl']
        valid_smpl = vis_data_stage1['valid']
        verts = vis_data_stage1['verts']
        cam = vis_data_stage1['cam']

        # Filter scene depth
        scene_depth = cv2.bilateralFilter(scene_depth, 15, sigmaColor=1, sigmaSpace=31)
        pred_disp = cv2.bilateralFilter(1 / scene_depth, 15, sigmaColor=0.3, sigmaSpace=31)
        scene_depth = 1/pred_disp

        # Compute vertices from Stage I
        T, N = optvar_stage1['poses_T'].shape[0:2]
        H, W = images.shape[1:3]

        results = SMPLPY(betas=optvar_stage1['betas_smpl'].reshape((T * N, -1)), poses=optvar_stage1['poses_smpl'].reshape((T * N, -1)))
        local_verts = results['verts'].cpu().numpy().reshape((T, N, -1, 3))
        verts3d_stage1 = optvar_stage1['scale_factor'] * local_verts  + optvar_stage1['poses_T']

        vis3d = SceneHumansVisualizer(scene_img.copy(),
                    scene_depth.copy(),
                    vis_data_stage1['cam'],
                    vertices=verts3d_stage1[:, :],
                    faces=SMPLPY.faces,
                    valid_vertices=vis_data_stage1['valid'] >= 0.5,
                    capture_rendered_color=False,
                    capture_rendered_depth=False,
                    show_coordinate_axis=False,
                    renderoption_filename='data/renderoption.json',
                    output_path=parsed_args.output_path,
                    verbose=True)
        vis3d.run()
