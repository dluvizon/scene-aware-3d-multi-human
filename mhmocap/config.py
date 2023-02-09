import os
import sys
import argparse
import yaml
import time

currentfile = os.path.abspath(__file__)
project_dir = currentfile.replace('/mhmocap/config.py','')

model_dir = os.path.join(project_dir, 'model_data')
trained_model_dir = os.path.join(project_dir, 'trained_models')


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description='Scene-Aware 3D Multi-Human Motion Capture')

    parser.add_argument('-f', type=str, help='Default argument (used only for compatibility in Jupyter Lab)')
    parser.add_argument('--configs_yml', type=str, default='configs/default.yml', help='Config file with additional options')
    parser.add_argument('--ts_id', type=int, default=1, help='Test set ID, required for MuPoTs')
    parser.add_argument('--cam', type=int, default=0, help='Camera ID, required for Studio sequences')

    parser.add_argument('--cmu_sequence_id', type=str, default='', help='CMU Panoptic sequence ID')
    parser.add_argument('--cmu_camera_node', type=int, default=16, help='CMU Panoptic camera node (from 0 to 30)')
    parser.add_argument('--cmu_clip_id', type=int, default=1, help='CMU Panoptic clip ID (custom split)')

    parser.add_argument('--input_path', type=str, default='', help='Input path')
    parser.add_argument('--output_path', type=str, default='./output', help='Output path')

    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--resize_factor', type=float, default=0.25, help='Image rescaling factor')
    parser.add_argument('--num_iter', type=int, default=200, help='Number of iterations in the optimization')
    parser.add_argument('--proj2d_loss_coef', type=float, default=1.0, help='Coefficient of the 2D projection loss')
    parser.add_argument('--depth_loss_coef', type=float, default=1.0, help='Coefficient of the depth loss')
    parser.add_argument('--silhouette_loss_coef', type=float, default=1.0, help='Coefficient of the silhouette loss')
    parser.add_argument('--reg_velocity_coef', type=float, default=1.0, help='Coefficient of the velocity regularization loss')
    parser.add_argument('--reg_verts_filter_coef', type=float, default=1.0, help='Coefficient of the vertices filtering regularization loss')
    parser.add_argument('--reg_poses_coef', type=float, default=10.0, help='Coefficient of the reference 3D pose loss')
    parser.add_argument('--reg_scales_coef', type=float, default=10.0, help='Coefficient of the scale regularization loss')
    parser.add_argument('--reg_contact_coef', type=float, default=1.0, help='Coefficient of the contact regularization loss')
    parser.add_argument('--reg_foot_sliding_coef', type=float, default=1.0, help='Coefficient of the food sliding regularization loss')

    parsed_args = parser.parse_args(args=input_args)

    config_yml_path = os.path.join(project_dir, parsed_args.configs_yml)
    # Update args with the content from yml file
    with open(config_yml_path) as file:
        configs_update = yaml.full_load(file)

    for key, value in configs_update['ARGS'].items():
        # make sure to update the configurations from .yml that not appears in input_args.
        appear_in_input_args = False
        for input_arg in input_args:
            if isinstance(input_arg, str):
                if '--{}'.format(key) in input_arg:
                    appear_in_input_args = True
        if appear_in_input_args:
            continue
        
        if isinstance(value, str):
            exec("parsed_args.{} = '{}'".format(key, value))
        else:
            exec("parsed_args.{} = {}".format(key, value))

    if 'smpl' in configs_update:
        parsed_args.smpl = configs_update['smpl']

    if 'data' in configs_update:
        parsed_args.data = configs_update['data']
        # if input_path is given, overwrite data.data_path
        if parsed_args.input_path != "":
            parsed_args.data['data_path'] = parsed_args.input_path

    if 'studio' in configs_update:
        parsed_args.studio = configs_update['studio']

    if 'internet' in configs_update:
        parsed_args.internet = configs_update['internet']

    return parsed_args


class ConfigContext(object):
    """
    Class to manage the active current configuration, creates temporary `yaml`
    file containing the configuration currently being used so it can be
    accessed anywhere.
    """
    parsed_args = parse_args(sys.argv[1:])

    def __init__(self, parsed_args=None):
        if parsed_args is not None:
            self.parsed_args = parsed_args

    def __enter__(self):
        return self.parsed_args
            
    def clean(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        # delete the yaml file
        self.clean()


def args():
    return ConfigContext.parsed_args