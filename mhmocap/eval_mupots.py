import os
import sys

import numpy as np
import json
import pickle
import torch
import copy

from .config import ConfigContext
from .config import parse_args
from .predict_mupots import build_mupots_dataloader
from .evaluate import compute_smpl_pred_error_3dproj
from .evaluate import masked_average_error
from .evaluate import masked_average_pck


def compute_mm_pck_results(optvar, ref_poses3d, visibility, dataset):

    metrics = compute_smpl_pred_error_3dproj(
            optvar, ref_poses3d=ref_poses3d, visibility=visibility,
            SMPLPY=dataset.SMPLPY, cam_K=dataset.cam['K'])

    mm_abs = 1000 * masked_average_error(metrics['abs_dist'], metrics['valid_joints'])
    mm_rel = 1000 * masked_average_error(metrics['rel_dist'], metrics['valid_joints'])
    mm_mrpe = 1000 * masked_average_error(metrics['abs_root_pos_err'], metrics['valid_root'])
    pck_rel = 100 * masked_average_pck(metrics['rel_dist'], metrics['valid_joints'], 0.15)
    ap_25_root = 100 * masked_average_pck(metrics['abs_root_pos_err'], metrics['valid_root'], 0.25)
    print (f'metrics[abs_dist]', metrics['abs_dist'].shape)
    print (f'metrics[abs_jitter]', metrics['abs_jitter'].shape)
    abs_jitter = 1000 * masked_average_error(metrics['abs_jitter'], metrics['valid_joints'])

    return {
        'mm_abs_error': mm_abs,
        'mm_rel_error': mm_rel,
        'mm_mrpe': mm_mrpe,
        'pck_rel': pck_rel,
        'ap25_root': ap_25_root,
        'abs_jitter': abs_jitter,
    }


def write_results_as_markdown(results, labels, filename):
    """
    # Arguments
        results: dict with {metric_a: [v1, v2...], metric_b: [v1, v2, ...], ...}
        labels: list of strings with labels [label_v1, label_v2, ...]
        filename: string
    """

    with open(filename, 'w') as fp:
        # Write the header and ruler
        fp.write(f"| |")
        for s in labels:
            fp.write(f" {s} |")
        fp.write(f"\n| :--: |")
        for _ in labels:
            fp.write(f" :--: |")

        # Write each metric
        for k in results.keys():
            assert len(results[k]) == len(labels), (
                f'Invalid result[{k}] ({len(results[k])}) for given labels ({len(labels)})'
                )
            fp.write(f"\n| {k} |")
            for v in results[k]:
                fp.write(f" {v:.2f} |")
        fp.write(f"\n")


def compute_average_metrics_mupots(results, list_num_instances):
    for k in results.keys():
        assert len(results[k]) == len(list_num_instances), (
            f'Invalid results[{k}] ({len(results[k])}) for given list_num_instances ({len(list_num_instances)})'
            )
        avg = np.sum(np.array(results[k]) * np.array(list_num_instances)) / np.sum(list_num_instances)
        results[k].append(float(avg))

    return results


if __name__ == '__main__':
    parsed_args = parse_args(sys.argv[1:]) if len(sys.argv) > 1 else None
    with ConfigContext(parsed_args):
        kargs = {}
        for key, value in parsed_args.smpl.items():
            kargs[key] = value

    for key, value in parsed_args.data.items():
        kargs[key] = value

    final_results = {}
    labels = []
    num_instances = [
        402, 502, 802, 602,
        522, 1082, 1293, 1102,
        1002, 502, 2103, 730,
        1023, 1878, 2287, 1503,
        1203, 378, 1293, 1503
    ]

    for ts_id in range(1, 20 + 1):
        labels.append(f"TS{ts_id}")
        print (f'Evaluating from {parsed_args.input_path}/TS{ts_id}')

        inputs_set_path = os.path.join(parsed_args.input_path, f'TS{ts_id}')

        with open(os.path.join(inputs_set_path, 'mupots_annot.pkl'), 'rb') as fip:
            mupots_annot = pickle.load(fip)

        with open(os.path.join(inputs_set_path, 'optvar_init.pkl'), 'rb') as fip:
            optvar_init = pickle.load(fip)

        with open(os.path.join(inputs_set_path, 'optvar_stage1.pkl'), 'rb') as fip:
            optvar_stage1 = pickle.load(fip)

        T = optvar_init['poses_smpl'].shape[0]
        optvar_init['betas_smpl'] = np.repeat(optvar_init['betas_smpl'], T, axis=0)
        optvar_stage1['betas_smpl'] = np.repeat(optvar_stage1['betas_smpl'], T, axis=0)

        dataset, pose3d_gt, pose3d_univ_gt, visibility = build_mupots_dataloader(
                ts_id=ts_id,
                resize_factor=parsed_args.resize_factor,
                erode_segmentation_iters=0,
                erode_backmask_iters=0,
                renormalize_depth=False,
                post_process_depth=False,
                **kargs)

        # Pre-load the initial SMPL predictions and 2D keypoints
        init_keys = ['pose2d', 'poses_smpl', 'betas_smpl', 'valid_smpl']

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

        eval_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        eval_data = next(iter(eval_loader))

        results_stage = compute_mm_pck_results(optvar_stage1,
                mupots_annot['pose3d_gt'], mupots_annot['visibility'], dataset)

        for key, val in results_stage.items():
            if key not in final_results.keys():
                final_results[key] = []
            final_results[key].append(round(val, 2))

        optvar_init_univ = copy.deepcopy(optvar_init)
        optvar_init_univ['scale_factor'] = np.ones_like(optvar_init_univ['scale_factor'])
        optvar_stage1_univ = copy.deepcopy(optvar_stage1)
        optvar_stage1_univ['scale_factor'] = np.ones_like(optvar_stage1_univ['scale_factor'])

        results_init = compute_mm_pck_results(optvar_init_univ,
                mupots_annot['pose3d_univ_gt'], mupots_annot['visibility'], dataset)
        results_stage = compute_mm_pck_results(optvar_stage1_univ,
                mupots_annot['pose3d_univ_gt'], mupots_annot['visibility'], dataset)

        for key, val in results_init.items():
            nkey = key + '_univ'

        for key, val in results_stage.items():
            nkey = key + '_univ'
            if nkey not in final_results.keys():
                final_results[nkey] = []
            final_results[nkey].append(round(val, 2))

    labels.append(f"Avg.")
    final_results = compute_average_metrics_mupots(final_results, num_instances)

    fname = os.path.join(parsed_args.input_path, 'FinalResults.json')
    with open(fname, 'w') as fp:
        json.dump({
            'final_results': final_results,
            }, fp)

    # Write markdown files with the results
    write_results_as_markdown(final_results, labels,
            os.path.join(parsed_args.input_path, 'FinalResults.md'))
