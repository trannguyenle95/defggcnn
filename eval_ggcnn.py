import argparse
import logging

import torch.utils.data

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GG-CNN')

    # Network
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')

    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--use-stiffness', type=int, default=1, help='Use Stiffness image for training (1/0)')

    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.1, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--write2isaac', action='store_true', help='Jacquard-dataset style output')

    parser.add_argument('--vis', action='store_true', help='Visualise the network output')
    parser.add_argument('--predict', action='store_true', help='Predict output')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()
    # Load Network
    net = torch.load(args.network)
    device = torch.device("cuda:0")

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                           random_rotate=args.augment, random_zoom=args.augment,
                           include_stiffness=args.use_stiffness,include_depth=args.use_depth, include_rgb=args.use_rgb)
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    results = {'correct': 0, 'failed': 0}

    if args.jacquard_output:
        jo_fn = args.network + '_jacquard_output.txt'
        with open(jo_fn, 'w') as f:
            pass
 
    with torch.no_grad():
        if args.dataset == "softnettest":
            df = pd.DataFrame({"name":["asd"],
            "center_j1":[0],
            "center_i1":[0],
            "angle1":[0],
            "center_3d1":["0,0,0"],
            "center_j2":[0],
            "center_i2":[0],
            "angle2":[0],
            "center_3d2":["0,0,0"],
            "center_j3":[0],
            "center_i3":[0],
            "angle3":[0],
            "center_3d3":["0,0,0"],
            "center_j4":[0],
            "center_i4":[0],
            "angle4":[0],
            "center_3d4":["0,0,0"],
            "center_j5":[0],
            "center_i5":[0],
            "angle5":[0],
            "center_3d5":["0,0,0"]})
            for idx, (x, didx, rot, zoom) in enumerate(test_data):
                logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
                xc = x.to(device)
                yc = None
                lossd = net.compute_loss(xc, yc)
                q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])
                print(test_data.dataset.get_name(didx))
                if args.vis:
                    evaluation.plot_output(test_data.dataset.get_stiffness(didx, rot, zoom),
                                        test_data.dataset.get_depth(didx, rot, zoom), q_img, width_img,
                                        ang_img,None, no_grasps=args.n_grasps, grasp_width_img=width_img)
                if args.write2isaac:
                    gs = []
                    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=args.n_grasps)
                    for g in grasps:
                        gs.append(g)
                    if len(gs) == 5:
                        df2 = pd.DataFrame({"name":[test_data.dataset.get_name(didx)],
                        "center_j1":[gs[0].center[0]],
                        "center_i1":[gs[0].center[1]],
                        "angle1":[np.degrees(gs[0].angle)],
                        "center_j2":[gs[1].center[0]],
                        "center_i2":[gs[1].center[1]],
                        "angle2":[np.degrees(gs[1].angle)],
                        "center_j3":[gs[2].center[0]],
                        "center_i3":[gs[2].center[1]],
                        "angle3":[np.degrees(gs[2].angle)],
                        "center_j4":[gs[3].center[0]],
                        "center_i4":[gs[3].center[1]],
                        "angle4":[np.degrees(gs[3].angle)],
                        "center_j5":[gs[4].center[0]],
                        "center_i5":[gs[4].center[1]],
                        "angle5":[np.degrees(gs[4].angle)],
                        })
                        df = df.append(df2,ignore_index=True)  
                    else:
                        df2 = pd.DataFrame({"name":[test_data.dataset.get_name(didx)],
                        "center_j1":[gs[0].center[0]],
                        "center_i1":[gs[0].center[1]],
                        "angle1":[np.degrees(gs[0].angle)],
                        "center_j2":[gs[1].center[0]],
                        "center_i2":[gs[1].center[1]],
                        "angle2":[np.degrees(gs[1].angle)],
                        "center_j3":[gs[2].center[0]],
                        "center_i3":[gs[2].center[1]],
                        "angle3":[np.degrees(gs[2].angle)]
                        })
                        df = df.append(df2,ignore_index=True)  
                    df.to_csv("best_grasp_egad.csv")

        else:   
            for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
                logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
                xc = x.to(device)
                yc = [yi.to(device) for yi in y]
                lossd = net.compute_loss(xc, yc)
                q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

                if args.iou_eval:
                    s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                    no_grasps=args.n_grasps,
                                                    grasp_width=width_img,
                                                    )
                    if s:
                        results['correct'] += 1
                    else:
                        results['failed'] += 1

                if args.jacquard_output:
                    grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                    with open(jo_fn, 'a') as f:
                        for g in grasps:
                            f.write(test_data.dataset.get_jname(didx) + '\n')
                            f.write(g.to_jacquard(scale=1024 / 300) + '\n')

                if args.vis:
                    evaluation.plot_output(test_data.dataset.get_stiffness(didx, rot, zoom),
                                        test_data.dataset.get_depth(didx, rot, zoom), q_img, width_img,
                                        ang_img,y[0][0,0,], no_grasps=args.n_grasps, grasp_width_img=width_img)

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                              results['correct'] + results['failed'],
                              results['correct'] / (results['correct'] + results['failed'])))

    if args.jacquard_output:
        logging.info('Jacquard output saved to {}'.format(jo_fn))