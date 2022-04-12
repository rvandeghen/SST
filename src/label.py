import datetime
import json
import os
import time

import numpy as np
import random

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
from models import fasterrcnn_resnet50_fpn

from dataset import build_dataset

from engine import label

import presets
import utils


def get_dataset(root, image_set, transform):

    dataset = build_dataset(root, image_set, transform)
    if 'train' in image_set:
        num_classes = dataset.num_classes()
        return dataset, num_classes
    else:
        return dataset


def get_transform():
    return presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='Script to generate the pseudo-labels', add_help=add_help)

    # Dataset
    parser.add_argument('--data-path',
                        help='dataset')
    parser.add_argument('--split-labeled', default="",
                        help='chose between labeled split')
    parser.add_argument('--split-unlabeled',
                        help='chose between unlabeled split')
    parser.add_argument('--output-dir', default='',
                        help='path where to save')
    parser.add_argument('-j', '--workers', default=10, type=int)
        
    # Model
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn',
                        help='model')
    parser.add_argument('--checkpoint',
                        help='checkpoint to load')

    # Score
    parser.add_argument('--score', default=0.5,
                        help='score threshold')    

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    print(torch.cuda.get_device_name())
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('No cuda device detected')
        return

    # Data loading code
    print("Loading data")

    print('Loading training data')
    dataset, num_classes = get_dataset(args.data_path, args.split_labeled, get_transform())
    print(dataset.__len__())
    print('Loading unlabeled data')
    dataset_test = get_dataset(args.data_path, args.split_unlabeled, get_transform())

    print("Creating data loaders")
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    print('mAP achieved with the model used: {}'.format(checkpoint['map']))
    anchor_generator = checkpoint['anchor_generator']
    kwargs = {}
    kwargs["rpn_anchor_generator"] = anchor_generator
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes+1, **kwargs)

    print('model created')

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Load model
    model_without_ddp.load_state_dict(checkpoint['model'])

    print("Start Labelling")
    start_time = time.time()
    results_ = label(model, data_loader_test, device, args.score)

    del model, model_without_ddp

    results_ = utils.all_gather(results_)
    results = {}
    for r in results_:
        results.update(r)

    train_dict = dataset.data
    train_dict.update(results)
    json.dump(results, open(os.path.join(args.output_dir, 'results_annotations.json'), 'w'))
    json.dump(train_dict, open(os.path.join(args.output_dir, 'pseudo_annotations.json'), 'w'))

    del results, train_dict

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Labelling time {}'.format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
