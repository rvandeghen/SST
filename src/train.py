"""
Script adpated from https://github.com/pytorch/vision/tree/main/references/detection
"""

import datetime
import os
import time

import numpy as np
import random
import wandb

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
from models import fasterrcnn_resnet50_fpn, AnchorGenerator

from dataset import build_dataset

from engine import train_one_epoch, evaluate

import presets
import utils


def get_dataset(root, image_set, transform):

    dataset = build_dataset(root, image_set, transform)
    num_classes = dataset.num_classes()

    return dataset, num_classes


def get_transform(train):
    return presets.DetectionPresetTrain() if train else presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='SoccerNetv3 Detection Training', add_help=add_help)

    # Dataset
    parser.add_argument('--data-path',
                        help='dataset')
    parser.add_argument('--split', default='train',
                        help='chose between training or pseudo-labeled dataset')
    parser.add_argument('--output-dir', default='',
                        help='path where to save, leave empty if no saving')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', type=str,
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--epochs', default=200, type=int)
        
    # Model
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn',
                        help='model')
    parser.add_argument('--rpn-score-thresh', default=None, type=float,
                        help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=0, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--sync-bn', action='store_true',
                        help='use sync batch norm')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='use pretrained backbone')
    parser.add_argument('--tl', default=0.9, type=float,
                        help='Value for tau_low (default: 0.9')
    parser.add_argument('--th', default=1., type=float,
                        help='Value for tau_high (default: 1.)')

    # Optimizer
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')

    # Learning rate
    parser.add_argument('--lr', default=0.04, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr-scheduler', default='multisteplr',
                        help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr-steps', nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--patience', default=5, type=int,
                        help='Number of epochs t')

    # Misc
    parser.add_argument('--test-only', action='store_true',
                        help='Only test the model')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    if torch.distributed.get_rank() == 0:
        wandb.init(config=args, project="debug")
    config = wandb.config
    print(args)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('No cuda device detected')
        return
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print('Using seed: {}'. format(torch.initial_seed()))

    # Data loading code
    print("Loading data")

    print('Loading training data')
    dataset, num_classes = get_dataset(args.data_path, args.split, get_transform(True))
    print('Loading eval data')
    dataset_test, _ = get_dataset(args.data_path, "eval", get_transform(False))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)


    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    kwargs = {}
    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers
    }

    scales = tuple((x * 0.337, x * 0.517, x * 1.939) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.289, 1.0, 3.458),) * len(scales)

    anchor_generator = AnchorGenerator(scales, aspect_ratios)
    kwargs["rpn_anchor_generator"] = anchor_generator
    kwargs["tau_l"] = args.tl
    kwargs["tau_h"] = args.th
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes+1, pretrained_backbone=args.pretrained,
                                                              **kwargs)

    print('model created')

    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=args.lr_gamma, patience=args.patience, verbose=True)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                           "are supported.".format(args.lr_scheduler))

    if args.resume:
        print('Resuming')
        checkpoint = torch.load(args.resume, map_location=device)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

        for g in optimizer.param_groups:
            g['lr'] = 0.02

        lr_scheduler.best = 0

    if args.test_only:
        stats = evaluate(model, data_loader_test, device=device)
        print(stats[0]*100)
        return

    best_map = 0.
    counter = 0
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args)
        # evaluate after every epoch

        stats = evaluate(model, data_loader_test, device=device)
        if args.lr_scheduler != 'plateau':
            lr_scheduler.step()
        else:
            lr_scheduler.step(100*stats[0])
        print('mAP .50:.95 = {}'.format(100*stats[0]))
        if torch.distributed.get_rank() == 0:
            wandb.log({'map': 100*stats[0]})

        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'anchor_generator': anchor_generator,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch,
                'map': 100*stats[0]
            }
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

        if 100*stats[0] > best_map:
            print('Improved mAP .50:.95 from {} to {} (delta = {})'.format(best_map, 100*stats[0], (100*stats[0]-best_map)))
            best_map = 100*stats[0]
            counter = 0
        else:
            counter += 1

        if counter == 10:
            print('Early stropping')
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Best mAP .50: {}'.format(best_map))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
