import os
import sys
import time
import datetime
import json
import logging

import torch
import torch.nn as nn
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate
import torchvision
from torchvision.transforms import Compose
import numpy as np

import utils
import network
from vis import *
from dataset import FWIDataset
import transforms as T
import pytorch_ssim

def setup_logger(save_dir):
    log_file = os.path.join(save_dir, 'test.log')
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging.getLogger('')

def evaluate(model, criterions, dataloader, device, k, ctx,
             vis_path, vis_batch, vis_sample, missing, std, logger):
    model.eval()

    label_list, label_pred_list= [], []
    label_tensor, label_pred_tensor = [], []
    if missing or std:
        data_list, data_noise_list = [], []

    with torch.no_grad():
        batch_idx = 0
        for data, label in dataloader:
            data = data.type(torch.FloatTensor).to(device, non_blocking=True)
            label = label.type(torch.FloatTensor).to(device, non_blocking=True)

            label_np = T.tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'], exp=False)
            label_list.append(label_np)
            label_tensor.append(label)

            if missing or std:
                data_noise = torch.clip(data + (std ** 0.5) * torch.randn(data.shape).to(device, non_blocking=True), min=-1, max=1)
                mute_idx = np.random.choice(data.shape[3], size=missing, replace=False)
                data_noise[:, :, :, mute_idx] = data[0, 0, 0, 0]
                data_np = T.tonumpy_denormalize(data, ctx['data_min'], ctx['data_max'], k=k)
                data_noise_np = T.tonumpy_denormalize(data_noise, ctx['data_min'], ctx['data_max'], k=k)
                data_list.append(data_np)
                data_noise_list.append(data_noise_np)
                pred = model(data_noise)
            else:
                pred = model(data)

            label_pred_np = T.tonumpy_denormalize(pred, ctx['label_min'], ctx['label_max'], exp=False)
            label_pred_list.append(label_pred_np)
            label_pred_tensor.append(pred)

            if vis_path and batch_idx < vis_batch:
                for i in range(vis_sample):
                    plot_velocity(label_pred_np[i, 0], label_np[i, 0], f'{vis_path}/V_{batch_idx}_{i}.png')
                    if missing or std:
                        for ch in [2]:
                            plot_seismic(data_np[i, ch], data_noise_np[i, ch], f'{vis_path}/S_{batch_idx}_{i}_{ch}.png',
                                vmin=ctx['data_min'] * 0.01, vmax=ctx['data_max'] * 0.01)
            batch_idx += 1

    label, label_pred = np.concatenate(label_list), np.concatenate(label_pred_list)
    label_t, pred_t = torch.cat(label_tensor), torch.cat(label_pred_tensor)
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    logger.info(f'MAE: {l1(label_t, pred_t)}')
    logger.info(f'MSE: {l2(label_t, pred_t)}')
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    logger.info(f'SSIM: {ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)}')

    for name, criterion in criterions.items():
        logger.info(f' * Velocity {name}: {criterion(label, label_pred)}')

def main(args):
    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    utils.mkdir(args.output_path)
    logger = setup_logger(args.output_path)

    logger.info(args)
    logger.info(f"torch version: {torch.__version__}, torchvision version: {torchvision.__version__}")

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            logger.error('Unsupported dataset.')
            sys.exit()

    if args.file_size is not None:
        ctx['file_size'] = args.file_size

    logger.info("Loading validation data")
    log_data_min = T.log_transform(ctx['data_min'], k=args.k)
    log_data_max = T.log_transform(ctx['data_max'], k=args.k)
    transform_valid_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(log_data_min, log_data_max),
    ])

    transform_valid_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])
    if args.val_anno[-3:] == 'txt':
        dataset_valid = FWIDataset(
            args.val_anno,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data=transform_valid_data,
            transform_label=transform_valid_label
        )
    else:
        dataset_valid = torch.load(args.val_anno)

    logger.info("Creating data loaders")
    valid_sampler = SequentialSampler(dataset_valid)
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    logger.info("Creating model")
    if args.model not in network.model_dict:
        logger.error('Unsupported model.')
        sys.exit()

    # 修改后：
    if args.model == "UViT":
        model = network.model_dict["UViT"]().to(device)
    else:
        model = network.model_dict[args.model](
            upsample_mode=args.up_mode,
            sample_spatial=args.sample_spatial,
            sample_temporal=args.sample_temporal,
            norm=args.norm
        ).to(device)
    
    criterions = {
        'MAE': lambda x, y: np.mean(np.abs(x - y)),
        'MSE': lambda x, y: np.mean((x - y) ** 2)
    }

    if args.resume:
        logger.info(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])  # ✅ 只加载权重
        else:
            model.load_state_dict(checkpoint)  # 兼容只包含权重的 .pth 文件
        logger.info('Loaded model checkpoint.')

    vis_path = None
    if args.vis:
        vis_folder = f'visualization_{args.vis_suffix}' if args.vis_suffix else 'visualization'
        vis_path = os.path.join(args.output_path, vis_folder)
        utils.mkdir(vis_path)

    logger.info("Start testing")
    start_time = time.time()
    evaluate(model, criterions, dataloader_valid, device, args.k, ctx,
             vis_path, args.vis_batch, args.vis_sample, args.missing, args.std, logger)
    total_time = time.time() - start_time
    logger.info('Testing time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Testing')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='flatfault-b', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=int, help='number of samples in each npy file')
    parser.add_argument('-ap', '--anno-path', default='split_files', help='annotation files location')
    parser.add_argument('-v', '--val-anno', default='flatfault_b_val_invnet.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models', help='path to parent folder to save checkpoints')
    parser.add_argument('-n', '--save-name', default='fcn_l1loss_ffb', help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')
    parser.add_argument('-m', '--model', type=str, help='inverse model name')
    parser.add_argument('-no', '--norm', default='bn', help='normalization layer type, support bn, in, ln (default: bn)')
    parser.add_argument('-um', '--up-mode', default=None, help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    parser.add_argument('-b', '--batch-size', default=50, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--vis', help='visualization option', action="store_true")
    parser.add_argument('-vsu','--vis-suffix', default=None, type=str, help='visualization suffix')
    parser.add_argument('-vb','--vis-batch', help='number of batch to be visualized', default=0, type=int)
    parser.add_argument('-vsa', '--vis-sample', help='number of samples in a batch to be visualized', default=0, type=int)
    parser.add_argument('--missing', default=0, type=int, help='number of missing traces')
    parser.add_argument('--std', default=0, type=float, help='standard deviation of gaussian noise')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)