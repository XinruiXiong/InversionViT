# python vit_train.py \
#   --dataset flatvel-a \
#   --train-anno flatvel_a_train_vit_full.txt \
#   --val-anno flatvel_a_val_vit_full.txt \
#   --anno-path split_files \
#   --output-path vit_output \
#   --batch-size 16 \

import os
import sys
import time
import datetime
import json
import logging

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

import utils
from dataset import FWIDataset
from network import InversionViT
from scheduler import WarmupMultiStepLR
import transforms as T

def setup_logger(save_dir):
    log_file = os.path.join(save_dir, 'train.log')
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
    return logging

def train_one_epoch(model, criterion, optimizer, scheduler, dataloader, device, epoch, writer, logger):
    model.train()
    step = epoch * len(dataloader)
    for batch_idx, (data, label) in enumerate(dataloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss, l1, l2 = criterion(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if writer:
            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('l1_loss', l1.item(), step)
            writer.add_scalar('l2_loss', l2.item(), step)
        step += 1

        if batch_idx % 20 == 0:
            logger.info(f"Epoch {epoch} Batch {batch_idx}: Loss = {loss.item():.4f}")

def evaluate(model, criterion, dataloader, device, writer, step, logger):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss, _, _ = criterion(output, label)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    if writer:
        writer.add_scalar('val_loss', avg_loss, step)
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main(args):
    utils.mkdir(args.output_path)
    logger = setup_logger(args.output_path)
    logger.info("Starting single-GPU training with InversionViT")

    device = torch.device(args.device)

    with open('dataset_config.json') as f:
        ctx = json.load(f)[args.dataset]

    transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=args.k), T.log_transform(ctx['data_max'], k=args.k))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])

    dataset_train = FWIDataset(os.path.join(args.anno_path, args.train_anno), preload=True, 
                               file_size=ctx['file_size'], transform_data=transform_data, transform_label=transform_label)
    dataset_val = FWIDataset(os.path.join(args.anno_path, args.val_anno), preload=True, 
                             file_size=ctx['file_size'], transform_data=transform_data, transform_label=transform_label)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.workers)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, sampler=RandomSampler(dataset_val), num_workers=args.workers)

    model = InversionViT().to(device)

    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    def criterion(pred, gt):
        l1 = l1loss(pred, gt)
        l2 = l2loss(pred, gt)
        return args.lambda_g1v * l1 + args.lambda_g2v * l2, l1, l2

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupMultiStepLR(optimizer, milestones=[10, 20], gamma=0.1, warmup_iters=100, warmup_factor=1e-5)

    writer = SummaryWriter(log_dir=os.path.join(args.output_path, 'logs')) if args.tensorboard else None

    best_loss = float('inf')
    for epoch in range(args.epochs):
        train_one_epoch(model, criterion, optimizer, scheduler, dataloader_train, device, epoch, writer, logger)
        val_loss = evaluate(model, criterion, dataloader_val, device, writer, step=epoch, logger=logger)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_path, 'best_model.pth'))
            logger.info(f"Model saved at epoch {epoch} with val loss {val_loss:.4f}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', type=str, default='flatvel-a')
    parser.add_argument('-t', '--train-anno', type=str, default='flatvel_a_train.txt')
    parser.add_argument('-v', '--val-anno', type=str, default='flatvel_a_val.txt')
    parser.add_argument('-ap', '--anno-path', type=str, default='split_files')
    parser.add_argument('-o', '--output-path', type=str, default='vit_output')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lambda_g1v', type=float, default=1.0)
    parser.add_argument('--lambda_g2v', type=float, default=0.0)
    parser.add_argument('--k', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
