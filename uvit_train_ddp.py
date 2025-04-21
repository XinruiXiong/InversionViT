import os
import sys
import time
import datetime
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
import torchvision

import utils
import network
from dataset import FWIDataset
from scheduler import WarmupMultiStepLR
import transforms as T

step = 0

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
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
    return logging.getLogger('')

def train_one_epoch(model, criterion, optimizer, scheduler, loader, device, epoch, print_freq, writer, logger):
    global step
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(fmt="{value:.6f}"))
    metric_logger.add_meter("samples/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}"))
    header = f"Epoch [{epoch}]"

    for data, label in metric_logger.log_every(loader, print_freq, header):
        start = time.time()
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        pred = model(data)
        loss, g1, g2 = criterion(pred, label)
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item(), loss_g1v=g1.item(), loss_g2v=g2.item(), lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters["samples/s"].update(data.size(0) / (time.time() - start))

        if writer:
            writer.add_scalar("loss", loss.item(), step)
            writer.add_scalar("loss_g1v", g1.item(), step)
            writer.add_scalar("loss_g2v", g2.item(), step)
        step += 1
        scheduler.step()

def evaluate(model, criterion, loader, device, writer, logger):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"
    with torch.no_grad():
        for data, label in metric_logger.log_every(loader, 20, header):
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss, g1, g2 = criterion(pred, label)
            metric_logger.update(loss=loss.item(), loss_g1v=g1.item(), loss_g2v=g2.item())

    metric_logger.synchronize_between_processes()
    val_loss = metric_logger.loss.global_avg
    logger.info(f"Validation Loss: {val_loss:.6f}")
    if writer:
        writer.add_scalar("val_loss", val_loss, step)
        writer.add_scalar("val_loss_g1v", metric_logger.loss_g1v.global_avg, step)
        writer.add_scalar("val_loss_g2v", metric_logger.loss_g2v.global_avg, step)
    return val_loss

def main(args):
    global step

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.log_path = os.path.join(args.log_path, args.save_name, args.suffix or '')
    args.train_anno = os.path.join(args.anno_path, args.train_anno)
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    args.epochs = args.epoch_block * args.num_block

    utils.mkdir(args.output_path)
    utils.init_distributed_mode(args)

    is_main = not args.distributed or (args.rank == 0 and args.local_rank == 0)
    logger = setup_logger(args.output_path) if is_main else logging.getLogger()

    if is_main:
        logger.info("Training args:\n" + str(args))
        logger.info(f"Torch: {torch.__version__}, TorchVision: {torchvision.__version__}")

    train_writer = val_writer = None
    if args.tensorboard and is_main:
        utils.mkdir(args.log_path)
        train_writer = SummaryWriter(os.path.join(args.output_path, "logs/train"))
        val_writer = SummaryWriter(os.path.join(args.output_path, "logs/val"))

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open("dataset_config.json") as f:
        ctx = json.load(f)[args.dataset]
    if args.file_size:
        ctx["file_size"] = args.file_size

    transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(ctx['data_min'], args.k), T.log_transform(ctx['data_max'], args.k))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])

    dataset_train = FWIDataset(
        args.train_anno,
        sample_ratio=args.sample_temporal,
        file_size=ctx["file_size"],
        transform_data=transform_data,
        transform_label=transform_label
    )
    dataset_val = FWIDataset(
        args.val_anno,
        sample_ratio=args.sample_temporal,
        file_size=ctx["file_size"],
        transform_data=transform_data,
        transform_label=transform_label
    )

    train_sampler = DistributedSampler(dataset_train, shuffle=True) if args.distributed else torch.utils.data.RandomSampler(dataset_train)
    val_sampler = DistributedSampler(dataset_val, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dataset_val)

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=default_collate)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.workers, pin_memory=True, collate_fn=default_collate)

    # ✅ UViT 不需要 upsample_mode / sample_temporal 参数
    if args.model == "UViT":
        model = network.model_dict["UViT"]().to(device)
    else:
        model = network.model_dict[args.model](upsample_mode=args.up_mode,
            sample_spatial=args.sample_spatial, sample_temporal=args.sample_temporal).to(device)

    if args.distributed and args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    def criterion(pred, gt):
        g1, g2 = l1(pred, gt), l2(pred, gt)
        return args.lambda_g1v * g1 + args.lambda_g2v * g2, g1, g2

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * args.world_size, weight_decay=args.weight_decay)
    warmup_iters = args.lr_warmup_epochs * len(loader_train)
    lr_milestones = [len(loader_train) * m for m in args.lr_milestones]
    scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
                                   warmup_iters=warmup_iters, warmup_factor=1e-5)

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(network.replace_legacy(ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["lr_scheduler"])
        args.start_epoch = ckpt["epoch"] + 1
        step = ckpt["step"]
        scheduler.milestones = lr_milestones
        logger.info(f"Resumed from checkpoint at epoch {args.start_epoch}")

    best_loss = 1e10
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, criterion, optimizer, scheduler, loader_train, device, epoch, args.print_freq, train_writer, logger)
        val_loss = evaluate(model, criterion, loader_val, device, val_writer, logger)

        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "args": args
        }

        if val_loss < best_loss and is_main:
            torch.save(checkpoint, os.path.join(args.output_path, "checkpoint.pth"))
            logger.info(f"[Epoch {epoch}] New best val loss: {val_loss:.6f} (saved)")
            best_loss = val_loss
        elif is_main and (epoch + 1) % args.epoch_block == 0:
            torch.save(checkpoint, os.path.join(args.output_path, f"model_{epoch + 1}.pth"))
            logger.info(f"[Epoch {epoch}] Checkpoint block saved.")

    if is_main:
        total_time = time.time() - scheduler.last_epoch
        logger.info(f"Training complete in {str(datetime.timedelta(seconds=int(total_time)))}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='UViT DDP Training')
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-ds', '--dataset', required=True)
    parser.add_argument('-fs', '--file-size', type=int, default=None)
    parser.add_argument('-ap', '--anno-path', default='split_files')
    parser.add_argument('-t', '--train-anno', required=True)
    parser.add_argument('-v', '--val-anno', required=True)
    parser.add_argument('-o', '--output-path', default='UViT_models')
    parser.add_argument('-l', '--log-path', default='UViT_models')
    parser.add_argument('-n', '--save-name', default='uvit_exp')
    parser.add_argument('-s', '--suffix', type=str, default=None)
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('--up-mode', default=None)
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0)
    parser.add_argument('-st', '--sample-temporal', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('-lm', '--lr-milestones', nargs='+', type=int, default=[])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-gamma', type=float, default=0.1)
    parser.add_argument('--lr-warmup-epochs', type=int, default=0)
    parser.add_argument('-eb', '--epoch_block', type=int, default=40)
    parser.add_argument('-nb', '--num_block', type=int, default=3)
    parser.add_argument('-j', '--workers', type=int, default=16)
    parser.add_argument('--k', type=float, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('-r', '--resume', default=None)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=1.0)
    parser.add_argument('--sync-bn', action='store_true')
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--dist-url', default='env://')
    parser.add_argument('--tensorboard', action='store_true')
    args = parser.parse_args()
    main(args)
