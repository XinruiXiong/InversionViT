import os
import json
import torch
import logging
import datetime
import argparse
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
import transforms as T
from dataset import FWIDataset
from network import ResAttUNetTransformerFWI

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        filemode='a',
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    return logging.getLogger("")

def denormalize(tensor, min_val, max_val):
    return tensor * (max_val - min_val) / 2 + (max_val + min_val) / 2

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer, logger):
    model.train()
    running_loss = 0.0
    for i, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if writer:
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(loader) + i)
    avg_loss = running_loss / len(loader)
    logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
    return avg_loss

def evaluate(model, loader, criterion, ctx, device, epoch, writer, logger):
    model.eval()
    mae_fn = nn.L1Loss()
    total_loss, total_mae = 0.0, 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()

            pred_denorm = denormalize(pred, ctx["label_min"], ctx["label_max"])
            y_denorm = denormalize(y, ctx["label_min"], ctx["label_max"])
            mae = mae_fn(pred_denorm, y_denorm)
            total_mae += mae.item()

            if writer:
                writer.add_scalar("Val/MAE", mae.item(), epoch * len(loader) + i)

    avg_loss = total_loss / len(loader)
    avg_mae = total_mae / len(loader)
    logger.info(f"Epoch {epoch} | Val Loss: {avg_loss:.6f} | Val MAE: {avg_mae:.6f}")
    return avg_loss, avg_mae

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("dataset_config.json") as f:
        ctx = json.load(f)[args.dataset]

    logger = setup_logger(args.log_dir)
    writer = SummaryWriter(args.log_dir)

    transform_data = Compose([
        T.LogTransform(k=1.0),
        T.MinMaxNormalize(T.log_transform(ctx["data_min"], 1.0), T.log_transform(ctx["data_max"], 1.0))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(ctx["label_min"], ctx["label_max"])
    ])

    train_dataset = FWIDataset(
        args.train_anno,
        sample_ratio=1,
        file_size=ctx["file_size"],
        transform_data=transform_data,
        transform_label=transform_label
    )

    val_dataset = FWIDataset(
        args.val_anno,
        sample_ratio=1,
        file_size=ctx["file_size"],
        transform_data=transform_data,
        transform_label=transform_label
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset), num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = ResAttUNetTransformerFWI().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.L1Loss()

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, logger)
        _, val_mae = evaluate(model, val_loader, criterion, ctx, device, epoch, writer, logger)
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), os.path.join(args.log_dir, "best_model.pth"))
            logger.info(f"Epoch {epoch}: Best model saved with MAE {best_mae:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResAttUNetTransformerFWI for FWI")
    parser.add_argument("--dataset", type=str, required=True, help="dataset name (e.g., kagglemix)")
    parser.add_argument("--train-anno", type=str, required=True, help="path to train annotation file")
    parser.add_argument("--val-anno", type=str, required=True, help="path to val annotation file")
    parser.add_argument("--log-dir", type=str, default="resatt_logs", help="directory to save logs and model")
    parser.add_argument("--batch-size", type=int, default=16, help="training batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="number of DataLoader workers")
    parser.add_argument("--epochs", type=int, default=100, help="total number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    args = parser.parse_args()
    main(args)