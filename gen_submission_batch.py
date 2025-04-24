# import os
# import json
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# from network import InversionNet
# import transforms as T
# from torchvision.transforms import Compose
# from torch.utils.data import Dataset, DataLoader

# # ======== 配置 ========
# model_path = "Invnet_models/fcn_l1loss/checkpoint.pth"
# test_dir = "/home/xinrui/fwi_data/test"
# config_path = "dataset_config.json"
# dataset_name = "kagglemix"
# output_csv = "submission.csv"
# batch_size = 32
# num_workers = 16
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ======== 加载 dataset 配置 ========
# with open(config_path, 'r') as f:
#     cfg = json.load(f)[dataset_name]
# data_min, data_max = cfg["data_min"], cfg["data_max"]
# label_min, label_max = cfg["label_min"], cfg["label_max"]
# n_grid = cfg["n_grid"]

# # ======== 数据预处理 ========
# transform = Compose([
#     T.LogTransform(k=1.0),
#     T.MinMaxNormalize(T.log_transform(data_min, 1.0), T.log_transform(data_max, 1.0))
# ])

# # ======== 构建 Dataset 类 ========
# class TestFWIDataset(Dataset):
#     def __init__(self, test_dir, transform=None):
#         self.fnames = sorted([f for f in os.listdir(test_dir) if f.endswith(".npy")])
#         self.test_dir = test_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.fnames)

#     def __getitem__(self, idx):
#         fname = self.fnames[idx]
#         path = os.path.join(self.test_dir, fname)
#         data = np.load(path)  # [5, 1000, 70]
#         if self.transform:
#             data = self.transform(data)
#         return torch.tensor(data, dtype=torch.float32), fname

# # ======== 加载模型 ========
# model = InversionNet().to(device)
# checkpoint = torch.load(model_path, map_location=device)
# state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
# model.load_state_dict(state_dict)
# model.eval()

# # ======== 创建 DataLoader ========
# dataset = TestFWIDataset(test_dir, transform=transform)
# loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# # ======== 准备列名 ========
# x_cols = [f"x_{i}" for i in range(1, n_grid, 2)]  # x_1, x_3, ..., x_69
# header = ["oid_ypos"] + x_cols
# all_rows = []

# # ======== 批量推理并写入行 ========
# for data_batch, fname_batch in tqdm(loader, desc="Batch Predicting"):
#     data_batch = data_batch.to(device)  # [B, 5, 1000, 70]
#     with torch.no_grad():
#         pred_batch = model(data_batch).squeeze(1).cpu().numpy()  # [B, 70, 70]

#     # 反归一化
#     pred_batch = pred_batch * (label_max - label_min) / 2 + (label_max + label_min) / 2

#     for b, fname in enumerate(fname_batch):
#         for y in range(pred_batch.shape[1]):
#             row_id = f"{fname[:-4]}_y_{y}"
#             row = [row_id] + list(pred_batch[b, y, ::2])  # 取 x_1, x_3, ..., x_69
#             all_rows.append(row)

# # ======== 写入 CSV ========
# df = pd.DataFrame(all_rows, columns=header)
# df.to_csv(output_csv, index=False)
# print(f"✅ Saved submission to {output_csv}")


import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from network import UViT  # ✅ 使用 UViT 模型
import transforms as T
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader

# ======== 配置 ========
model_path = "./UViT_models/uvit_final/checkpoint.pth"
test_dir = "/home/xinrui/fwi_data/test"
config_path = "dataset_config.json"
dataset_name = "kagglemix"
output_csv = "submission.csv"
batch_size = 32
num_workers = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== 加载 dataset 配置 ========
with open(config_path, 'r') as f:
    cfg = json.load(f)[dataset_name]
data_min, data_max = cfg["data_min"], cfg["data_max"]
label_min, label_max = cfg["label_min"], cfg["label_max"]
n_grid = cfg["n_grid"]

# ======== 数据预处理 ========
transform = Compose([
    T.LogTransform(k=1.0),
    T.MinMaxNormalize(T.log_transform(data_min, 1.0), T.log_transform(data_max, 1.0))
])

# ======== 构建 Dataset 类 ========
class TestFWIDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.fnames = sorted([f for f in os.listdir(test_dir) if f.endswith(".npy")])
        self.test_dir = test_dir
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.test_dir, fname)
        data = np.load(path)  # [5, 1000, 70]
        if self.transform:
            data = self.transform(data)
        return torch.tensor(data, dtype=torch.float32), fname

# ======== 加载 UViT 模型 ========
model = UViT().to(device)
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

# ======== 创建 DataLoader ========
dataset = TestFWIDataset(test_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# ======== 准备列名 ========
x_cols = [f"x_{i}" for i in range(1, n_grid, 2)]  # x_1, x_3, ..., x_69
header = ["oid_ypos"] + x_cols
all_rows = []

# ======== 批量推理并写入行 ========
for data_batch, fname_batch in tqdm(loader, desc="Batch Predicting"):
    data_batch = data_batch.to(device)  # [B, 5, 1000, 70]
    with torch.no_grad():
        pred_batch = model(data_batch).squeeze(1).cpu().numpy()  # [B, 70, 70]

    # 反归一化
    pred_batch = pred_batch * (label_max - label_min) / 2 + (label_max + label_min) / 2

    for b, fname in enumerate(fname_batch):
        for y in range(pred_batch.shape[1]):
            row_id = f"{fname[:-4]}_y_{y}"
            row = [row_id] + list(pred_batch[b, y, ::2])  # 取 x_1, x_3, ..., x_69
            all_rows.append(row)

# ======== 写入 CSV ========
df = pd.DataFrame(all_rows, columns=header)
df.to_csv(output_csv, index=False)
print(f"✅ Saved submission to {output_csv}")
