import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from network import InversionNet
import transforms as T
from torchvision.transforms import Compose

# ======== 配置 ========
model_path = "Invnet_models/fcn_l1loss/checkpoint.pth"
test_dir = "/home/xinrui/fwi_data/test"
config_path = "dataset_config.json"
dataset_name = "kagglemix"
output_csv = "submission.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== 加载 dataset 配置 ========
with open(config_path, 'r') as f:
    cfg = json.load(f)[dataset_name]

data_min, data_max = cfg["data_min"], cfg["data_max"]
label_min, label_max = cfg["label_min"], cfg["label_max"]
n_grid = cfg["n_grid"]  # 通常是 70

# ======== 数据预处理 ========
transform = Compose([
    T.LogTransform(k=1.0),
    T.MinMaxNormalize(T.log_transform(data_min, 1.0), T.log_transform(data_max, 1.0))
])

# ======== 加载模型 ========
model = InversionNet().to(device)
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.eval()

# ======== 构建列名 ========
x_cols = [f"x_{i}" for i in range(1, n_grid, 2)]  # x_1, x_3, ..., x_69
header = ["oid_ypos"] + x_cols
all_rows = []

# ======== 执行预测并填充每行 ========
file_list = sorted([f for f in os.listdir(test_dir) if f.endswith(".npy")])

for fname in tqdm(file_list, desc="Predicting"):
    npy_path = os.path.join(test_dir, fname)
    data = np.load(npy_path)  # shape: [5, 1000, 70]
    data = transform(data)
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 5, 1000, 70]

    with torch.no_grad():
        pred = model(data).squeeze(0).squeeze(0).cpu().numpy()  # [70, 70]

    # 反归一化
    pred = pred * (label_max - label_min) / 2 + (label_max + label_min) / 2
    
    # 每一行作为一个 submission 的 entry
    for y in range(pred.shape[0]):
        row_id = f"{fname[:-4]}_y_{y}"  # 去掉 .npy，加上行号
        row = [row_id] + list(pred[y, ::2])  # 隔列采样
        all_rows.append(row)

    del data, pred
    torch.cuda.empty_cache()

# ======== 写入 CSV ========
df = pd.DataFrame(all_rows, columns=header)
df.to_csv(output_csv, index=False)
print(f"✅ Saved submission to {output_csv}")
