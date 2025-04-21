# # © 2022. Triad National Security, LLC. All rights reserved.

# # This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# # National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# # Department of Energy/National Nuclear Security Administration. All rights in the program are

# # reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# # Security Administration. The Government is granted for itself and others acting on its behalf a

# # nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# # derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# # others to do so.

# import os
# import numpy as np
# from torch.utils.data import Dataset
# from torchvision.transforms import Compose
# import transforms as T

# class FWIDataset(Dataset):
#     ''' FWI dataset
#     For convenience, in this class, a batch refers to a npy file 
#     instead of the batch used during training.

#     Args:
#         anno: path to annotation file
#         preload: whether to load the whole dataset into memory
#         sample_ratio: downsample ratio for seismic data
#         file_size: # of samples in each npy file
#         transform_data|label: transformation applied to data or label
#     '''
#     def __init__(self, anno, preload=True, sample_ratio=1, file_size=500,
#                     transform_data=None, transform_label=None):
#         if not os.path.exists(anno):
#             print(f'Annotation file {anno} does not exists')
#         self.preload = preload
#         self.sample_ratio = sample_ratio
#         self.file_size = file_size
#         self.transform_data = transform_data
#         self.transform_label = transform_label
#         with open(anno, 'r') as f:
#             self.batches = f.readlines()
#         if preload: 
#             self.data_list, self.label_list = [], []
#             for batch in self.batches: 
#                 data, label = self.load_every(batch)
#                 self.data_list.append(data)
#                 if label is not None:
#                     self.label_list.append(label)

#     # Load from one line
#     def load_every(self, batch):
#         batch = batch.split('\t')
#         data_path = batch[0] if len(batch) > 1 else batch[0][:-1]
#         data = np.load(data_path)[:, :, ::self.sample_ratio, :]
#         data = data.astype('float32')
#         if len(batch) > 1:
#             label_path = batch[1][:-1]    
#             label = np.load(label_path)
#             label = label.astype('float32')
#         else:
#             label = None
        
#         return data, label
        
#     def __getitem__(self, idx):
#         batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
#         if self.preload:
#             data = self.data_list[batch_idx][sample_idx]
#             label = self.label_list[batch_idx][sample_idx] if len(self.label_list) != 0 else None
#         else:
#             data, label = self.load_every(self.batches[batch_idx])
#             data = data[sample_idx]
#             label = label[sample_idx] if label is not None else None
#         if self.transform_data:
#             data = self.transform_data(data)
#         if self.transform_label and label is not None:
#             label = self.transform_label(label)
#         return data, label if label is not None else np.array([])
        
#     def __len__(self):
#         return len(self.batches) * self.file_size


# if __name__ == '__main__':
#     transform_data = Compose([
#         T.LogTransform(k=1),
#         T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1))
#     ])
#     transform_label = Compose([
#         T.MinMaxNormalize(2000, 6000)
#     ])
#     dataset = FWIDataset(f'relevant_files/temp.txt', transform_data=transform_data, transform_label=transform_label, file_size=1)
#     data, label = dataset[0]
#     print(data.shape)
#     print(label is None)

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import transforms as T


class FWIDataset(Dataset):
    """
    Efficient FWI dataset loader using memory-mapped I/O to avoid high RAM and I/O pressure.

    Args:
        anno (str): Path to annotation file, each line should be "data_path\tlabel_path".
        sample_ratio (int): Downsample ratio for seismic data along time axis.
        file_size (int): Number of samples in each .npy file.
        transform_data: Transformations applied to input data.
        transform_label: Transformations applied to target labels.
    """
    def __init__(self, anno, sample_ratio=1, file_size=500,
                 transform_data=None, transform_label=None):
        if not os.path.exists(anno):
            raise FileNotFoundError(f'Annotation file {anno} does not exist.')

        self.sample_ratio = sample_ratio
        self.transform_data = transform_data
        self.transform_label = transform_label
        self.file_size = file_size

        self.entries = []  # Each entry is (data_path, label_path)
        with open(anno, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.entries.append((parts[0], parts[1]))
                elif len(parts) == 1:
                    self.entries.append((parts[0], None))
                else:
                    raise ValueError(f"Invalid annotation line: {line}")

        self.data_handles = {}
        self.label_handles = {}

    def _load_memmap(self, path, key, is_label=False):
        handle_dict = self.label_handles if is_label else self.data_handles
        if key not in handle_dict:
            mmap = np.load(path, mmap_mode='r')
            if not is_label and self.sample_ratio > 1:
                mmap = mmap[:, :, ::self.sample_ratio, :]
            handle_dict[key] = mmap
        return handle_dict[key]

    def __len__(self):
        return len(self.entries) * self.file_size

    def __getitem__(self, idx):
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size
        data_path, label_path = self.entries[batch_idx]

        # .copy() to allow transform modification
        data = self._load_memmap(data_path, data_path)[sample_idx].copy()
        label = self._load_memmap(label_path, label_path, is_label=True)[sample_idx].copy() if label_path else None

        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_label and label is not None:
            label = self.transform_label(label)

        # # Add this for debug
        # if idx < 5:
        #     print(f"[DEBUG] idx={idx}, data.shape={data.shape}, data.dtype={data.dtype}")
        #     if label is not None:
        #         print(f"[DEBUG] label.shape={label.shape}, label.dtype={label.dtype}, label.sum={label.sum()}")
        #     else:
        #         print(f"[DEBUG] label is None")

        return (
            torch.tensor(data, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32) if label is not None else torch.tensor([])
        )



if __name__ == '__main__':
    transform_data = Compose([
        T.LogTransform(k=1),
        T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(2000, 6000)
    ])
    dataset = FWIDataset(
        anno='relevant_files/temp.txt',
        sample_ratio=1,
        file_size=1,
        transform_data=transform_data,
        transform_label=transform_label
    )
    data, label = dataset[0]
    print(data.shape)
    print(label.shape if label is not None else "No label")
