from torch.utils.data import Dataset
import numpy as np
import os
import torch
import rasterio


class WorCapDataset(Dataset):
    def __init__(self, T10_dir, T20_dir, mask_dir, transform=None):
        self.T10_dir = T10_dir
        self.T20_dir = T20_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.ids = sorted([
            f.split('_')[-1].replace('.tif', '')
            for f in os.listdir(T10_dir)
            if f.startswith('recorte_') and
               os.path.isfile(os.path.join(T20_dir, f)) and
               os.path.isfile(os.path.join(mask_dir, f))
        ])

    def __len__(self):
        return len(self.ids)

    def read_image(self, path):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)
            img = np.nan_to_num(img, nan=0.0)
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)
        return torch.tensor(img, dtype=torch.float32)

    def read_mask(self, path):
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.float32)
            mask = np.nan_to_num(mask, nan=0.0)
            mask = np.where(mask > 0, 1.0, 0.0)
        return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        fname = f"recorte_{id_}.tif"
        T10_path = os.path.join(self.T10_dir, fname)
        T20_path = os.path.join(self.T20_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        t1 = self.read_image(T10_path)
        t2 = self.read_image(T20_path)
        mask = self.read_mask(mask_path)

        if self.transform:
            t1 = self.transform(t1)
            t2 = self.transform(t2)
            mask = self.transform(mask)

        T = torch.cat([t1, t2], dim=0)
        
        
        return T, mask