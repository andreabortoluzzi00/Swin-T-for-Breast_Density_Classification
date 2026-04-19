import os
import random
from itertools import product
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit

from monai.transforms import Compose, ScaleIntensity, Resize, RandFlip, RandRotate, RandZoom

# ---------------------------------------------------------------------------
# Unified INbreast dataset file
# - contains both single-view and multi-view dataset classes
# - provides a get_inbreast_dataloader(...) helper to return train/val loaders
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------- split helper ---------------------------------------
def split_train_val(csv_path, out_dir, group_col="patient_key", test_size=0.3, seed=42):
    """Group-wise train/val split. Groups are patient keys."""
    df = pd.read_csv(csv_path)
    gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df[group_col]))
    train_df, val_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    train_csv, val_csv = os.path.join(out_dir, "train_split.csv"), os.path.join(out_dir, "val_split.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print("Class distribution (train):\n", train_df['acr'].value_counts())
    print("Class distribution (val):\n", val_df['acr'].value_counts())
    return train_csv, val_csv


# --------------------- base dataset ---------------------------------------
class _BaseINbreastDataset(Dataset):
    """Base dataset with common utilities."""

    def __init__(self, df: pd.DataFrame, images_root: str, img_size: int = 224, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.img_size = img_size
        self.augment = augment

        self.base_transform = Compose([
            ScaleIntensity(),
            Resize((self.img_size, self.img_size)),
        ])

        if self.augment:
            self.aug_transform = Compose([
                RandFlip(prob=0.9, spatial_axis=0),
                RandFlip(prob=0.9, spatial_axis=1),
                RandRotate(range_x=10, prob=0.9, keep_size=True, mode='bilinear'),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.9, keep_size=True, mode='bilinear'),
            ])
        else:
            self.aug_transform = None

    def _load_image_tensor(self, rel_path: str):
        img_path = os.path.join(self.images_root, rel_path)
        if not os.path.exists(img_path):
            print(f"Warning: missing file {img_path}")
            return torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        img = Image.open(img_path).convert('I')
        img_np = np.array(img, dtype=np.float32)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_t = torch.from_numpy(img_np).unsqueeze(0).float()
        img_t = self.base_transform(img_t)
        if self.aug_transform is not None:
            img_t = self.aug_transform(img_t)
        return img_t


# --------------------- single-view dataset --------------------------------
class INbreastSingleViewDataset(_BaseINbreastDataset):
    """Single-view dataset."""

    def __init__(self, csv_path=None, df: pd.DataFrame = None, images_root=None, img_size=224, augment=False):
        if df is None:
            assert csv_path is not None and images_root is not None
            df = pd.read_csv(csv_path)
        super().__init__(df, images_root, img_size, augment)

        required = ['file', 'acr', 'patient_key', 'view', 'laterality']
        for c in required:
            if c not in self.df.columns:
                raise ValueError(f"Missing required column '{c}' in CSV")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image_tensor(row['file'])
        label = int(row['acr'])
        meta = {
            'patient_id': row['patient_key'],
            'view': row['view'],
            'laterality': row['laterality'],
            'file_path': row['file']
        }
        return img, label, meta


# --------------------- multi-view dataset ---------------------------------
class INbreastMultiViewDataset(_BaseINbreastDataset):
    """Multi-view dataset returning (CC, MLO) pairs per patient and laterality."""

    def __init__(self, csv_path=None, df: pd.DataFrame = None, images_root=None, img_size=224, augment=False, keep_incomplete=False):
        if df is None:
            assert csv_path is not None and images_root is not None
            df = pd.read_csv(csv_path)
        super().__init__(df, images_root, img_size, augment)

        required = ['file', 'acr', 'patient_key', 'view', 'laterality']
        for c in required:
            if c not in self.df.columns:
                raise ValueError(f"Missing required column '{c}' in CSV")

        self.keep_incomplete = keep_incomplete
        self.grouped = self.df.groupby('patient_key')
        self.pairs, self.labels, self.meta = self._generate_pairs()

    def _generate_pairs(self):
        pairs = []
        labels = []
        metas = []
        for patient_id, group in self.grouped:
            for lat in ['L', 'R']:
                lat_group = group[group['laterality'] == lat]
                cc = lat_group[lat_group['view'] == 'CC']
                mlo = lat_group[lat_group['view'] == 'MLO']

                if len(cc) > 0 and len(mlo) > 0:
                    for cc_row, mlo_row in product(cc.to_dict('records'), mlo.to_dict('records')):
                        pairs.append((cc_row['file'], mlo_row['file']))
                        labels.append(int(cc_row['acr']))
                        metas.append({'patient_id': patient_id, 'laterality': lat})
                else:
                    if self.keep_incomplete:
                        if len(cc) > 0:
                            for cc_row in cc.to_dict('records'):
                                pairs.append((cc_row['file'], None))
                                labels.append(int(cc_row['acr']))
                                metas.append({'patient_id': patient_id, 'laterality': lat})
                        if len(mlo) > 0:
                            for mlo_row in mlo.to_dict('records'):
                                pairs.append((None, mlo_row['file']))
                                labels.append(int(mlo_row['acr']))
                                metas.append({'patient_id': patient_id, 'laterality': lat})
        return pairs, labels, metas

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cc_path, mlo_path = self.pairs[idx]
        label = int(self.labels[idx])
        img_cc = self._load_image_tensor(cc_path) if cc_path else torch.zeros(1, self.img_size, self.img_size)
        img_mlo = self._load_image_tensor(mlo_path) if mlo_path else torch.zeros(1, self.img_size, self.img_size)
        meta = self.meta[idx]
        return torch.stack([img_cc, img_mlo], dim=0), label, meta

    def check_pairs(self):
        for i, ((cc, mlo), lbl, meta) in enumerate(zip(self.pairs, self.labels, self.meta)):
            if cc and mlo:
                assert cc != mlo, f"Same file for CC and MLO at pair {i}"


# --------------------- collate functions ----------------------------------
def collate_multi(batch):
    imgs_cc = torch.stack([item[0][0] for item in batch], dim=0)
    imgs_mlo = torch.stack([item[0][1] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metas = [item[2] for item in batch]
    return imgs_cc, imgs_mlo, labels, metas

def collate_single(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metas = [item[2] for item in batch]
    return imgs, labels, metas


# --------------------- sampler helper ------------------------------------
def make_weighted_sampler_from_labels(labels):
    counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# --------------------- main helper ---------------------------------------
def get_inbreast_dataloader(csv_path, images_root, out_dir, view_mode='multi', batch_size=32, img_size=224,
                            augment=False, num_workers=0, test_size=0.3, seed=42, keep_incomplete=False,
                            use_weighted_sampler=True):
    """Create train/val dataloaders for INbreast."""
    set_seed(seed)
    train_csv, val_csv = split_train_val(csv_path, out_dir, group_col='patient_key', test_size=test_size, seed=seed)

    if view_mode == 'single':
        train_ds = INbreastSingleViewDataset(csv_path=train_csv, images_root=images_root, img_size=img_size, augment=augment)
        val_ds = INbreastSingleViewDataset(csv_path=val_csv, images_root=images_root, img_size=img_size, augment=False)
        collate = collate_single
    elif view_mode == 'multi':
        train_ds = INbreastMultiViewDataset(csv_path=train_csv, images_root=images_root, img_size=img_size, augment=augment, keep_incomplete=keep_incomplete)
        val_ds = INbreastMultiViewDataset(csv_path=val_csv, images_root=images_root, img_size=img_size, augment=False, keep_incomplete=keep_incomplete)
        collate = collate_multi
    else:
        raise ValueError("view_mode must be 'single' or 'multi'")

    if use_weighted_sampler:
        labels = train_ds.labels if hasattr(train_ds, 'labels') else train_ds.df['acr'].tolist()
        sampler = make_weighted_sampler_from_labels(labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                                  num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds, val_ds


# --------------------- demo / quick test ---------------------------------
if __name__ == '__main__':
    csv_path = './inbreast_preprocessed.csv'
    images_root = './inbreast_cropped'
    out_dir = './dataset_inbreast'

    train_loader, val_loader, train_ds, val_ds = get_inbreast_dataloader(
        csv_path=csv_path,
        images_root=images_root,
        out_dir=out_dir,
        view_mode='multi',
        batch_size=8,
        img_size=224,
        augment=True,
        num_workers=0,
        test_size=0.3,
        seed=42,
        keep_incomplete=False,
        use_weighted_sampler=True,
    )

    if view_mode := 'multi':
        imgs_cc, imgs_mlo, labels, metas = next(iter(train_loader))
        print('batch shapes:', imgs_cc.shape, imgs_mlo.shape, labels.shape)
    else:
        imgs, labels, metas = next(iter(train_loader))
        print('batch shapes:', imgs.shape, labels.shape)

    print('Train class counts (approx):', Counter([int(x) for x in (train_ds.labels if hasattr(train_ds, 'labels') else train_ds.df['acr'].tolist())]))
    print('Val length:', len(val_ds))
