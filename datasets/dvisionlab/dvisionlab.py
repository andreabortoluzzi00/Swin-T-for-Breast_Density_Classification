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
# Unified DVisionLab dataset file
#
# Purpose:
# - Provide a clean, reusable dataset implementation for DVisionLab
# - Support both single-view and multi-view (CC + MLO) training
# - Handle patient-level splitting to avoid data leakage
# - Handle class imbalance via weighted sampling
# ---------------------------------------------------------------------------




def set_seed(seed: int = 5042):
    """Set random seed for reproducibility (Python + torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------- split helper ---------------------------------------

def split_train_val(csv_path, out_dir, group_col="patient_ID", test_size=0.3, seed=42):

    """
    Perform a group-wise train/validation split using patient_ID.

    This ensures that all images from the same patient appear
    either in train or validation, never in both (no data leakage).
    """

    df = pd.read_csv(csv_path)
    gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df[group_col]))
    train_df, val_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    train_csv, val_csv = os.path.join(out_dir, "train_split.csv"), os.path.join(out_dir, "val_split.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print("Class distribution (train):\n", train_df['rounded_label'].value_counts())
    print("Class distribution (val):\n", val_df['rounded_label'].value_counts())
    return train_csv, val_csv


# --------------------- base dataset ---------------------------------------

class _BaseDVisionLabDataset(Dataset):

    """
    Base dataset class containing shared logic:
    - image loading
    - normalization
    - resizing
    - optional data augmentation
    """

    def __init__(self, df: pd.DataFrame, images_root: str, img_size: int = 224, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.img_size = img_size
        self.augment = augment

        # Basic transforms
        self.base_transform = Compose([
            ScaleIntensity(),
            Resize((self.img_size, self.img_size)),
        ])

        if self.augment:
            self.aug_transform = Compose([
                RandFlip(prob=0.6, spatial_axis=0),
                RandFlip(prob=0.6, spatial_axis=1),
                RandRotate(range_x=10, prob=0.6, keep_size=True, mode='bilinear'),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.6, keep_size=True, mode='bilinear'),
            ])
        else:
            self.aug_transform = None

    def _load_image_tensor(self, rel_path: str):
        """Load image given relative path (from CSV) and return torch tensor shape (1,H,W).

        If file missing, returns a zero tensor so DataLoader keeps stable batch sizes.
        """
        img_path = os.path.join(self.images_root, rel_path)
        if not os.path.exists(img_path):
            
            print(f"Warning: missing file {img_path}")
            return torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        img = Image.open(img_path).convert('I')
        img_np = np.array(img, dtype=np.float32)
        # normalize
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_t = torch.from_numpy(img_np).unsqueeze(0).float()
        img_t = self.base_transform(img_t)
        if self.aug_transform is not None:
            img_t = self.aug_transform(img_t)
        return img_t


# --------------------- single-view dataset --------------------------------

class DVisionLabSingleViewDataset(_BaseDVisionLabDataset):
    """Single-view dataset: each row in CSV is one sample."""

    def __init__(self, csv_path=None, df: pd.DataFrame = None, images_root=None, img_size=224, augment=False):
        if df is None:
            assert csv_path is not None and images_root is not None
            df = pd.read_csv(csv_path)
        super().__init__(df, images_root, img_size, augment)

        # expected columns: file_path, rounded_label, patient_ID, view_position, image_laterality
        required = ['file_path', 'rounded_label', 'patient_ID', 'view_position']
        for c in required:
            if c not in self.df.columns:
                raise ValueError(f"Missing required column '{c}' in CSV")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image_tensor(row['file_path'])
        label = int(row['rounded_label'])
        meta = {
            'patient_id': row.get('patient_ID'),
            'view': row.get('view_position'),
            'laterality': row.get('image_laterality') if 'image_laterality' in row.index else None,
            'file_path': row['file_path']
        }
        return img, label, meta


# --------------------- multi-view dataset ---------------------------------

class DVisionLabMultiViewDataset(_BaseDVisionLabDataset):

    """
    Multi-view dataset:
    each sample is a (CC, MLO) pair for a given patient, study, and laterality.

    By default, only complete CC+MLO pairs are kept.
    """

    def __init__(self, csv_path=None, df: pd.DataFrame = None, images_root=None, img_size=224, augment=False,
                 keep_incomplete=False):
        if df is None:
            assert csv_path is not None and images_root is not None
            df = pd.read_csv(csv_path)
        super().__init__(df, images_root, img_size, augment)

        required = ['file_path', 'rounded_label', 'patient_ID', 'view_position', 'image_laterality', 'study_date', 'study_ID']
        for c in required:
            if c not in self.df.columns:
                raise ValueError(f"Missing required column '{c}' in CSV")

        self.keep_incomplete = keep_incomplete
        # group by patient, then by study (study_date + study_ID)
        self.grouped = self.df.groupby('patient_ID')
        self.pairs, self.labels, self.meta = self._generate_pairs()

    def _generate_pairs(self):

        """
        Generate all valid (CC, MLO) pairs per patient, study, and laterality.
        """

        pairs = []
        labels = []
        metas = []
        for patient_id, group in self.grouped:
            # group further by study
            for (_, _), study in group.groupby(['study_date', 'study_ID']):
                for lat in ['L', 'R']:
                    lat_group = study[study['image_laterality'] == lat]
                    cc = lat_group[lat_group['view_position'] == 'CC']
                    mlo = lat_group[lat_group['view_position'] == 'MLO']

                    # if both exist, make all combinations (in case multiple images per view)
                    if (len(cc) > 0) and (len(mlo) > 0):
                        for cc_row, mlo_row in product(cc.to_dict('records'), mlo.to_dict('records')):
                            pairs.append((cc_row['file_path'], mlo_row['file_path']))
                            labels.append(int(study['rounded_label'].iloc[0]))
                            metas.append({'patient_id': patient_id, 'laterality': lat, 'study_ID': study['study_ID'].iloc[0]})

                    else:
                        # handle incomplete cases
                        if self.keep_incomplete:
                            if len(cc) > 0:
                                for cc_row in cc.to_dict('records'):
                                    pairs.append((cc_row['file_path'], None))
                                    labels.append(int(study['rounded_label'].iloc[0]))
                                    metas.append({'patient_id': patient_id, 'laterality': lat, 'study_ID': study['study_ID'].iloc[0]})
                            if len(mlo) > 0:
                                for mlo_row in mlo.to_dict('records'):
                                    pairs.append((None, mlo_row['file_path']))
                                    labels.append(int(study['rounded_label'].iloc[0]))
                                    metas.append({'patient_id': patient_id, 'laterality': lat, 'study_ID': study['study_ID'].iloc[0]})
                        # otherwise skip
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
        # sanity checks
        for i, ((cc, mlo), lbl, meta) in enumerate(zip(self.pairs, self.labels, self.meta)):
            if cc is not None and mlo is not None:
                assert cc != mlo, f"Same file used for CC and MLO at pair {i}"


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

    """
    Build a WeightedRandomSampler to counter class imbalance.
    Each class is weighted inversely proportional to its frequency.
    """

    counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True)


# --------------------- main helper ---------------------------------------

def get_dvisionlab_dataloader(csv_path, images_root, out_dir, view_mode='multi', batch_size=32, img_size=224,
                              augment=False, num_workers=0, test_size=0.3, seed=42, keep_incomplete=False,
                              use_weighted_sampler=True, view_type=None):
    """Create train/val dataloaders for DVisionLab. Use view_mode in {'single','multi'}.

    Returns: train_loader, val_loader, train_dataset, val_dataset
    """
    set_seed(seed)
    train_csv, val_csv = split_train_val(csv_path, out_dir, group_col='patient_ID', test_size=test_size, seed=seed)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)


    # Filter by view_type if single-view and view_type specified
    if view_mode == 'single' and view_type is not None:
        train_df = train_df[train_df['view_position'] == view_type].reset_index(drop=True)
        val_df = val_df[val_df['view_position'] == view_type].reset_index(drop=True)

    if view_mode == 'single':
        train_ds = DVisionLabSingleViewDataset(df=train_df, images_root=images_root, img_size=img_size, augment=augment)
        val_ds = DVisionLabSingleViewDataset(df=val_df, images_root=images_root, img_size=img_size, augment=False)
        collate = collate_single
    elif view_mode == 'multi':
        train_ds = DVisionLabMultiViewDataset(csv_path=train_csv, images_root=images_root, img_size=img_size, augment=augment, keep_incomplete=keep_incomplete)
        val_ds = DVisionLabMultiViewDataset(csv_path=val_csv, images_root=images_root, img_size=img_size, augment=False, keep_incomplete=keep_incomplete)
        collate = collate_multi
    else:
        raise ValueError("view_mode must be 'single' or 'multi'")

    if use_weighted_sampler :
        sampler = make_weighted_sampler_from_labels(train_ds.labels if hasattr(train_ds, 'labels') else train_ds.df['rounded_label'].tolist())
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, collate_fn=collate, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds, val_ds


# ------------------------------------------------------
if __name__ == '__main__':
    # Quick demo: adjust paths before running
    csv_path = 'thesis/datasets/dvisionlab/preprocessed_dataset.csv'
    images_root = 'thesis/datasets/dvisionlab/'
    out_dir = 'dataset_multi'

    train_loader, val_loader, train_ds, val_ds = get_dvisionlab_dataloader(
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

    # Inspect one batch
    if next(iter(train_loader), None) is not None:
        if 'multi' in 'multi':
            imgs_cc, imgs_mlo, labels, metas = next(iter(train_loader))
            print('batch shapes:', imgs_cc.shape, imgs_mlo.shape, labels.shape)
        else:
            imgs, labels, metas = next(iter(train_loader))
            print('batch shapes:', imgs.shape, labels.shape)

    print('Train class counts :', Counter([int(x) for x in (train_ds.labels if hasattr(train_ds, 'labels') else train_ds.df['rounded_label'].tolist())]))
    print('Val class counts :', Counter([int(x) for x in (val_ds.labels if hasattr(val_ds, 'labels') else val_ds.df['rounded_label'].tolist())]))

    print('Val length:', len(val_ds))
