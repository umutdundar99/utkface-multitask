import os
import random
from copy import deepcopy

import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentations import get_contrastive_augmentations


class UTKFaceContrastiveDataset(Dataset):
    img_size = (256, 256)

    def __init__(self, cfg, split):
        self.cfg = deepcopy(cfg)
        self.cfg["data"]["root_dir"] = os.path.join(self.cfg["data"]["root_dir"], split)
        self.df = pd.DataFrame(columns=["image", "age"])

        self.span = self.cfg.data.max_age - self.cfg.data.min_age + 1
        data_cfg = self.cfg.data
        samp_cfg = self.cfg.sampling

        ages, files = [], []
        for fname in os.listdir(data_cfg.root_dir):
            if fname.endswith("_image.png"):
                age = int(fname.split("_")[0])
                if data_cfg.min_age <= age <= data_cfg.max_age:
                    ages.append(age)
                    files.append(fname)

        if samp_cfg.oversample and split == "train":
            counts = {age: ages.count(age) for age in set(ages)}
            max_count = int(max(counts.values()))
            aug = []
            for age, f in zip(ages, files):
                n_rep = max_count // counts[age]
                aug += [f] * n_rep
            files = aug
            ages = [int(f.split("_")[0]) for f in files]

        self.root = data_cfg.root_dir
        self.items = list(zip(files, ages))
        self.groups = samp_cfg.groups
        self.transform = get_contrastive_augmentations(data_cfg.img_size)

        # compute group_size once
        self.span = data_cfg.max_age - data_cfg.min_age + 1
        self.group_size = self.span // self.groups

        # build index by group
        self.by_group = {}
        # self.debug = {f"{int(g)}": [] for g in range(self.groups)}

        for idx, (_, age) in enumerate(self.items):
            grp = (age - data_cfg.min_age) // self.group_size
            # self.debug[f"{int(grp)}"].append(age)

            self.df = pd.concat(
                [self.df, pd.DataFrame({"image": [files[idx]], "age": [age]})],
                ignore_index=True,
            )
            self.by_group.setdefault(grp, []).append(idx)

        print(f"Number of items in {split} set: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def read_image(self, fname):
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image {fname} could not be read.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def preprocess_image(self, image):
        image = cv2.resize(image, self.img_size)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        return image

    def __getitem__(self, idx):
        fname, age = self.items[idx]
        raw = self.read_image(os.path.join(self.root, fname))
        anchor = self.preprocess_image(self.transform(image=raw)["image"])

        # positive sample
        grp = (age - self.cfg.data.min_age) // self.group_size
        # choose pos_idx from the same age, from df
        pos_idxs = self.df[self.df["age"] == age].index.tolist()
        pos_idxs = [i for i in pos_idxs if i != idx]
        if not pos_idxs:
            raise ValueError(f"No positive samples found for age {age} in group {grp}.")
        pos_idx = random.choice(pos_idxs)
        pos_fname, _ = self.items[pos_idx]

        pos_raw = self.read_image(os.path.join(self.root, pos_fname))
        pos = self.preprocess_image(self.transform(image=pos_raw)["image"])

        # one negative per other group
        negs, neg_age_diffs = [], []
        for g, idxs in self.by_group.items():
            if g == grp:
                continue
            n = random.choice(idxs)
            fn, fn_age = self.items[n]
            neg_raw = self.read_image(os.path.join(self.root, fn))
            negs.append(self.preprocess_image(self.transform(image=neg_raw)["image"]))
            neg_age_diffs.append(abs(age - fn_age))

        negs = [neg.clone().detach() for neg in negs]
        negs = torch.stack(negs)
        # normalize neg age diffs by self.span
        neg_age_diffs = torch.tensor(neg_age_diffs, dtype=torch.float32) / self.span
        return anchor, pos, negs, age, neg_age_diffs


class UTKFaceDataModule(L.LightningDataModule):
    def __init__(self, cfg, num_workers=12):
        super().__init__()
        self.train_dataset = UTKFaceContrastiveDataset(cfg, split="train")
        self.val_dataset = UTKFaceContrastiveDataset(cfg, split="val")

        self.batch_size = cfg["training"]["batch_size"]
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
