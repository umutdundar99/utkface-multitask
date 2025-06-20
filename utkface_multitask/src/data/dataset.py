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

age_groups = {
    "0": list(range(0,5)),
    "1": list(range(5,15)),
    "2": list(range(15, 45)),  
    "3": list(range(45, 65)),
    "4": list(range(65,100))
}



class UTKFaceContrastiveDataset(Dataset):
    span = 100  

    def __init__(self, cfg, split):
        self.cfg = deepcopy(cfg)
        self.img_size = (cfg.data.img_size, cfg.data.img_size)
        self.root = os.path.join(self.cfg["data"]["root_dir"], split)
        self.df = pd.DataFrame(columns=["image", "age", "group"])
        
        self.transform = get_contrastive_augmentations(cfg.data.img_size)
        
        for image in os.listdir(self.root):
            if image.endswith("_mask.png"):
                continue
            age = int(image.split("_")[0])
            if age < 0 or age > 100:
                continue
            for group_name, age_range in age_groups.items():
                if age in age_range:
                    self.df = pd.concat(
                        [self.df, pd.DataFrame({"image": [image], "age": [age], "group": [group_name]})],
                        ignore_index=True
                    )
        # upsample to highest value to all groups
        max_group_size = self.df["group"].value_counts().max()
        self.df = self.df.groupby("group").apply(
            lambda x: x.sample(max_group_size, replace=True)
        ).reset_index(drop=True)
       
    def __len__(self):
        return len(self.df)

    def read_image(self, fname):
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image {fname} could not be read.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def preprocess_image(self, image):
        #image = cv2.resize(image, self.img_size)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        return image

    def __getitem__(self, idx):
        fname, age, grp = self.df.iloc[idx]
        raw = self.read_image(os.path.join(self.root, fname))
        anchor = self.preprocess_image(self.transform(image=raw)["image"])


        pos_age= self.df[self.df["age"] == age]
        pos_fname = random.choice(pos_age["image"].values)
        pos_raw = self.read_image(os.path.join(self.root, pos_fname))
        pos = self.preprocess_image(self.transform(image=pos_raw)["image"])

        # one negative per other group
        negs, neg_age_diffs = [], []
        for g in self.df["group"].unique():
            if g == grp:
                continue
            idxs = self.df[self.df["group"] == g].index.tolist()
            n = random.choice(idxs)
            fn, fn_age = self.df.iloc[n][["image", "age"]]
            neg_raw = self.read_image(os.path.join(self.root, fn))
            negs.append(self.preprocess_image(self.transform(image=neg_raw)["image"]))
            neg_age_diffs.append(abs(age - fn_age))

        negs = [neg.clone().detach() for neg in negs]
        negs = torch.stack(negs)
      
        neg_age_diffs = torch.tensor(neg_age_diffs, dtype=torch.float32) / (self.span//2) 
        return anchor, pos, negs, age, neg_age_diffs
    
class UTKFaceMultitaskDataset(Dataset):
   
    def __init__(self, cfg, split):
        self.cfg = deepcopy(cfg)
        self.task = self.cfg.training.multitask
        self.img_size = (cfg.data.img_size, cfg.data.img_size)
        self.root = os.path.join(self.cfg["data"]["root_dir"], split)
        self.transform = get_contrastive_augmentations(cfg.data.img_size)
        self.all_frames = os.listdir(self.root)
        
        self.all_frames = [img for img in self.all_frames if int(img.split("_")[0]) < 100]
        self.images = sorted([img for img in self.all_frames if img.endswith("_image.png")])
        
        self.multitask = cfg.training.multitask
        if self.multitask:
            self.masks = sorted([img for img in self.all_frames if img.endswith("_mask.png")])

    def __len__(self):
        return len(self.images)

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
        image_path = self.images[idx]
        age = int(image_path.split("_")[0])
        
        for group_name , age_range in age_groups.items():
            if age in age_range:
                group = torch.tensor(int(group_name), dtype=torch.long)
                break
        
        image = self.read_image(os.path.join(self.root, image_path))
        if self.multitask:
            mask_path = self.masks[idx]
            mask = self.read_image(os.path.join(self.root, mask_path))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 

           
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  
            mask = augmented["mask"]   

            mask = mask.unsqueeze(0)  
            mask = mask / 255.0

            return image, group, mask

            
        else: 
            image = self.transform(image=image)["image"]
            return image, group
        

class UTKFaceDataModule(L.LightningDataModule):
    def __init__(self, 
                 cfg:dict,
                 num_workers:int=12,
                 task:str= "contrastive"):
        
        super().__init__()
        if task == "contrastive":
            self.train_dataset = UTKFaceContrastiveDataset(cfg, split="train")
            self.val_dataset = UTKFaceContrastiveDataset(cfg, split="val")
        elif task == "classification":
            self.train_dataset = UTKFaceMultitaskDataset(cfg, split="train")
            self.val_dataset = UTKFaceMultitaskDataset(cfg, split="val")

        self.batch_size = cfg["training"]["batch_size"]
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory= True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
