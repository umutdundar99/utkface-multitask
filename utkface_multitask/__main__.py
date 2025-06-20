import argparse
import os

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import DictConfig
from src.data.dataset import UTKFaceDataModule
from src.trainers.contrastive_module import ContrastiveModule
from src.trainers.classification_module import ClassificationModule
from src.utils.logger import get_logger
from src.models.backbone import ResNetBackbone

@hydra.main(config_path="./configs", config_name="contrastive.yaml", version_base=None)
def main_contrastive(cfg: DictConfig):
    dm = UTKFaceDataModule(cfg)
    train_module = ContrastiveModule(cfg)
    logger = get_logger(cfg)
    checkpoint = ModelCheckpoint(monitor="train_loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        logger=logger,
        callbacks=[checkpoint],
        precision=32,
        log_every_n_steps=10,
        val_check_interval=0.25,
    )
    trainer.fit(train_module, datamodule=dm)


@hydra.main(config_path="./configs", config_name="multitask.yaml", version_base=None)
def main_multitask(cfg: DictConfig):
    data_module = UTKFaceDataModule(cfg, task="classification")
    model = ResNetBackbone(
            pretrained=True,
            task="classification",
            multitask=cfg.training.multitask,
        )
    if cfg.training.checkpoint:
        state_dict = torch.load(cfg.training.checkpoint, map_location="cpu")["state_dict"]
        
        state_dict = {(k[8:] if k.startswith("encoder.") else k): v for k, v in state_dict.items()}
        unexpecteds= model.load_state_dict(state_dict, strict=False)
        if unexpecteds.missing_keys:
            print(f"Missing keys in loaded state dict: {unexpecteds.missing_keys}")
        if unexpecteds.unexpected_keys:
            print(f"Unexpected keys in loaded state dict: {unexpecteds.unexpected_keys}")
    else:
        print("No checkpoint provided, initializing model from scratch.")
        

    training_module = ClassificationModule(cfg, model, multitask=cfg.training.multitask)
    logger = get_logger(cfg)
    
    checkpoint = ModelCheckpoint(monitor="val/loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        logger=logger,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        precision=32,
        default_root_dir=os.getcwd(),
    )
    trainer.fit(training_module, data_module)



if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(
        description="Train a contrastive model on UTKFace dataset"
    )
    parser.add_argument("--task", type=str, default="contrastive", help="Task to run")
    parser.add_argument("--pretrained-weights", type=str, default=None, help="Path to pretrained weights")

    args = parser.parse_args(
        ["--task", "classification"]
    ) 
    if args.task == "contrastive":
        main_contrastive()
    elif args.task == "classification":
        main_multitask()
    else:
        raise ValueError(
            f"Unknown task: {args.task}. Use 'contrastive' or 'classification'."
        )
