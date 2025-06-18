import argparse
import os

import hydra
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
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
    data_module = UTKFaceDataModule(cfg)
    
    model = ResNetBackbone(
        pretrained=False,
        task="classification",
        multitask=False,
    )

    training_module = ClassificationModule(cfg, model)
    logger = get_logger(cfg)
    
    checkpoint = ModelCheckpoint(monitor="train_loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=cfg.training.num_epochs,
        logger=logger,
        callbacks=[checkpoint],
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
        ["--task", "multitask"]
    ) 
    if args.task == "contrastive":
        main_contrastive()
    elif args.task == "multitask":
        main_multitask()
    else:
        raise ValueError(
            f"Unknown task: {args.task}. Use 'contrastive' or 'multitask'."
        )
