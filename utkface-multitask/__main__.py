import argparse
import os

import hydra
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from src.data.dataset import UTKFaceDataModule
from src.trainers.contrastive_module import ContrastiveModule
from src.utils.logger import get_logger


@hydra.main(config_path="./configs", config_name="contrastive.yaml", version_base=None)
def main_contrastive(cfg: DictConfig):
    dm = UTKFaceDataModule(cfg)
    model = ContrastiveModule(cfg)
    logger = get_logger(cfg)
    checkpoint = ModelCheckpoint(monitor="train_loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        callbacks=[checkpoint],
        precision=32,
        log_every_n_steps=10,
        val_check_interval=50,
    )
    trainer.fit(model, datamodule=dm)


@hydra.main(config_path="./configs", config_name="multitask.yaml", version_base=None)
def main_multitask(cfg: DictConfig):
    dm = UTKFaceDataModule(cfg)
    model = ContrastiveModule(cfg)
    logger = get_logger(cfg)
    checkpoint = ModelCheckpoint(monitor="train_loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        callbacks=[checkpoint],
        precision=32,
        default_root_dir=os.getcwd(),
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(
        description="Train a contrastive model on UTKFace dataset"
    )
    parser.add_argument("--task", type=str, default="contrastive", help="Task to run")

    args = parser.parse_args(
        ["--task", "contrastive"]
    )  # Default to contrastive for testing
    if args.task == "contrastive":
        main_contrastive()
    elif args.task == "multitask":
        main_multitask()
    else:
        raise ValueError(
            f"Unknown task: {args.task}. Use 'contrastive' or 'multitask'."
        )
