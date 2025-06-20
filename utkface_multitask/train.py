# train.py (yeniden yazılmış)
import os

import lightning as pl
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig

from utkface_multitask.src.data.dataset import UTKFaceDataModule
from utkface_multitask.src.models.backbone import ResNetBackbone
from utkface_multitask.src.trainers.classification_module import ClassificationModule
from utkface_multitask.src.trainers.contrastive_module import ContrastiveModule
from utkface_multitask.src.utils.logger import get_logger


def train(task: str = "contrastive"):
    config_path = os.path.join(os.path.dirname(__file__), "configs")
    cfg_name = "contrastive.yaml" if task == "contrastive" else "multitask.yaml"

    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg: DictConfig = compose(config_name=cfg_name)

    if task == "contrastive":
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

    elif task == "classification":
        data_module = UTKFaceDataModule(cfg, task="classification")
        model = ResNetBackbone(
            pretrained=True,
            task="classification",
            multitask=cfg.training.multitask,
        )
        if cfg.training.checkpoint:
            state_dict = torch.load(cfg.training.checkpoint, map_location="cpu")[
                "state_dict"
            ]
            state_dict = {
                (k[8:] if k.startswith("encoder.") else k): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict, strict=False)
        training_module = ClassificationModule(
            cfg, model, multitask=cfg.training.multitask
        )
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

    else:
        raise ValueError(f"Unknown task: {task}")
