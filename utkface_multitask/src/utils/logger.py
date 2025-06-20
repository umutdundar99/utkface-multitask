from lightning.pytorch.loggers import WandbLogger


def get_logger(cfg: str):
    cfg = cfg["logging"]["wandb"]
    return WandbLogger(
        project=cfg["project"],
        name=cfg["name"],
        offline=cfg["offline"],
        log_model=cfg["log_model"],
        save_dir=cfg["save_dir"],
        group=cfg.get("group", None),
    )
