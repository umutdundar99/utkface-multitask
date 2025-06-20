import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from utkface_multitask.src.models.backbone import ResNetBackbone


def distance_weighted_contrastive_loss(
    anchor_z, positive_z, negative_z, neg_age_diffs, temperature=0.1, sigma=0.5
):
    anchor_z = F.normalize(anchor_z, dim=1)
    positive_z = F.normalize(positive_z, dim=1)
    negative_z = F.normalize(negative_z, dim=2)

    pos_sim = torch.sum(anchor_z * positive_z, dim=1) / temperature
    pos_exp = torch.exp(pos_sim)

    neg_sim = (
        torch.bmm(anchor_z.unsqueeze(1), negative_z.transpose(1, 2)).squeeze(1)
        / temperature
    )
    neg_exp = torch.exp(neg_sim)

    neg_weights = torch.exp(neg_age_diffs / sigma)
    weighted_neg_exp = neg_exp * neg_weights

    loss = -torch.log(pos_exp / (pos_exp + weighted_neg_exp.sum(dim=1)))

    return loss.mean()


class ContrastiveModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()

        tcfg = cfg["training"]
        self.encoder = ResNetBackbone(pretrained=True)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 512), nn.ReLU(), nn.Linear(512, 128)
        )
        self.lr = tcfg["lr"]
        self.num_epochs = tcfg["num_epochs"]
        self.temperature = tcfg["temperature"]
        self.sigma = tcfg.get("sigma", 0.5)

    def training_step(self, batch, _):
        anchor, pos, negs, _, neg_age_diffs = batch

        h_anchor = self.encoder(anchor)
        h_pos = self.encoder(pos)

        batch_size, num_negs = negs.shape[0], negs.shape[1]
        negs_flat = negs.view(-1, *negs.shape[2:])
        h_negs_flat = self.encoder(negs_flat)

        z_anchor = self.projector(h_anchor)
        z_pos = self.projector(h_pos)
        z_negs = self.projector(h_negs_flat).view(batch_size, num_negs, -1)

        loss = distance_weighted_contrastive_loss(
            z_anchor, z_pos, z_negs, neg_age_diffs, self.temperature, self.sigma
        )

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, _):
        anchor, pos, negs, _, neg_age_diffs = batch

        h_anchor = self.encoder(anchor)
        h_pos = self.encoder(pos)

        batch_size, num_negs = negs.shape[0], negs.shape[1]
        negs_flat = negs.view(-1, *negs.shape[2:])
        h_negs_flat = self.encoder(negs_flat)

        z_anchor = self.projector(h_anchor)
        z_pos = self.projector(h_pos)
        z_negs = self.projector(h_negs_flat).view(batch_size, num_negs, -1)

        loss = distance_weighted_contrastive_loss(
            z_anchor, z_pos, z_negs, neg_age_diffs, self.temperature, self.sigma
        )

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.num_epochs)
        return [opt], [sch]
