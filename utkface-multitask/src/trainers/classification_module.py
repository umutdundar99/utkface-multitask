import torch
import torch.nn as nn
import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from src.models.backbone import ResNetBackbone

class ClassificationModule(L.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.multitask = cfg.data.multitask 
        self.lr = cfg.training.lr
        self.max_epochs = cfg.training.max_epochs

        self.model = model

        self.classification_loss = nn.BCEWithLogitsLoss()
        if self.multitask:
            self.segmentation_loss = nn.BCEWithLogitsLoss()

        
        train_metrics = {
            'class_acc': BinaryAccuracy(),
            'class_f1':  BinaryF1Score()
        }
        if self.multitask:
            train_metrics.update({
                'seg_acc': BinaryAccuracy(),
                'seg_f1':  BinaryF1Score()
            })
        self.train_metrics = MetricCollection(train_metrics, prefix='train/')

        
        val_metrics = {
            'class_acc': BinaryAccuracy(),
            'class_f1':  BinaryF1Score()
        }
        if self.multitask:
            val_metrics.update({
                'seg_acc': BinaryAccuracy(),
                'seg_f1':  BinaryF1Score()
            })
        self.val_metrics = MetricCollection(val_metrics, prefix='val/')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.multitask:
            x, y, mask = batch
            seg_logits, class_logits = self(x)
            loss = (
                self.classification_loss(class_logits.squeeze(), y.float())
              + self.segmentation_loss(seg_logits, mask)
            )

           
            class_pred = torch.sigmoid(class_logits.squeeze())
            seg_pred   = torch.sigmoid(seg_logits)

            
            metrics = self.train_metrics(
                {'class_acc': class_pred,
                 'class_f1':  class_pred,
                 'seg_acc':   seg_pred,
                 'seg_f1':    seg_pred},
                {'class_acc': y.int(),
                 'class_f1':  y.int(),
                 'seg_acc':   mask.int(),
                 'seg_f1':    mask.int()}
            )
        else:
            x, y = batch
            class_logits = self(x)
            loss = self.classification_loss(class_logits.squeeze(), y.float())

            class_pred = torch.sigmoid(class_logits.squeeze())
            metrics = self.train_metrics(
                {'class_acc': class_pred,
                 'class_f1':  class_pred},
                {'class_acc': y.int(),
                 'class_f1':  y.int()}
            )

        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        if self.multitask:
            x, y, mask = batch
            seg_logits, class_logits = self(x)
            loss = (
                self.classification_loss(class_logits.squeeze(), y.float())
              + self.segmentation_loss(seg_logits, mask)
            )
            class_pred = torch.sigmoid(class_logits.squeeze())
            seg_pred   = torch.sigmoid(seg_logits)
            metrics = self.val_metrics(
                {'class_acc': class_pred,
                 'class_f1':  class_pred,
                 'seg_acc':   seg_pred,
                 'seg_f1':    seg_pred},
                {'class_acc': y.int(),
                 'class_f1':  y.int(),
                 'seg_acc':   mask.int(),
                 'seg_f1':    mask.int()}
            )
        else:
            x, y = batch
            class_logits = self(x)
            loss = self.classification_loss(class_logits.squeeze(), y.float())
            class_pred = torch.sigmoid(class_logits.squeeze())
            metrics = self.val_metrics(
                {'class_acc': class_pred,
                 'class_f1':  class_pred},
                {'class_acc': y.int(),
                 'class_f1':  y.int()}
            )

        self.log("val/loss", loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )
        return [optimizer], [scheduler]
