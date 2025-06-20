import torch
import torch.nn as nn
import lightning as L
from torchmetrics import MetricCollection , JaccardIndex
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision
import torch.nn.functional as F
from torchmetrics.classification import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

class MulticlassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        gamma: Focusing parameter. Larger gamma → focus more on hard examples.
        alpha: Class balancing weights. Can be a scalar or a list/tensor of shape (num_classes,)
        reduction: 'mean', 'sum' or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (N, C) unnormalized scores from the model (before softmax)
        targets: (N,) integer class labels (0 ≤ target < C)
        """
        num_classes = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)        # (N, C)
        probs = torch.exp(log_probs)                    # (N, C)
        #targets_onehot = F.one_hot(targets, num_classes=num_classes)  # (N, C)

        # Calculate p_t and log(p_t) for the true class
        pt = (probs * targets).sum(dim=1)        # (N,)
        log_pt = (log_probs * targets).sum(dim=1)

        # Apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha_t = torch.tensor(self.alpha, device=logits.device)[targets]
            else:
                alpha_t = self.alpha
        else:
            alpha_t = 1.0

        # Focal Loss formula
        loss = -alpha_t * (1 - pt) ** self.gamma * log_pt

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape (N,)

class ClassificationModule(L.LightningModule):
    num_classes = 6
    def __init__(self, cfg, model, multitask: bool = False):
        super().__init__()
        self.cfg = cfg
        self.multitask = multitask
        self.lr = cfg.training.lr
        self.max_epochs = cfg.training.num_epochs

        self.model = model
        # kaiming init for segmentation head (segmente)

        self.classification_loss = nn.CrossEntropyLoss()
        #self.classification_loss = MulticlassFocalLoss()
        if self.multitask:
            self.segmentation_loss = nn.CrossEntropyLoss()
            for m in self.model.segmenter.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        self.train_classification_metrics = MetricCollection(
            metrics=[
                Accuracy(task="multiclass", num_classes=self.num_classes),
                F1Score(task="multiclass", num_classes=self.num_classes),
                AUROC(task="multiclass", num_classes=self.num_classes),
                AveragePrecision(task="multiclass", num_classes=self.num_classes),
                
            ],
            prefix='train/classification/'
        )

        self.val_classification_metrics = MetricCollection(
            metrics=[
                Accuracy(task="multiclass", num_classes=self.num_classes),
                F1Score(task="multiclass", num_classes=self.num_classes),
                AUROC(task="multiclass", num_classes=self.num_classes),
                AveragePrecision(task="multiclass", num_classes=self.num_classes),
                ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
            ],
            prefix='val/classification/'
        )


        if self.multitask:
            self.train_segmentation_metrics = MetricCollection([JaccardIndex(task="binary", num_classes=2, threshold=0.5)], prefix='train/segmentation/')
            self.val_segmentation_metrics = MetricCollection([JaccardIndex(task="binary", num_classes=2, threshold=0.5)], prefix='val/segmentation/')
        
             

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.multitask:
            x, y, mask = batch
            seg_logits, class_logits = self(x)
            labels_ohe = nn.functional.one_hot(y, num_classes=self.num_classes).float()
            loss = self.classification_loss(class_logits, labels_ohe) + 0.50 * self.segmentation_loss(seg_logits, mask)
        
            
            cls_metrics = self.train_classification_metrics(
                class_logits, y.int())
            seg_metrics = self.train_segmentation_metrics(
                seg_logits, mask.int()
            )
            metrics = {**cls_metrics, **seg_metrics}
            self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)

        else:
            x, y = batch
            class_logits = self(x)
            labels_ohe = nn.functional.one_hot(y, num_classes=self.num_classes).float()
            loss = self.classification_loss(class_logits, labels_ohe)
            
            cls_metrics = self.train_classification_metrics(
                class_logits, y.int())
            
            self.log_dict(cls_metrics, prog_bar=True, on_step=True, on_epoch=True)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        if self.multitask: 
            x, y, mask = batch
            seg_logits, class_logits = self(x)
            labels_ohe = nn.functional.one_hot(y, num_classes=self.num_classes).float()
            loss = (
                self.classification_loss(class_logits, labels_ohe) 
              + self.segmentation_loss(seg_logits, mask)
            )
           
            seg_pred   = torch.sigmoid(seg_logits)
            self.val_classification_metrics.update(
                class_logits, y.int())
            self.val_segmentation_metrics.update(
                seg_pred, mask.int()
            )
            
        else:
            x, y = batch
            class_logits = self(x)
            labels_ohe = nn.functional.one_hot(y, num_classes=self.num_classes).float()
            loss = self.classification_loss(class_logits, labels_ohe)
            
            self.val_classification_metrics.update(
                class_logits, y.int())    
            
        self.log("val/loss", loss, prog_bar=True)
        

    def on_validation_epoch_end(self):
        cls_metrics = self.val_classification_metrics.compute()
        seg_metrics = None
        if self.multitask:
            seg_metrics = self.val_segmentation_metrics.compute()
        
        confmat = cls_metrics.pop("val/classification/MulticlassConfusionMatrix")  # shape: (C, C)
        if not self.multitask:
            self.log_dict(cls_metrics, prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.log_dict({**cls_metrics, **seg_metrics}, prog_bar=True, on_step=False, on_epoch=True)


       
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confmat.cpu().numpy(), annot=True, fmt="d", cmap="Blues",
                    xticklabels=[str(i+1) for i in range(self.num_classes)],
                    yticklabels=[str(i+1) for i in range(self.num_classes)],
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Validation Confusion Matrix")
        plt.tight_layout()

       
        if self.logger:
            self.logger.experiment.log({"val_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        

        self.val_classification_metrics.reset()
        if self.multitask:
            self.val_segmentation_metrics.reset()
            
    def on_train_epoch_end(self):
        self.train_classification_metrics.reset()
        if self.multitask:
            self.train_segmentation_metrics.reset()
            

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # encoder has 1e-5, others 1e-3
        # encoder_params = [p for n, p in self.model.named_parameters() if "encoder" in n]
        # other_params = [p for n, p in self.model.named_parameters() if "encoder" not in n]
        # optimizer = torch.optim.AdamW(
        #     [
        #         {"params": encoder_params, "lr": 1e-6},
        #         {"params": other_params, "lr": self.lr},
        #     ],
        #     weight_decay=1e-2,
        # )
        
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        return [optimizer], [scheduler]
