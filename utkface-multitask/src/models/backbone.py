import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, task:str="contrastive", multitask:bool=False, num_classes:int=5):
        super().__init__()
        r = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(r.children())[:-1]) if task == "contrastive" else nn.Sequential(*list(r.children())[:-2])
        self.out_dim = r.fc.in_features
        self.multitask = multitask
        self.task = task
        if self.task == "classification":
            self.classify_head = nn.Sequential(
                    nn.Linear(self.out_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.50),
                    nn.Linear(256, num_classes)
                )
            if multitask == True:

                self.segmenter = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 28x28 -> 56x56
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 56x56 -> 112x112
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),     # 112x112 -> 224x224
                    
                )
                        
    def forward(self, x):
        x = self.encoder(x)
        if self.task == "contrastive":
            return x.view(x.size(0), -1)
        elif self.task == "classification":
            if self.multitask:
                x_cls = nn.AdaptiveAvgPool2d((1, 1))(x)
                x_cls = x_cls.view(x_cls.size(0), -1)
                segment, classify = self.segmenter(x), self.classify_head(x_cls)
                return segment, classify
            else:
                x = nn.AdaptiveAvgPool2d((1, 1))(x)
                return self.classify_head(x.view(x.size(0), -1))
