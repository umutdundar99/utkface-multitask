import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, task:str="contrastive", multitask:bool=False):
        super().__init__()
        r = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(r.children())[:-1])
        self.out_dim = r.fc.in_features
        self.multitask = multitask
        self.task = task
        if self.task == "classification":
            self.classify_head = nn.Sequential(
                    nn.Linear(self.out_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )
            if multitask == True:

                self.segmenter = nn.Sequential(
                    nn.Linear(self.out_dim, 256*7*7),  
                    nn.ReLU(),
                    nn.Unflatten(1, (256, 7, 7)),  
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 28x28
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 56x56
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 112x112
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   
                )
           
                
        
    def forward(self, x):
        x = self.encoder(x)
        if self.task == "contrastive":
            return x.view(x.size(0), -1)
        elif self.task == "classification":
            if self.multitask:
                segment, classify = self.segmenter(x), self.classify_head(x.view(x.size(0), -1))
                return segment, classify
            else:
                return self.classify_head(x.view(x.size(0), -1))
