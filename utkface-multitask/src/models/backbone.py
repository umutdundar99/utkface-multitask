import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(r.children())[:-1])
        self.out_dim = r.fc.in_features

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
