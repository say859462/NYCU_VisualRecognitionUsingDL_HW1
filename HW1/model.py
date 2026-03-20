import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 保留 GeM (因為它對 FGVC 的特徵收束有幫助，且不干擾卷積)
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

# 保留 NormedLinear 以配合 LDAMLoss
class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        return torch.mm(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=0))

class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()

        # =========================================================
        # 直接使用官方 ResNeXt (此處以 101 為例，您也可改回 50)
        # =========================================================
        weights = models.ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None
        backbone = models.resnext101_32x8d(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # =========================================================
        # 極簡分類頭 (拔除所有 Attention 與 CBP)
        # =========================================================
        self.gem = GeM(p=3.0)
        
        # Layer 4 的輸出維度是 2048，直接降維到 512 後進行分類
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(0.4)
        )
        self.classifier = NormedLinear(512, num_classes)

    def forward(self, x, return_attn=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 2048, H, W]

        # 簡單純粹的池化與分類
        p = self.gem(x).flatten(1)
        embed = self.embedding(p)
        logits = self.classifier(embed)

        if self.training:
            # 為了相容原本的寫法，返回單一路徑的變數
            return logits, embed
        else:
            return logits

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"📊 Pure ResNeXt Status: {total/1e6:.2f}M params")
        return total < 100_000_000