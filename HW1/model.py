import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# --- 高階注意力機制：Coordinate Attention (CoordAtt) ---
class CoordAtt(nn.Module):
    """
    座標注意力機制：同時捕捉通道重要性與精確的空間位置資訊。
    能有效解決熱力圖破碎與偏移的問題。
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.PReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        
        # 1D 全域池化 (水平 + 垂直)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 座標資訊編碼
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        # 拆分回兩個維度
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 產生空間權重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_w * a_h

# --- 池化與線性層模組 ---
class GeM(nn.Module):
    def __init__(self, p=2.5, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

# --- 主模型架構：Res2Net101 + Coordinate Attention ---
class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()
        
        # 載入多尺度 Backbone
        backbone = timm.create_model('res2net101_26w_4s', pretrained=pretrained)
        
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.act1, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 🚀 Exp 43 核心優化：使用 CoordAtt 取代 SE/RSA
        # Layer 3 (1024 channels) 與 Layer 4 (2048 channels) 的特徵校準
        self.ca3 = CoordAtt(1024, 1024, reduction=32)
        self.ca4 = CoordAtt(2048, 2048, reduction=32)
        
        self.gem = GeM(p=2.5)

        # 強化版決策層
        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 1024),  # 擴張維度保留資訊
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(0.3)         # 0.3 Dropout 配合座標注意力
        )
        self.classifier = nn.Linear(1024, num_classes)

    def extract_features(self, x):
        # 基礎特徵流
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Layer 3 + 座標校準
        x = self.layer3(x)
        x = self.ca3(x)
        
        # Layer 4 + 座標校準 (取代 RSA)
        x = self.layer4(x)
        x = self.ca4(x)
        
        # 全域池化與資訊壓縮
        pool = self.gem(x).flatten(1)
        emb = self.bottleneck(pool)
        return emb

    def forward(self, x):
        emb = self.extract_features(x)
        logits = self.classifier(emb)
        return logits

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"📊 Res2Net101-CoordAtt Status: {total/1e6:.2f}M params")
        return total < 100_000_000