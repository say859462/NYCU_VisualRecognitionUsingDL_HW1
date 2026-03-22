import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)


class SKConv(nn.Module):
    """Selective Kernel Convolution: 動態變焦，自動決定感受野大小"""

    def __init__(self, features, reduction=16):
        super(SKConv, self).__init__()
        # 分支 1: 3x3 卷積 (專注微小特徵)
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        # 分支 2: 擴張卷積 (等效 5x5 感受野，專注全圖輪廓)
        self.conv2 = nn.Conv2d(features, features, 3,
                               padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.bn2 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)

        d = max(int(features / reduction), 32)
        self.fc1 = nn.Conv2d(features, d, 1, bias=False)
        self.bn_fc1 = nn.BatchNorm2d(d)
        self.fc2 = nn.Conv2d(d, features * 2, 1, bias=False)

    def forward(self, x):
        U1 = self.relu(self.bn1(self.conv1(x)))
        U2 = self.relu(self.bn2(self.conv2(x)))
        U = U1 + U2

        # 空間池化獲取全局資訊
        S = U.mean([-2, -1], keepdim=True)
        Z = self.relu(self.bn_fc1(self.fc1(S)))
        A = self.fc2(Z).view(x.size(0), 2, x.size(1), 1, 1)

        # Softmax 注意力分配權重
        A = F.softmax(A, dim=1)
        V = A[:, 0] * U1 + A[:, 1] * U2
        return V


class AFF(nn.Module):
    """Attentive Feature Fusion: 跨層動態融合，解決梯度污染"""

    def __init__(self, channels=512):
        super(AFF, self).__init__()
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial_fusion = x + y
        xl = self.local_att(initial_fusion)
        xg = self.global_att(initial_fusion)
        weight = self.sigmoid(xl + xg)
        return 2 * x * weight + 2 * y * (1 - weight)


class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super(ImageClassificationModel, self).__init__()
        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 1x1 降維 (保持低參數量，送入 SKConv)
        self.conv1x1_l3 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.conv1x1_l4 = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))

        # 裝配 SK-Conv (動態變焦) 與 AFF (融合)
        self.sk3 = SKConv(512)
        self.sk4 = SKConv(512)
        self.aff = AFF(channels=512)
        self.gem = GeM(p=3.0)

        # Soft-PMG：為 L3, L4 與融合層建立獨立表頭，強制淺層學習紋理
        self.emb_l3 = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(
            512), nn.PReLU(), nn.Dropout(p=0.4))
        self.cls_l3 = nn.Linear(512, num_classes)

        self.emb_l4 = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(
            512), nn.PReLU(), nn.Dropout(p=0.4))
        self.cls_l4 = nn.Linear(512, num_classes)

        self.emb_fused = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.PReLU(), nn.Dropout(p=0.4))
        self.cls_fused = nn.Linear(512, num_classes)

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # Layer 3 處理
        f3_orig = self.layer3(x)
        f3_c = self.conv1x1_l3(f3_orig)
        f3_sk = self.sk3(f3_c)

        # Layer 4 處理
        f4_orig = self.layer4(f3_orig)
        f4_c = self.conv1x1_l4(f4_orig)
        f4_sk = self.sk4(f4_c)

        # 跨層注意力融合
        f4_up = F.interpolate(
            f4_sk, size=f3_sk.shape[2:], mode='bilinear', align_corners=False)
        f_fused = self.aff(f3_sk, f4_up)

        # GeM 池化
        p3 = self.gem(f3_sk).flatten(1)
        p4 = self.gem(f4_sk).flatten(1)
        p_fused = self.gem(f_fused).flatten(1)

        # Embedding 與分類
        e3 = self.emb_l3(p3)
        e4 = self.emb_l4(p4)
        e_fused = self.emb_fused(p_fused)

        out3 = self.cls_l3(e3)
        out4 = self.cls_l4(e4)
        out_fused = self.cls_fused(e_fused)

        if self.training:
            # 訓練模式：返回多路徑輸出，用於 Soft-PMG 與 SupCon Loss
            return out3, out4, out_fused, e_fused
        else:
            # 評估模式：自動屏蔽複雜度，僅返回主分類器 (完美相容原本的 val/test 腳本)
            return out_fused

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"📊 ResNet152-SK-PMG (Exp 46) Trainable Params: {total/1e6:.2f}M")
        return total < 100_000_000
