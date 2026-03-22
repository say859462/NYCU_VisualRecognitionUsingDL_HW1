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
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=False)
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
        S = U.mean([-2, -1], keepdim=True)
        Z = self.relu(self.bn_fc1(self.fc1(S)))
        A = self.fc2(Z).view(x.size(0), 2, x.size(1), 1, 1)
        A = F.softmax(A, dim=1)
        return A[:, 0] * U1 + A[:, 1] * U2


class DropBlock2D(nn.Module):
    """DropBlock: 強制擴展注意力視野，防止熱力圖坍塌在局部特徵"""

    def __init__(self, drop_prob=0.15, block_size=5):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        gamma = self.drop_prob / (self.block_size ** 2) * \
            (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        mask = (torch.rand(x.shape[0], *x.shape[2:],
                device=x.device) < gamma).float()
        mask = F.max_pool2d(mask, self.block_size, stride=1,
                            padding=self.block_size // 2)
        mask = 1. - mask
        return x * mask.unsqueeze(1) * (mask.numel() / mask.sum())


class CompactBilinearPooling(nn.Module):
    """CBP: 捕捉斑紋與輪廓同時出現的二階統計特徵 (透過 Count Sketch 加速)"""

    def __init__(self, input_dim=512, output_dim=1024):
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim
        # 預先生成隨機投影矩陣 (Count Sketch)
        self.register_buffer(
            'sketch1', self.generate_sketch_matrix(input_dim, output_dim))
        self.register_buffer(
            'sketch2', self.generate_sketch_matrix(input_dim, output_dim))

    def generate_sketch_matrix(self, input_dim, output_dim):
        rand_h = torch.randint(0, output_dim, (input_dim,))
        rand_s = torch.randint(0, 2, (input_dim,)).float() * 2 - 1
        sketch = torch.zeros(output_dim, input_dim)
        sketch[rand_h, torch.arange(input_dim)] = rand_s
        return sketch

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1_flat = x1.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        x2_flat = x2.view(B, C, -1).permute(0, 2, 1)

        proj1 = F.linear(x1_flat, self.sketch1)  # [B, HW, out_dim]
        proj2 = F.linear(x2_flat, self.sketch2)

        # ⭐ 核心修正：FFT 不支援 BFloat16，在此暫時轉為 float32
        orig_dtype = proj1.dtype
        proj1_f32 = proj1.to(torch.float32)
        proj2_f32 = proj2.to(torch.float32)

        # 透過 FFT 計算卷積等效於外積的 Count Sketch
        fft1 = torch.fft.fft(proj1_f32)
        fft2 = torch.fft.fft(proj2_f32)

        # 算完 IFFT 後，再轉回原本的 dtype (bfloat16) 以無縫銜接後續網路
        cbp_flat = torch.fft.ifft(fft1 * fft2).real.to(orig_dtype)

        # 將二階特徵還原為 2D 空間形狀，供 Grad-CAM 或 GeM 使用
        cbp_spatial = cbp_flat.permute(0, 2, 1).view(B, self.output_dim, H, W)
        return cbp_spatial


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

        self.conv1x1_l3 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.conv1x1_l4 = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))

        self.sk3 = SKConv(512)
        self.sk4 = SKConv(512)

        # ⭐ 加入 DropBlock 防止注意力坍塌
        self.dropblock = DropBlock2D(drop_prob=0.15, block_size=5)

        # ⭐ 替換 AFF 為 CBP，升級二階特徵 (輸出 1024D)
        self.cbp = CompactBilinearPooling(input_dim=512, output_dim=1024)

        self.gem = GeM(p=3.0)

        self.emb_l3 = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(
            512), nn.PReLU(), nn.Dropout(p=0.4))
        self.cls_l3 = nn.Linear(512, num_classes)

        self.emb_l4 = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(
            512), nn.PReLU(), nn.Dropout(p=0.4))
        self.cls_l4 = nn.Linear(512, num_classes)

        # 融合層降維漏斗 (1024D -> 512D)
        self.emb_fused = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.PReLU(), nn.Dropout(p=0.4))
        self.cls_fused = nn.Linear(512, num_classes)

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        f3_orig = self.layer3(x)
        f3_c = self.conv1x1_l3(f3_orig)
        f3_sk = self.sk3(f3_c)
        f3_sk = self.dropblock(f3_sk)  # 實施特徵挖空

        f4_orig = self.layer4(f3_orig)
        f4_c = self.conv1x1_l4(f4_orig)
        f4_sk = self.sk4(f4_c)
        f4_sk = self.dropblock(f4_sk)

        f4_up = F.interpolate(
            f4_sk, size=f3_sk.shape[2:], mode='bilinear', align_corners=False)

        # ⭐ 二階外積融合
        f_fused = self.cbp(f3_sk, f4_up)

        p3 = self.gem(f3_sk).flatten(1)
        p4 = self.gem(f4_sk).flatten(1)
        p_fused = self.gem(f_fused).flatten(1)

        e3 = self.emb_l3(p3)
        e4 = self.emb_l4(p4)
        e_fused = self.emb_fused(p_fused)

        out3 = self.cls_l3(e3)
        out4 = self.cls_l4(e4)
        out_fused = self.cls_fused(e_fused)

        if self.training:
            return out3, out4, out_fused, e_fused
        else:
            return out_fused

    # ⭐ 新增：cRT 解耦訓練專用介面
    def freeze_features_for_crt(self):
        """凍結所有特徵提取器，僅保留分類頭 (Classifier Heads) 可訓練"""
        for param in self.parameters():
            param.requires_grad = False

        # 僅解凍最終的分類器
        for head in [self.cls_l3, self.cls_l4, self.cls_fused]:
            for param in head.parameters():
                param.requires_grad = True

    def get_classifier_parameters(self):
        return list(self.cls_l3.parameters()) + list(self.cls_l4.parameters()) + list(self.cls_fused.parameters())

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
            f"📊 ResNet152-CBP-DropBlock (Exp 47) Trainable Params: {total/1e6:.2f}M")
        return total < 100_000_000
