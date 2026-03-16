import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models

# CBMA module
# Reference: https://arxiv.org/pdf/1807.06521.pdf


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# 原本的 SpatialAttention 改為 ResidualSpatialAttention 以擴大感受野，解決手電筒效應


class ResidualSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ResidualSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv1(x_cat))
        return x * (1 + attn)

# SEBlock (Squeeze-and-Excitation Block)
# Reference: https://arxiv.org/pdf/1709.01507.pdf


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

# Compact Bilinear Pooling Layer
# Reference: https://arxiv.org/pdf/1506.02310.pdf


class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim=512, output_dim=4096):
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim

        self.register_buffer('h1', torch.randint(0, output_dim, (input_dim,)))
        self.register_buffer('s1', torch.randint(0, 2, (input_dim,)) * 2 - 1)
        self.register_buffer('h2', torch.randint(0, output_dim, (input_dim,)))
        self.register_buffer('s2', torch.randint(0, 2, (input_dim,)) * 2 - 1)

    def forward(self, x):
        B, C, H, W = x.size()

        x_flat = x.view(B, C, -1).float()

        x1 = x_flat * self.s1.view(1, C, 1).float()
        x2 = x_flat * self.s2.view(1, C, 1).float()

        sketch1 = torch.zeros(B, self.output_dim, H * W,
                              dtype=torch.float32, device=x.device)
        sketch1.scatter_add_(1, self.h1.view(1, C, 1).expand(B, C, H * W), x1)

        sketch2 = torch.zeros(B, self.output_dim, H * W,
                              dtype=torch.float32, device=x.device)
        sketch2.scatter_add_(1, self.h2.view(1, C, 1).expand(B, C, H * W), x2)

        fft1 = torch.fft.fft(sketch1, dim=1)
        fft2 = torch.fft.fft(sketch2, dim=1)
        fft_product = fft1 * fft2

        cbp = torch.fft.ifft(fft_product, dim=1).real
        cbp = cbp.sum(dim=-1)
        cbp = torch.sign(cbp) * torch.sqrt(torch.abs(cbp) + 1e-5)
        cbp = F.normalize(cbp, p=2, dim=1)
        return cbp.to(x.dtype)


# GeM Pooling Layer(Generalized Mean Pooling)
# Reference: https://arxiv.org/pdf/1711.02512.pdf
class GeM(nn.Module):
    def __init__(self, p=2.5, eps=1e-6):  # 調降至 2.5 擴大範圍
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = eps

    def forward(self, x):
        x_float = x.float()
        p_float = self.p.float()

        out = F.avg_pool2d(
            x_float.clamp(min=self.eps).pow(p_float),
            (x_float.size(-2), x_float.size(-1))
        ).pow(1.0 / p_float)

        return out.to(x.dtype)

# 新增 NormedLinear (搭配 LDAM Loss 使用)


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

        # Backbone model : ResNet
        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
        self.backbone_l1_l3 = nn.Sequential(*list(resnet.children())[:7])
        self.backbone_l4 = nn.Sequential(*list(resnet.children())[7:8])

        # --- Layer 3 Processing (維持不變) ---
        self.se_l3 = SEBlock(in_channels=1024, reduction=16)
        self.reduce3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.gem3 = GeM(p=2.5)

        # --- Layer 4 Processing ---
        self.se_l4 = SEBlock(in_channels=2048, reduction=16)
        self.rsa = ResidualSpatialAttention(kernel_size=7)
        self.reduce4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # ⭐ 新增：Layer 4 的 Spatial Branch (GeM)
        self.gem4 = GeM(p=2.5)

        # ⭐ 修改：調降 CBP 維度 4096 -> 2048，防止過擬合
        self.output_dim = 2048
        self.cbp = CompactBilinearPooling(
            input_dim=512, output_dim=self.output_dim)

        # ⭐ 修改：漸進式特徵融合的輸入維度
        # 輸入 = p3 (512) + p4_gem (512) + p4_cbp (2048) = 3072
        self.embedding = nn.Sequential(
            nn.Linear(512 + 512 + 2048, 768),
            nn.BatchNorm1d(768),
            nn.PReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(p=0.4)
        )
        self.classifier = NormedLinear(512, num_classes)

    def forward(self, x, return_attn=False):
        # Layer 3 pipeline
        f3_raw = self.backbone_l1_l3(x)
        f3_att = self.se_l3(f3_raw)
        p3 = self.gem3(self.reduce3(f3_att)).flatten(1)          # [B, 512]

        # Layer 4 CBP pipeline
        f4_raw = self.backbone_l4(f3_raw)
        f4_se = self.se_l4(f4_raw)

        # 計算 RSA
        avg_out = torch.mean(f4_se, dim=1, keepdim=True)
        max_out, _ = torch.max(f4_se, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.rsa.sigmoid(self.rsa.conv1(x_cat))
        f4_att = f4_se * (1 + spatial_attn)

        # 降維
        # [B, 512, 14, 14]
        f4_reduced = self.reduce4(f4_att)

        # ⭐ 分支 1：Spatial Branch (GeM) - 保留空間資訊供 Grad-CAM 使用
        p4_gem = self.gem4(f4_reduced).flatten(1)                # [B, 512]

        # ⭐ 分支 2：Feature Interaction Branch (CBP) - 抓取細粒度二階特徵
        p4_cbp = self.cbp(f4_reduced)                            # [B, 2048]

        # Fusion: Concat(p3, p4_gem, p4_cbp)
        fused = torch.cat([p3, p4_gem, p4_cbp], dim=1)           # [B, 3072]
        embeddings = self.embedding(fused)
        logits = self.classifier(embeddings)

        if return_attn:
            # ⭐ 核心改動：改用 Activation Map 作為 Attention Crop 的依據
            activation_map = torch.mean(f4_se, dim=1, keepdim=True)
            return logits, activation_map

        return logits

    def check_parameters(self):
        # Check the total number of trainable parameters in the model
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

        if total_params < 100_000_000:
            print("Model size is within the 100M limit.")
            return True
        else:
            print("Warning: Model size exceeds the 100M limit!")

        return False
