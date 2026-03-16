import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models

# --- CBAM & Attention Modules ---

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

# CBAM 的 Spatial Attention 模組改為 Residual Spatial Attention (RSA)，以擴大感受野，解決手電筒效應
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
        # 產生 0~1 之間的注意力權重
        attn = self.sigmoid(self.conv1(x_cat))
        # 殘差加強：1+attn 範圍為 1~2，確保不會有區域被完全抹除
        return x * (1 + attn)

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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- Pooling & Linear Layers ---

class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim=2048, output_dim=8192):
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

        sketch1 = torch.zeros(B, self.output_dim, H * W, dtype=torch.float32, device=x.device)
        sketch1.scatter_add_(1, self.h1.view(1, C, 1).expand(B, C, H * W), x1)
        sketch2 = torch.zeros(B, self.output_dim, H * W, dtype=torch.float32, device=x.device)
        sketch2.scatter_add_(1, self.h2.view(1, C, 1).expand(B, C, H * W), x2)

        fft1 = torch.fft.fft(sketch1, dim=1)
        fft2 = torch.fft.fft(sketch2, dim=1)
        cbp = torch.fft.ifft(fft1 * fft2, dim=1).real
        cbp = cbp.sum(dim=-1)
        cbp = torch.sign(cbp) * torch.sqrt(torch.abs(cbp) + 1e-5)
        cbp = F.normalize(cbp, p=2, dim=1)
        return cbp.to(x.dtype)

class GeM(nn.Module):
    """
    Exp 21: 調降 p 值至 3.0
    增加池化層的平均性，有助於辨識依賴整體花紋的物種。
    """
    def __init__(self, p=3.0, eps=1e-6):
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

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    def forward(self, x):
        return torch.mm(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=0))

# --- Main Model ---

class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()

        # Backbone: ResNet152
        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
        self.backbone_l1_l3 = nn.Sequential(*list(resnet.children())[:7])
        self.backbone_l4 = nn.Sequential(*list(resnet.children())[7:8])

        # --- Layer 3 Processing (Texture/Detail) ---
        self.se_l3 = SEBlock(in_channels=1024, reduction=16)
        self.reduce3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.gem = GeM(p=3.0) # Exp 21: p=3.0

        # --- Layer 4 Processing (Semantic/Global) ---
        # 加入 RSA 以擴張視野，解決手電筒效應
        self.rsa = ResidualSpatialAttention(kernel_size=7)
        
        self.output_dim = 4096
        self.cbp = CompactBilinearPooling(input_dim=2048, output_dim=self.output_dim)

        self.fc_cbp = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(p=0.5)
        )

        # --- Fusion & Classifier ---
        self.embedding = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Dropout(p=0.5)
        )
        self.classifier = NormedLinear(512, num_classes)

    def forward(self, x):
        # Layer 3 Pipeline
        f3 = self.backbone_l1_l3(x)
        f3_att = self.se_l3(f3)
        f3_reduced = self.reduce3(f3_att)
        p3 = self.gem(f3_reduced).flatten(1)       # [B, 512]

        # Layer 4 Pipeline
        f4_raw = self.backbone_l4(f3)
        # 套用空間注意力 RSA
        f4_rsa = self.rsa(f4_raw)
        
        p4_cbp = self.cbp(f4_rsa)                  # [B, 2048]
        p4 = self.fc_cbp(p4_cbp)                   # [B, 512]

        # Fusion
        fused = torch.cat([p3, p4], dim=1)         # [B, 1024]
        embeddings = self.embedding(fused)
        return self.classifier(embeddings)

    def check_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        return total_params < 100_000_000