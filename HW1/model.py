import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ==========================================
# 1. 手寫 ResNeXt 化區塊 (導入 Grouped Convolution)
# ==========================================


class ResNeXtifiedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32, base_width=4):
        super(ResNeXtifiedBottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3,
                               stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# ==========================================
# 2. FGVC 增強組件 (CBP, SE, RSA, GeM)
# ==========================================


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
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y


class ResidualSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ResidualSpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))
        return x * (1 + attn)


class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim=512, output_dim=2048):
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
        sketch1 = torch.zeros(B, self.output_dim, H*W, device=x.device).scatter_add_(
            1, self.h1.view(1, C, 1).expand(B, C, H*W), x1)
        sketch2 = torch.zeros(B, self.output_dim, H*W, device=x.device).scatter_add_(
            1, self.h2.view(1, C, 1).expand(B, C, H*W), x2)
        fft_product = torch.fft.fft(
            sketch1, dim=1) * torch.fft.fft(sketch2, dim=1)
        cbp = torch.fft.ifft(fft_product, dim=1).real.sum(dim=-1)
        cbp = torch.sign(cbp) * torch.sqrt(torch.abs(cbp) + 1e-5)
        return F.normalize(cbp, p=2, dim=1).to(x.dtype)


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        return torch.mm(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=0))


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

# ==========================================
# 3. 改裝版模型主體 (ResNet-152 骨架 + ResNeXt 區塊)
# ==========================================


class ImageClassificationModel(nn.Module):
    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super(ImageClassificationModel, self).__init__()
        resnet = models.resnet152(
            weights=models.ResNet152_Weights.DEFAULT if pretrained else None)
        self.inplanes = 64
        self.groups, self.base_width = 32, 4

        self.conv1, self.bn1, self.relu, self.maxpool = resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 8, stride=2)
        self.layer3 = self._make_layer(256, 36, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.backbone_l1_l3 = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3)
        self.backbone_l4 = self.layer4

        # FGVC Components
        self.se_l3 = SEBlock(1024)
        self.rsa_l3 = ResidualSpatialAttention()
        self.reduce3 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.gem3 = GeM()
        self.se_l4 = SEBlock(2048)
        self.rsa_l4 = ResidualSpatialAttention()
        self.reduce4 = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.gem4 = GeM()
        self.cbp = CompactBilinearPooling(512, 2048)

        self.embedding_cbp = nn.Sequential(
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.PReLU(), nn.Dropout(0.5))
        self.classifier_cbp = NormedLinear(512, num_classes)
        self.embedding_gem = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.PReLU(), nn.Dropout(0.5))
        self.classifier_gem = NormedLinear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(
                self.inplanes, planes * 4, 1, stride, bias=False), nn.BatchNorm2d(planes * 4))
        layers = [ResNeXtifiedBottleneck(
            self.inplanes, planes, stride, downsample, self.groups, self.base_width)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(ResNeXtifiedBottleneck(
                self.inplanes, planes, groups=self.groups, base_width=self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x, return_attn=False):

        f3 = self.backbone_l1_l3(x)
        p3 = self.gem3(self.reduce3(self.rsa_l3(self.se_l3(f3)))).flatten(1)
        f4_att = self.rsa_l4(self.se_l4(self.backbone_l4(f3)))
        f4_red = self.reduce4(f4_att)
        p4_gem, p4_cbp = self.gem4(f4_red).flatten(1), self.cbp(f4_red)

        l_cbp, l_gem = self.classifier_cbp(self.embedding_cbp(
            p4_cbp)), self.classifier_gem(self.embedding_gem(torch.cat([p3, p4_gem], 1)))

        if self.training:
            # 訓練模式回傳 4 或 5 個值 (對齊 train.py)
            if return_attn:
                return l_cbp, l_gem, self.embedding_cbp(p4_cbp), self.embedding_gem(torch.cat([p3, p4_gem], 1)), torch.mean(f4_att, 1, keepdim=True)
            return l_cbp, l_gem, self.embedding_cbp(p4_cbp), self.embedding_gem(torch.cat([p3, p4_gem], 1))
        else:
            # 推論模式回傳 1 或 2 個值 (對齊 val.py/test.py/gradcam_vis.py)
            l_ens = l_cbp * 0.8 + l_gem * 0.2
            if return_attn:
                return l_ens, torch.mean(f4_att, 1, keepdim=True)
            return l_ens

    def check_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Total Params: {total/1e6:.2f}M")
        return total < 100_000_000
