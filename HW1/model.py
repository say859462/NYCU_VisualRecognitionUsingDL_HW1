import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet152_Weights


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.tensor([p], dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(self.p, min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / p)
        return x.flatten(1)


class Res2Adapter(nn.Module):
    def __init__(self, channels: int, scale: int = 4, bottleneck_ratio: int = 4):
        super().__init__()
        assert scale >= 2
        inner_channels = channels // bottleneck_ratio
        assert inner_channels % scale == 0

        self.scale = scale
        self.split_channels = inner_channels // scale

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.split_channels,
                    self.split_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.split_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(scale - 1)
        ])

        self.expand = nn.Sequential(
            nn.Conv2d(inner_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.reduce(x)
        splits = torch.split(out, self.split_channels, dim=1)

        outputs = [splits[0]]
        for idx in range(1, self.scale):
            if idx == 1:
                y = self.scale_convs[idx - 1](splits[idx])
            else:
                y = self.scale_convs[idx - 1](splits[idx] + outputs[idx - 1])
            outputs.append(y)

        out = torch.cat(outputs, dim=1)
        out = self.expand(out)
        out = out + identity
        out = self.relu(out)
        return out


class SubCenterClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_subcenters: int = 3,
        scale: float = 16.0,
        learn_scale: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_subcenters = num_subcenters

        self.weight = nn.Parameter(torch.randn(num_classes, num_subcenters, in_features))
        if learn_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=2)
        logits_all = torch.einsum("bd,ckd->bck", x, w)
        logits_all = logits_all * self.scale.clamp(min=1.0, max=32.0)
        class_logits, _ = logits_all.max(dim=2)
        return class_logits, logits_all


class PMGHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        num_subcenters: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = SubCenterClassifier(
            in_features=embed_dim,
            num_classes=num_classes,
            num_subcenters=num_subcenters,
            scale=16.0,
            learn_scale=True,
        )

    def forward(self, x: torch.Tensor):
        embed = self.proj(x)
        logits, logits_all = self.classifier(embed)
        return logits, embed, logits_all
class RawEvidenceFusionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.20,
    ):
        super().__init__()
        stat_dim = 9
        support_dim = 3
        interaction_dim = embed_dim * 6
        in_dim = embed_dim * 3 + interaction_dim + num_classes * 3 + stat_dim + support_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _branch_stats(logits: torch.Tensor):
        prob = torch.softmax(logits, dim=1)
        top2_prob, top2_idx = torch.topk(prob, k=2, dim=1)
        top1 = top2_prob[:, 0:1]
        gap = top2_prob[:, 0:1] - top2_prob[:, 1:2]
        entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        pred = top2_idx[:, 0]
        return {
            "prob": prob,
            "top1": top1,
            "gap": gap,
            "entropy": entropy,
            "pred": pred,
        }

    def forward(
        self,
        global_embed: torch.Tensor,
        part2_embed: torch.Tensor,
        part4_embed: torch.Tensor,
        global_logits: torch.Tensor,
        part2_logits: torch.Tensor,
        part4_logits: torch.Tensor,
    ):
        g = self._branch_stats(global_logits)
        p2 = self._branch_stats(part2_logits)
        p4 = self._branch_stats(part4_logits)

        agreement_gp2 = (g["pred"] == p2["pred"]).float().unsqueeze(1)
        agreement_gp4 = (g["pred"] == p4["pred"]).float().unsqueeze(1)
        agreement_p24 = (p2["pred"] == p4["pred"]).float().unsqueeze(1)

        support_stats = torch.cat(
            [
                agreement_gp2,
                agreement_gp4,
                agreement_p24,
            ],
            dim=1,
        )

        branch_stats = torch.cat(
            [
                g["top1"], g["gap"], g["entropy"],
                p2["top1"], p2["gap"], p2["entropy"],
                p4["top1"], p4["gap"], p4["entropy"],
            ],
            dim=1,
        )

        interaction_feat = torch.cat(
            [
                global_embed * part2_embed,
                global_embed * part4_embed,
                part2_embed * part4_embed,
                torch.abs(global_embed - part2_embed),
                torch.abs(global_embed - part4_embed),
                torch.abs(part2_embed - part4_embed),
            ],
            dim=1,
        )

        fusion_input = torch.cat(
            [
                global_embed,
                part2_embed,
                part4_embed,
                interaction_feat,
                global_logits,
                part2_logits,
                part4_logits,
                branch_stats,
                support_stats,
            ],
            dim=1,
        )

        hidden = self.fusion_mlp(fusion_input)
        final_logits = self.classifier(hidden)

        return final_logits, {
            "fusion_hidden": hidden,
            "global_prob": g["prob"],
            "part2_prob": p2["prob"],
            "part4_prob": p4["prob"],
            "agreement_gp2": agreement_gp2,
            "agreement_gp4": agreement_gp4,
            "agreement_p24": agreement_p24,
            "support_stats": support_stats,
            "fusion_input": fusion_input,
        }

class ImageClassificationModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        pretrained: bool = True,
        num_subcenters: int = 3,
        embed_dim: int = 256,
        backbone_name: str = "resnet152_partial_res2net",
    ):
        super().__init__()
        if backbone_name != "resnet152_partial_res2net":
            raise ValueError("Only 'resnet152_partial_res2net' is supported.")
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        weights = ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet152(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.res2_layer3 = Res2Adapter(1024, scale=4, bottleneck_ratio=4)
        self.res2_layer4 = Res2Adapter(2048, scale=4, bottleneck_ratio=4)

        self.global_proj = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.part2_proj = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.part4_proj = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.gem = GeM(p=3.0, learn_p=True)
        self.pool_2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool_4_avg = nn.AdaptiveAvgPool2d((4, 4))
        self.pool_4_max = nn.AdaptiveMaxPool2d((4, 4))

        self.global_head = PMGHead(512, embed_dim, num_classes, num_subcenters=num_subcenters, dropout=0.2)
        self.part2_head = PMGHead(512 * 4, embed_dim, num_classes, num_subcenters=num_subcenters, dropout=0.2)
        self.part4_head = PMGHead(512 * 16, embed_dim, num_classes, num_subcenters=num_subcenters, dropout=0.2)
        self.concat_head = RawEvidenceFusionHead(embed_dim=embed_dim, num_classes=num_classes, hidden_dim=max(256, embed_dim * 2))

        self._freeze_shallow_layers()
        self._init_new_layers()

    def _freeze_shallow_layers(self):
        for module in [self.stem, self.layer1]:
            for param in module.parameters():
                param.requires_grad = False

    def _init_new_layers(self):
        for module in [self.res2_layer3, self.res2_layer4, self.global_proj, self.part2_proj, self.part4_proj, self.concat_head]:
            for submodule in module.modules():
                if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(submodule.weight, mode="fan_out", nonlinearity="relu")
                    if getattr(submodule, "bias", None) is not None:
                        nn.init.zeros_(submodule.bias)
                elif isinstance(submodule, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(submodule.weight)
                    nn.init.zeros_(submodule.bias)

    def check_parameters(self) -> bool:
        total = sum(param.numel() for param in self.parameters())
        print(f"Parameters: {total}")
        return total < 100_000_000

    def get_parameter_groups(self, lr_base: float):
        head_modules = [
            self.res2_layer3,
            self.res2_layer4,
            self.global_proj,
            self.part2_proj,
            self.part4_proj,
            self.gem,
            self.global_head,
            self.part2_head,
            self.part4_head,
            self.concat_head,
        ]
        head_params = []
        for module in head_modules:
            head_params.extend([param for param in module.parameters() if param.requires_grad])
        return [
            {"params": [p for p in self.layer2.parameters() if p.requires_grad], "lr": lr_base * 0.1},
            {"params": [p for p in self.layer3.parameters() if p.requires_grad], "lr": lr_base * 0.5},
            {"params": [p for p in self.layer4.parameters() if p.requires_grad], "lr": lr_base * 1.0},
            {"params": head_params, "lr": lr_base * 1.5},
        ]

    def forward_backbone(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat_l3 = self.layer3(x)
        feat_l3 = self.res2_layer3(feat_l3)
        feat_l4 = self.layer4(feat_l3)
        feat_l4 = self.res2_layer4(feat_l4)
        return feat_l3, feat_l4

    @staticmethod
    def _normalize_map(x: torch.Tensor):
        x = x - x.amin(dim=(2, 3), keepdim=True)
        x = x / (x.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return x

    def build_attention_map(self, global_map: torch.Tensor, part2_map: torch.Tensor):
        global_sal = self._normalize_map(global_map.mean(dim=1, keepdim=True))
        part2_sal = self._normalize_map(part2_map.mean(dim=1, keepdim=True))
        return self._normalize_map(0.5 * global_sal + 0.5 * part2_sal)

    def _build_global_feature(self, global_map: torch.Tensor):
        return self.gem(global_map)

    def _build_part2_feature(self, part2_map: torch.Tensor):
        part2_grid = self.pool_2(part2_map)
        return part2_grid.flatten(1), part2_grid

    def _build_part4_feature(self, part4_map: torch.Tensor):
        part4_grid = 0.5 * (self.pool_4_avg(part4_map) + self.pool_4_max(part4_map))
        return part4_grid.flatten(1), part4_grid

    def forward_features(self, x: torch.Tensor):
        feat_l3, feat_l4 = self.forward_backbone(x)
        global_map = self.global_proj(feat_l4)
        part2_map = self.part2_proj(feat_l4)
        part4_map = self.part4_proj(feat_l3)
        attention_map = self.build_attention_map(global_map, part2_map)
        return {
            "feat_l3": feat_l3,
            "feat_l4": feat_l4,
            "global_map": global_map,
            "part2_map": part2_map,
            "part4_map": part4_map,
            "attention_map": attention_map,
        }

    def forward_pmg(self, x: torch.Tensor):
        feats = self.forward_features(x)
        global_feat = self._build_global_feature(feats["global_map"])
        part2_feat, part2_grid = self._build_part2_feature(feats["part2_map"])
        part4_feat, part4_grid = self._build_part4_feature(feats["part4_map"])

        global_logits, global_embed, global_logits_all = self.global_head(global_feat)
        part2_logits, part2_embed, part2_logits_all = self.part2_head(part2_feat)
        part4_logits, part4_embed, part4_logits_all = self.part4_head(part4_feat)
        concat_logits, fusion_aux = self.concat_head(
            global_embed=global_embed,
            part2_embed=part2_embed,
            part4_embed=part4_embed,
            global_logits=global_logits,
            part2_logits=part2_logits,
            part4_logits=part4_logits,
        )

        return {
            "global_logits": global_logits,
            "part2_logits": part2_logits,
            "part4_logits": part4_logits,
            "concat_logits": concat_logits,
            "global_embed": global_embed,
            "part2_embed": part2_embed,
            "part4_embed": part4_embed,
            "global_logits_all": global_logits_all,
            "part2_logits_all": part2_logits_all,
            "part4_logits_all": part4_logits_all,
            "part2_grid": part2_grid,
            "part4_grid": part4_grid,
            **fusion_aux,
            **feats,
        }

    def forward(self, x: torch.Tensor):
        return self.forward_pmg(x)["concat_logits"]
