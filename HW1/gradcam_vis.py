import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import ImageClassificationModel


def _denorm_image(img_tensor):
    img = img_tensor.detach().cpu()
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def _overlay_heatmap_on_image(rgb_img, heatmap, alpha=0.45):
    """
    rgb_img: HWC float [0,1]
    heatmap: HW float [0,1]
    """
    cmap = plt.get_cmap("jet")
    heatmap_color = cmap(heatmap)[..., :3]
    overlay = (1 - alpha) * rgb_img + alpha * heatmap_color
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def _normalize_map(x):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def compute_cls_gradcam(model, input_tensor):
    """
    對當前 self-attention / CLS token 版本：
    以 cls_logits 的 top-1 類別作為 target，
    對 fused_map 做 Grad-CAM。
    """
    model.eval()

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # 對 fuse 的輸出做 hook
    handle_fwd = model.fuse.register_forward_hook(forward_hook)
    handle_bwd = model.fuse.register_full_backward_hook(backward_hook)

    try:
        input_tensor = input_tensor.requires_grad_(True)
        outputs = model.forward_pmg(input_tensor, return_attn=True)

        cls_logits = outputs["cls_logits"]
        pred_idx = cls_logits.argmax(dim=1).item()
        score = cls_logits[:, pred_idx].sum()

        model.zero_grad(set_to_none=True)
        score.backward()

        feat = activations[0]          # [1, C, H, W]
        grad = gradients[0]            # [1, C, H, W]

        weights = grad.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * feat).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = _normalize_map(cam)

        cls_probs = torch.softmax(cls_logits, dim=1)[0].detach().cpu()
        return cam, pred_idx, cls_probs, outputs
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def _fmt_pred(name, probs, pred_idx):
    return f"{name}:{pred_idx} ({probs[pred_idx].item():.2f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--val_dir", type=str, default="./Dataset/data/val")
    parser.add_argument("--num_samples_per_class", type=int, default=3)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str,
                        default="./Plot/Attention_Outputs/current_cls")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    model_path = args.model_path if args.model_path is not None else config["best_model_path"]

    model = ImageClassificationModel(
        num_classes=config["num_classes"],
        pretrained=False,
        num_subcenters=config.get("num_subcenters", 3),
        embed_dim=config.get("embed_dim", 256),
        cls_num_heads=config.get("cls_num_heads", 4),
        cls_attn_dropout=config.get("cls_attn_dropout", 0.1),
        cls_ffn_ratio=config.get("cls_ffn_ratio", 2.0),
        cls_block_dropout=config.get("cls_block_dropout", 0.1),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preprocess_geo = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
    ])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for class_id in tqdm(range(config["num_classes"]), desc="Classes Processed"):
        class_dir = os.path.join(args.val_dir, str(class_id))
        if not os.path.exists(class_dir):
            continue

        class_save_dir = os.path.join(args.save_dir, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)

        all_images = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if not all_images:
            continue

        sampled_image_paths = random.sample(
            all_images, min(args.num_samples_per_class, len(all_images))
        )

        for img_path in sampled_image_paths:
            raw_img = Image.open(img_path).convert("RGB")
            cropped_img = preprocess_geo(raw_img)
            input_tensor = preprocess_tensor(
                cropped_img).unsqueeze(0).to(device)
            rgb_img = np.asarray(cropped_img).astype(np.float32) / 255.0

            cam, cls_pred, cls_probs, outputs = compute_cls_gradcam(
                model, input_tensor)

            # 也拿舊 PMG heads 的預測做對照
            with torch.no_grad():
                global_probs = torch.softmax(
                    outputs["global_logits"], dim=1)[0].cpu()
                part2_probs = torch.softmax(
                    outputs["part2_logits"], dim=1)[0].cpu()
                part4_probs = torch.softmax(
                    outputs["part4_logits"], dim=1)[0].cpu()
                concat_probs = torch.softmax(
                    outputs["concat_logits"], dim=1)[0].cpu()

                global_pred = int(global_probs.argmax().item())
                part2_pred = int(part2_probs.argmax().item())
                part4_pred = int(part4_probs.argmax().item())
                concat_pred = int(concat_probs.argmax().item())

                # cls_attn_weights: [B, num_heads, T, T] 或 [B, T, T] 視 pytorch 版本而定
                attn = outputs["cls_attn_weights"]
                if attn.dim() == 4:
                    # [B, H, T, T]
                    attn_map = attn[0].mean(dim=0)  # [T, T]
                else:
                    # [B, T, T]
                    attn_map = attn[0]

                # token 順序: [CLS, global, part2, part4]
                cls_to_tokens = attn_map[0].detach().cpu().numpy()
                # 我們只看 CLS 對 branch tokens 的注意力，不看自己
                branch_attn = cls_to_tokens[1:4]
                branch_attn = branch_attn / (branch_attn.sum() + 1e-8)

            overlay = _overlay_heatmap_on_image(rgb_img, cam, alpha=0.45)

            fig = plt.figure(figsize=(16, 5))
            gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.8])

            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[0, 2])

            cls_conf = float(cls_probs[cls_pred].item())
            title_color = "green" if str(cls_pred) == str(class_id) else "red"

            fig.suptitle(
                (
                    f"True: {class_id} | Pred(CLS): {cls_pred} "
                    f"(Conf: {cls_conf * 100:.1f}%)\n"
                    f"{_fmt_pred('G', global_probs, global_pred)} | "
                    f"{_fmt_pred('P2', part2_probs, part2_pred)} | "
                    f"{_fmt_pred('P4', part4_probs, part4_pred)} | "
                    f"{_fmt_pred('C', concat_probs, concat_pred)} | "
                    f"{_fmt_pred('CLS', cls_probs, cls_pred)}"
                ),
                color=title_color,
                fontweight="bold"
            )

            ax0.imshow(cropped_img)
            ax0.axis("off")
            ax0.set_title("Original")

            ax1.imshow(overlay)
            ax1.axis("off")
            ax1.set_title("CLS Grad-CAM")

            branch_names = ["Global", "Part2", "Part4"]
            ax2.bar(branch_names, branch_attn)
            ax2.set_ylim(0, 1.0)
            ax2.set_title("CLS → Branch Attention")
            for i, v in enumerate(branch_attn):
                ax2.text(i, float(v) + 0.02,
                         f"{v:.2f}", ha="center", va="bottom")

            save_name = f"cls_attn_{os.path.splitext(os.path.basename(img_path))[0]}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(class_save_dir, save_name),
                        bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
