"""Grad-CAM visualization script for PMG branches."""

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


def _normalize_map(x: np.ndarray) -> np.ndarray:
    """Normalize a NumPy heatmap into the [0, 1] range."""
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def _resize_heatmap(heatmap: np.ndarray, out_hw) -> np.ndarray:
    """Resize a heatmap to the requested output height and width."""
    if heatmap.shape[:2] == out_hw:
        return heatmap
    heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    heatmap_tensor = F.interpolate(
        heatmap_tensor,
        size=out_hw,
        mode="bilinear",
        align_corners=False,
    )
    return heatmap_tensor[0, 0].cpu().numpy()


def _overlay_heatmap_on_image(rgb_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap with the original RGB image for visualization."""
    heatmap = _resize_heatmap(_normalize_map(heatmap), rgb_img.shape[:2])
    cmap = plt.get_cmap("jet")
    heatmap_color = cmap(heatmap)[..., :3]
    overlay = (1.0 - alpha) * rgb_img + alpha * heatmap_color
    return np.clip(overlay, 0.0, 1.0)


def compute_gradcam(model, input_tensor, target_module, target_logits_key):
    """Compute a Grad-CAM map for one target module and logits key."""
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
            gradients.append(grad_output[0])

    handle_fwd = target_module.register_forward_hook(forward_hook)
    handle_bwd = target_module.register_full_backward_hook(backward_hook)

    try:
        input_tensor = input_tensor.requires_grad_(True)
        outputs = model.forward_pmg(input_tensor)
        logits = outputs[target_logits_key]
        pred_idx = logits.argmax(dim=1).item()
        score = logits[:, pred_idx].sum()

        model.zero_grad(set_to_none=True)
        score.backward()

        feat = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feat).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = _normalize_map(cam)

        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        return cam, pred_idx, probs, outputs
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def compute_concat_cam(model, input_tensor):
    """Build a CAM-like visualization directly from the model attention map."""
    cams = []
    pred_idx = None
    probs = None
    outputs = None

    for target_module in [model.global_proj, model.part2_proj, model.part4_proj]:
        cam, pred_idx, probs, outputs = compute_gradcam(
            model=model,
            input_tensor=input_tensor.clone(),
            target_module=target_module,
            target_logits_key="concat_logits",
        )
        cams.append(cam)

    concat_cam = np.mean(np.stack(cams, axis=0), axis=0)
    concat_cam = _normalize_map(concat_cam)
    return concat_cam, pred_idx, probs, outputs


def main():
    """Export branch-wise Grad-CAM figures for validation samples."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--val_dir", type=str, default="./Dataset/data/val")
    parser.add_argument("--num_samples_per_class", type=int, default=2)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./Plot/Attention_Outputs/UniformJointFusion")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    default_model_path = config.get("best_concat_model_path", config["best_model_path"])
    model_path = args.model_path if args.model_path is not None else default_model_path

    model = ImageClassificationModel(
        num_classes=config["num_classes"],
        pretrained=False,
        num_subcenters=config.get("num_subcenters", 3),
        embed_dim=config.get("embed_dim", 256),
        backbone_name=config.get("backbone_name", "resnet152_partial_res2net"),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    eval_resize = config.get("eval_resize", 576)
    preprocess_geo = transforms.Compose([
        transforms.Resize((eval_resize, eval_resize))
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
            os.path.join(class_dir, name)
            for name in os.listdir(class_dir)
            if name.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        if not all_images:
            continue

        sampled_paths = random.sample(all_images, min(args.num_samples_per_class, len(all_images)))

        for img_path in sampled_paths:
            raw_img = Image.open(img_path).convert("RGB")
            vis_img = preprocess_geo(raw_img)
            input_tensor = preprocess_tensor(vis_img).unsqueeze(0).to(device)
            rgb_img = np.asarray(vis_img).astype(np.float32) / 255.0

            global_cam, global_pred, global_probs, outputs = compute_gradcam(
                model=model,
                input_tensor=input_tensor.clone(),
                target_module=model.global_proj,
                target_logits_key="global_logits",
            )
            part2_cam, part2_pred, part2_probs, _ = compute_gradcam(
                model=model,
                input_tensor=input_tensor.clone(),
                target_module=model.part2_proj,
                target_logits_key="part2_logits",
            )
            part4_cam, part4_pred, part4_probs, _ = compute_gradcam(
                model=model,
                input_tensor=input_tensor.clone(),
                target_module=model.part4_proj,
                target_logits_key="part4_logits",
            )
            concat_cam, concat_pred, concat_probs, _ = compute_concat_cam(
                model=model,
                input_tensor=input_tensor.clone(),
            )

            attention_map = outputs["attention_map"][0, 0].detach().cpu().numpy()
            attention_overlay = _overlay_heatmap_on_image(rgb_img, attention_map, alpha=0.45)

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            axes[0].imshow(rgb_img)
            axes[0].set_title("Original")

            axes[1].imshow(_overlay_heatmap_on_image(rgb_img, global_cam, alpha=0.45))
            axes[1].set_title(f"Global CAM -> {global_pred} ({global_probs.max().item():.2f})")

            axes[2].imshow(_overlay_heatmap_on_image(rgb_img, part2_cam, alpha=0.45))
            axes[2].set_title(f"Part2 CAM -> {part2_pred} ({part2_probs.max().item():.2f})")

            axes[3].imshow(_overlay_heatmap_on_image(rgb_img, part4_cam, alpha=0.45))
            axes[3].set_title(f"Part4 CAM -> {part4_pred} ({part4_probs.max().item():.2f})")

            axes[4].imshow(attention_overlay)
            axes[4].set_title("Attention Map")

            axes[5].imshow(_overlay_heatmap_on_image(rgb_img, concat_cam, alpha=0.45))
            axes[5].set_title(f"Concat CAM (avg branches) -> {concat_pred} ({concat_probs.max().item():.2f})")

            for ax in axes:
                ax.axis("off")

            summary = (
                f"True: {class_id} | G:{global_pred} | P2:{part2_pred} | "
                f"P4:{part4_pred} | Concat:{concat_pred}"
            )
            fig.suptitle(summary, fontsize=14, fontweight="bold")
            plt.tight_layout()

            save_path = os.path.join(
                class_save_dir,
                f"{os.path.splitext(os.path.basename(img_path))[0]}_vis.png",
            )
            plt.savefig(save_path, dpi=250)
            plt.close(fig)


if __name__ == "__main__":
    main()
