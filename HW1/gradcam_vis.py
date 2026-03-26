import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import ImageClassificationModel


def _denorm(tensor):
    img = tensor.detach().cpu()
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return torch.clamp(img, 0.0, 1.0).permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--val_dir', type=str, default='./Dataset/data/val')
    parser.add_argument('--num_samples_per_class', type=int, default=3)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str,
                        default='./Plot/Attention_Outputs/current')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = args.model_path if args.model_path is not None else config['best_model_path']

    model = ImageClassificationModel(
        num_classes=config['num_classes'],
        pretrained=False,
        num_subcenters=config.get('num_subcenters', 3),
        embed_dim=config.get('embed_dim', 256)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preprocess_geo = transforms.Compose(
        [transforms.Resize(512), transforms.CenterCrop(448)])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for class_id in tqdm(range(config['num_classes']), desc="Classes Processed"):
        class_dir = os.path.join(args.val_dir, str(class_id))
        if not os.path.exists(class_dir):
            continue
        class_save_dir = os.path.join(args.save_dir, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)
        all_images = [os.path.join(class_dir, f) for f in os.listdir(
            class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not all_images:
            continue
        sampled_image_paths = random.sample(all_images, min(
            args.num_samples_per_class, len(all_images)))

        for img_path in sampled_image_paths:
            raw_img = Image.open(img_path).convert('RGB')
            cropped_img = preprocess_geo(raw_img)
            input_tensor = preprocess_tensor(
                cropped_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model.forward_pmg(input_tensor)
                global_probs = torch.softmax(
                    outputs['global_logits'], dim=1)[0]
                part2_probs = torch.softmax(outputs['part2_logits'], dim=1)[0]
                part4_probs = torch.softmax(outputs['part4_logits'], dim=1)[0]
                concat_probs = torch.softmax(
                    outputs['concat_logits'], dim=1)[0]
                saliency = model.get_saliency(input_tensor)[0].cpu().numpy()

            saliency = (saliency - saliency.min()) / \
                (saliency.max() - saliency.min() + 1e-8)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(
                f"True: {class_id} | G:{global_probs.argmax().item()}({global_probs.max().item():.2f}) | "
                f"P2:{part2_probs.argmax().item()}({part2_probs.max().item():.2f}) | "
                f"P4:{part4_probs.argmax().item()}({part4_probs.max().item():.2f}) | "
                f"C:{concat_probs.argmax().item()}({concat_probs.max().item():.2f})",
                fontweight='bold'
            )
            axes[0].imshow(cropped_img)
            axes[0].axis('off')
            axes[0].set_title('Original')
            axes[1].imshow(cropped_img)
            axes[1].imshow(saliency, cmap='jet', alpha=0.45)
            axes[1].axis('off')
            axes[1].set_title('PMG Saliency')
            save_name = f"pmg_{os.path.splitext(os.path.basename(img_path))[0]}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(class_save_dir, save_name),
                        bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    main()
