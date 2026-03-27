import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import ImageDataset
from model import ImageClassificationModel


def main():
    parser = argparse.ArgumentParser(description="Final Inference")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    model_path = args.model_path if args.model_path is not None else config["best_model_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = ImageDataset(
        root_dir=config["data_dir"],
        split="test",
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = ImageClassificationModel(
        num_classes=config["num_classes"],
        pretrained=False,
        num_subcenters=config.get("num_subcenters", 3),
        embed_dim=config.get("embed_dim", 256),
        token_grid_size=config.get("token_grid_size", 7),
        cls_num_heads=config.get("cls_num_heads", 4),
        cls_attn_dropout=config.get("cls_attn_dropout", 0.1),
        cls_ffn_ratio=config.get("cls_ffn_ratio", 2.0),
        cls_block_dropout=config.get("cls_block_dropout", 0.1),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_predictions = []
    print(f"🚀 Running Final Inference from: {model_path}")

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            outputs = model.forward_pmg(images)
            preds = torch.argmax(outputs["cls_logits"], dim=1)
            all_predictions.extend(preds.cpu().tolist())

    image_names = [
        os.path.splitext(os.path.basename(p))[0]
        for p in test_dataset.image_paths
    ]

    submission_df = pd.DataFrame({
        "image_name": image_names,
        "pred_label": all_predictions
    })
    submission_df.to_csv("prediction.csv", index=False)

    print("\n🎉 Submission CSV saved!")


if __name__ == "__main__":
    main()
