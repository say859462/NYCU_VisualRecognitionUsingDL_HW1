import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# 載入您的自定義模組
from dataset import ImageDataset
from model import ImageClassificationModel


def get_features_and_preds(model, dataloader, device):
    """
    提取 512 維 Embedding 和預測結果
    """
    model.eval()
    all_embeddings = []
    all_preds = []
    all_labels = []

    # 建立一個 Hook 來提取 embedding 層的輸出
    # 或是直接模擬 forward 過程
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)

            # 模擬 forward 到 embedding 為止
            f3 = model.backbone_l1_l3(images)
            f3_att = model.se_l3(f3)
            f3_reduced = model.reduce3(f3_att)
            p3 = model.gem(f3_reduced).flatten(1)

            f4 = model.backbone_l4(f3)
            p4_cbp = model.cbp(f4)
            p4 = model.fc_cbp(p4_cbp)

            fused = torch.cat([p3, p4], dim=1)
            embeddings = model.embedding(fused)  # 這裡就是 512 維特徵

            # 最後分類
            logits = model.classifier(embeddings)
            # 注意：這裡加上您訓練時的縮放係數 s=25 以獲得準確預測
            preds = torch.argmax(logits * 25.0, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    return (np.concatenate(all_embeddings),
            np.concatenate(all_preds),
            np.concatenate(all_labels))


def plot_local_confusion_matrix(y_true, y_pred, target_classes, save_path):
    """
    繪製特定類別的局部混淆矩陣
    """
    # 過濾出只包含目標類別的數據
    mask = np.isin(y_true, target_classes) & np.isin(y_pred, target_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    cm = confusion_matrix(
        y_true_filtered, y_pred_filtered, labels=target_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_classes, yticklabels=target_classes)
    plt.title(f"Local Confusion Matrix for Classes: {target_classes}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    print(f"Local Confusion Matrix saved to: {save_path}")


def plot_tsne(embeddings, labels, target_classes, save_path):
    """
    修正後的 t-SNE 繪圖函式，移除可能導致錯誤的參數
    """
    # 僅過濾出目標類別
    mask = np.isin(labels, target_classes)
    feat_subset = embeddings[mask]
    label_subset = labels[mask]

    print(f"Running t-SNE on {len(feat_subset)} samples...")

    # 修正點：移除 n_iter 以相容不同版本的 sklearn
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(feat_subset) - 1),
        random_state=42,
        init='pca',
        learning_rate='auto'
    )

    embed_2d = tsne.fit_transform(feat_subset)

    plt.figure(figsize=(10, 8))
    # 建立一個類別對應到名稱/ID 的 map 方便閱讀
    for class_id in target_classes:
        class_mask = (label_subset == class_id)
        plt.scatter(
            embed_2d[class_mask, 0],
            embed_2d[class_mask, 1],
            label=f"Class {class_id}",
            alpha=0.7,
            edgecolors='w'
        )

    plt.legend(title="Classes")
    plt.title(f"t-SNE Visualization (Focus Classes: {target_classes})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ t-SNE plot saved to: {save_path}")


def main():
    # 設定參數
    MODEL_PATH = './Model_Weight/best_model.pth'
    DATA_DIR = './Dataset/data'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 512
    # 定義您想觀察的類別 (例如蝴蝶 Class 2, 以及它常被誤認的 Class 76, 58 等)
    TARGET_CLASSES = [2, 76, 58, 20, 6, 16, 86, 20,
                      45, 44, 70, 48, 46, 56, 31, 75, 50, 81, 35]

    # 1. 準備數據
    val_transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    val_dataset = ImageDataset(
        root_dir=DATA_DIR, split="val", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 2. 載入模型
    model = ImageClassificationModel(
        num_classes=100, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded weights from {MODEL_PATH}")

    # 3. 提取特徵與預測
    embeddings, preds, labels = get_features_and_preds(
        model, val_loader, DEVICE)

    # 4. 繪製圖表
    os.makedirs('./Plot/19th/Diagnostic', exist_ok=True)

    # 局部混淆矩陣
    plot_local_confusion_matrix(
        labels, preds, TARGET_CLASSES, './Plot/19th/Diagnostic/cm_local.png')

    # t-SNE
    plot_tsne(embeddings, labels, TARGET_CLASSES,
              './Plot/19th/Diagnostic/tsne_local.png')


if __name__ == "__main__":
    main()
