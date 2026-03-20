import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import argparse
from torchvision import transforms
from tqdm import tqdm

from model import ImageClassificationModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ⭐ 新增一個 Wrapper 類別，專門用來處理 Tuple 解包與 s=15.0 縮放
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, scale=15.0):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.scale = scale

    def forward(self, x):
        logits, _ = self.model(x)
        # 直接在這裡乘上 scale，讓 Grad-CAM 抓到的梯度與機率都是最準確的
        return logits * self.scale


def main():
    parser = argparse.ArgumentParser(
        description="Pure ResNeXt Grad-CAM Visualization")
    parser.add_argument('--val_dir', type=str, default='./Dataset/data/val')
    parser.add_argument('--num_samples_per_class', type=int, default=3)
    parser.add_argument('--model_path', type=str,
                        default='./Model_Weight/best_model.pth')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--save_dir', type=str,
                        default='./Plot/GradCAM_Outputs/37th')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    model = ImageClassificationModel(
        num_classes=args.num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # ⭐ 將模型包裝起來
    wrapped_model = ModelWrapper(model, scale=15.0)

    # ⭐ 注意：這裡的 model 改傳入 wrapped_model，但 target_layers 依然綁定原本的 model.rsa
    cam_l3 = GradCAM(model=wrapped_model, target_layers=[model.rsa3])
    cam_l4 = GradCAM(model=wrapped_model, target_layers=[model.rsa4])

    preprocess_geo = transforms.Compose(
        [transforms.Resize(640), transforms.CenterCrop(576)])
    preprocess_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for class_id in tqdm(range(args.num_classes), desc="Classes Processed"):
        class_dir = os.path.join(args.val_dir, str(class_id))
        if not os.path.exists(class_dir):
            continue

        class_save_dir = os.path.join(args.save_dir, str(class_id))
        os.makedirs(class_save_dir, exist_ok=True)

        all_images = [os.path.join(class_dir, f) for f in os.listdir(
            class_dir) if f.lower().endswith(valid_extensions)]

        # 避免樣本不足的防呆機制
        if len(all_images) == 0:
            continue

        sampled_image_paths = random.sample(all_images, min(
            args.num_samples_per_class, len(all_images)))

        for img_path in sampled_image_paths:
            raw_img = Image.open(img_path).convert('RGB')
            cropped_img = preprocess_geo(raw_img)
            input_tensor = preprocess_tensor(
                cropped_img).unsqueeze(0).to(device)
            rgb_img = np.float32(cropped_img) / 255.0

            with torch.no_grad():
                # ⭐ 這裡可以直接呼叫 wrapper，它會吐出縮放好且單一的 Tensor
                outputs = wrapped_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                pred_class = probabilities.argmax().item()
                pred_score = probabilities[pred_class].item()

            # 產生兩層的熱力圖
            grayscale_cam_l3 = cam_l3(
                input_tensor=input_tensor, targets=None)[0, :]
            vis_l3 = show_cam_on_image(rgb_img, grayscale_cam_l3, use_rgb=True)

            grayscale_cam_l4 = cam_l4(
                input_tensor=input_tensor, targets=None)[0, :]
            vis_l4 = show_cam_on_image(rgb_img, grayscale_cam_l4, use_rgb=True)

            # 繪製圖片
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            color = 'green' if str(pred_class) == str(class_id) else 'red'
            fig.suptitle(
                f"True: {class_id} | Pred: {pred_class} (Conf: {pred_score*100:.1f}%)", color=color, fontweight='bold')

            axes[0].imshow(cropped_img)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            axes[1].imshow(vis_l3)
            axes[1].set_title("Layer 3: Grad-CAM")
            axes[1].axis('off')
            axes[2].imshow(vis_l4)
            axes[2].set_title("Layer 4: Grad-CAM")
            axes[2].axis('off')

            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            plt.savefig(os.path.join(class_save_dir,
                        f"cam_{base_filename}.png"))
            plt.close(fig)


if __name__ == '__main__':
    main()
