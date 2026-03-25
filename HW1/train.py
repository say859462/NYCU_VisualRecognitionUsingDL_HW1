import random

import torch
from tqdm import tqdm

from utils import make_background_suppressed_views


def generate_token_drop_view(
    model,
    images,
    drop_prob=1.0,
    max_drop_ratio=0.12
):
    """
    規格：
    - 只 drop top-1 token
    - 由外部決定是否啟用，因此這裡預設 drop_prob=1.0
    - 面積限制
    """
    if random.random() > drop_prob:
        return None

    with torch.no_grad():
        attn = model.get_cross_attention_map(images)  # [B, 1, 4, 4]

    batch_size, _, height, width = images.shape
    grid = attn.shape[-1]

    patch_h = height // grid
    patch_w = width // grid

    dropped = images.clone()

    for b in range(batch_size):
        attn_map = attn[b, 0]

        idx = torch.argmax(attn_map)
        row = idx // grid
        col = idx % grid

        h0 = int(row * patch_h)
        w0 = int(col * patch_w)

        drop_h = int(patch_h * 0.5)
        drop_w = int(patch_w * 0.5)

        max_area = int(height * width * max_drop_ratio)
        area = drop_h * drop_w

        if area > max_area:
            scale = (max_area / area) ** 0.5
            drop_h = max(1, int(drop_h * scale))
            drop_w = max(1, int(drop_w * scale))

        center_h = h0 + patch_h // 2
        center_w = w0 + patch_w // 2

        h1 = max(0, center_h - drop_h // 2)
        h2 = min(height, center_h + drop_h // 2)
        w1 = max(0, center_w - drop_w // 2)
        w2 = min(width, center_w + drop_w // 2)

        dropped[b, :, h1:h2, w1:w2] = 0.0

    return dropped


def train_one_epoch(
    model,
    train_loader,
    criterion,
    epoch,
    optimizer,
    device,
    scaler,
    use_bg_suppression=True,
    bg_aux_weight=0.20,
    proto_div_weight=0.01,
    drop_view_weight=0.15,
    max_grad_norm=5.0,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # =====================================================
        # 50% background suppression / 50% token drop
        # 同一個 batch 只啟用一種，降低 VRAM
        # =====================================================
        use_bg_branch = use_bg_suppression and (random.random() < 0.5)
        use_drop_branch = not use_bg_branch

        bg_views = None
        drop_images = None

        if use_bg_branch:
            with torch.no_grad():
                saliency = model.get_saliency(images)

            bg_views = make_background_suppressed_views(
                images,
                saliency,
                threshold_ratio=0.45,
                suppress_strength=0.35,
                blur_kernel=7
            ).to(device)

        if use_drop_branch:
            drop_images = generate_token_drop_view(
                model,
                images,
                drop_prob=1.0,
                max_drop_ratio=0.12
            )

        use_amp = (device.type == 'cuda')

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

            if bg_views is not None:
                logits_bg = model(bg_views)
                loss_bg = criterion(logits_bg, labels)
                loss = loss + bg_aux_weight * loss_bg

            if drop_images is not None:
                logits_drop = model(drop_images)
                loss_drop = criterion(logits_drop, labels)
                loss = loss + drop_view_weight * loss_drop

            loss_proto = model.prototype_diversity_loss(margin=0.2)
            loss = loss + proto_div_weight * loss_proto

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)

        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

        branch_name = "BG" if bg_views is not None else "DROP"
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct / total) * 100:.2f}%",
            "aux": branch_name
        })

    return running_loss / total, (correct / total) * 100