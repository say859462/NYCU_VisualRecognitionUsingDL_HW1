import torch
import torch.nn.functional as F
from tqdm import tqdm


def _build_box_from_peak(cy, cx, crop_h, crop_w, pad_h, pad_w, H, W, device):
    y1 = cy - crop_h // 2 - pad_h
    y2 = cy + crop_h // 2 + pad_h
    x1 = cx - crop_w // 2 - pad_w
    x2 = cx + crop_w // 2 + pad_w

    y1 = torch.clamp(y1, 0, H - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y2 = torch.clamp(y2, y1 + 1, H)
    x2 = torch.clamp(x2, x1 + 1, W)

    return torch.stack([
        y1.to(device=device, dtype=torch.long),
        x1.to(device=device, dtype=torch.long),
        y2.to(device=device, dtype=torch.long),
        x2.to(device=device, dtype=torch.long),
    ])


def generate_fast_top1_local_view(
    model,
    images,
    crop_ratio=0.40,
    padding_ratio=0.12,
):
    """
    Fast GPU-based top-1 saliency crop.
    用 saliency peak 建立單一 local bounding box，不做 CPU 搜索。
    """
    with torch.no_grad():
        response_map = model.get_saliency(images)  # (B, h, w)
        response_map = F.interpolate(
            response_map.unsqueeze(1),
            size=images.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

    B, H, W = response_map.shape
    device = images.device

    crop_h = max(2, int(H * crop_ratio))
    crop_w = max(2, int(W * crop_ratio))
    pad_h = max(1, int(crop_h * padding_ratio))
    pad_w = max(1, int(crop_w * padding_ratio))

    flat = response_map.view(B, -1)
    top1_idx = torch.argmax(flat, dim=1)

    ys = top1_idx // W
    xs = top1_idx % W

    local_list = []

    for b in range(B):
        box = _build_box_from_peak(
            ys[b], xs[b], crop_h, crop_w, pad_h, pad_w, H, W, device
        )

        y1, x1, y2, x2 = [int(v.item()) for v in box]

        crop = images[b:b + 1, :, y1:y2, x1:x2]
        crop = F.interpolate(
            crop, size=(H, W), mode='bilinear', align_corners=False
        )
        local_list.append(crop)

    local = torch.cat(local_list, dim=0)
    return local


def train_one_epoch(
    model,
    train_loader,
    criterion,
    epoch,
    optimizer,
    device,
    scaler,
    local1_view_weight=0.10,
    local_crop_ratio=0.40,
    local_crop_padding_ratio=0.12,
    max_grad_norm=5.0,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        local1_images = generate_fast_top1_local_view(
            model=model,
            images=images,
            crop_ratio=local_crop_ratio,
            padding_ratio=local_crop_padding_ratio,
        )

        use_amp = (device.type == 'cuda')
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model.forward_full_local(images, local1_images)
            fused_logits = outputs["fused_logits"]
            local1_logits = outputs["local1_logits"]

            loss = criterion(fused_logits, labels)

            if local1_view_weight > 0:
                loss = loss + local1_view_weight * criterion(local1_logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(fused_logits, dim=1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct / total) * 100:.2f}%",
        })

    return running_loss / total, (correct / total) * 100