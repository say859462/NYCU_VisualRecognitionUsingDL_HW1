import torch
import torch.nn.functional as F
from tqdm import tqdm


def generate_attention_guided_local_view(
    model,
    images,
    source="saliency",
    threshold_ratio=0.60,
    min_crop_ratio=0.35,
    padding_ratio=0.15,
):
    with torch.no_grad():
        if source == "cross_attention":
            response_map = model.get_cross_attention_map(images)
        else:
            response_map = model.get_saliency(images).unsqueeze(1)

        response_map = F.interpolate(
            response_map,
            size=images.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

    batch_size, _, height, width = images.shape
    min_crop_h = max(2, int(height * min_crop_ratio))
    min_crop_w = max(2, int(width * min_crop_ratio))
    local_views = []

    for idx in range(batch_size):
        attn_map = response_map[idx, 0]
        threshold = attn_map.max() * threshold_ratio
        mask = attn_map >= threshold
        nonzero = torch.nonzero(mask, as_tuple=False)

        if nonzero.numel() == 0:
            center_y, center_x = height // 2, width // 2
            crop_h = min_crop_h
            crop_w = min_crop_w
            y1 = max(0, center_y - crop_h // 2)
            y2 = min(height, center_y + crop_h // 2)
            x1 = max(0, center_x - crop_w // 2)
            x2 = min(width, center_x + crop_w // 2)
        else:
            y1 = int(nonzero[:, 0].min().item())
            y2 = int(nonzero[:, 0].max().item()) + 1
            x1 = int(nonzero[:, 1].min().item())
            x2 = int(nonzero[:, 1].max().item()) + 1

            box_h = y2 - y1
            box_w = x2 - x1
            pad_h = int(box_h * padding_ratio)
            pad_w = int(box_w * padding_ratio)

            y1 = max(0, y1 - pad_h)
            y2 = min(height, y2 + pad_h)
            x1 = max(0, x1 - pad_w)
            x2 = min(width, x2 + pad_w)

            if (y2 - y1) < min_crop_h:
                center_y = (y1 + y2) // 2
                y1 = max(0, center_y - min_crop_h // 2)
                y2 = min(height, y1 + min_crop_h)
                y1 = max(0, y2 - min_crop_h)

            if (x2 - x1) < min_crop_w:
                center_x = (x1 + x2) // 2
                x1 = max(0, center_x - min_crop_w // 2)
                x2 = min(width, x1 + min_crop_w)
                x1 = max(0, x2 - min_crop_w)

        crop = images[idx:idx + 1, :, y1:y2, x1:x2]
        crop = F.interpolate(
            crop,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        local_views.append(crop)

    return torch.cat(local_views, dim=0)


def train_one_epoch(
    model,
    train_loader,
    criterion,
    epoch,
    optimizer,
    device,
    scaler,
    local_view_weight=0.15,
    local_view_source="saliency",
    local_crop_threshold=0.60,
    local_min_crop_ratio=0.35,
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

        local_images = generate_attention_guided_local_view(
            model=model,
            images=images,
            source=local_view_source,
            threshold_ratio=local_crop_threshold,
            min_crop_ratio=local_min_crop_ratio,
        )

        use_amp = (device.type == 'cuda')
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model.forward_full_local(images, local_images)
            fused_logits = outputs["fused_logits"]
            local_logits = outputs["local_logits"]

            loss = criterion(fused_logits, labels)
            if local_view_weight > 0:
                loss = loss + local_view_weight * \
                    criterion(local_logits, labels)

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
