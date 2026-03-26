import torch
from tqdm import tqdm
from train import generate_cross_attention_bbox_local_view


def validate_one_epoch(model, val_loader, criterion, device, config, epoch=None):
    model.eval()

    running_loss = 0.0
    total_preds = 0
    correct_preds = 0

    all_predictions = []
    all_targets = []

    stage_a_epochs = config.get('stage_a_epochs', 6)
    full_view_weight = config.get('full_view_weight', 1.0)
    fused_view_weight = config.get('fused_view_weight', 0.5)
    local1_view_weight = config.get('local1_view_weight', 0.02)
    stage_a_only = epoch is not None and epoch <= stage_a_epochs

    pbar = tqdm(val_loader, desc="Validating", leave=False, colour="green")

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if stage_a_only:
                logits = model(images)
                loss = criterion(logits, labels)
            else:
                local1_views = generate_cross_attention_bbox_local_view(
                    model=model,
                    images=images,
                    threshold_ratio=config['local_crop_threshold'],
                    padding_ratio=config['local_crop_padding_ratio'],
                    min_crop_ratio=config['local_min_crop_ratio'],
                    max_crop_ratio=config['local_max_crop_ratio'],
                    fallback_crop_ratio=config['local_fallback_crop_ratio'],
                )

                outputs = model.forward_full_local(images, local1_views)
                full_logits = outputs['full_logits']
                fused_logits = outputs['fused_logits']
                local_logits = outputs['local1_logits']

                loss = full_view_weight * criterion(full_logits, labels)
                if fused_view_weight > 0:
                    loss = loss + fused_view_weight * \
                        criterion(fused_logits, labels)
                if local1_view_weight > 0:
                    loss = loss + local1_view_weight * \
                        criterion(local_logits, labels)

                logits = fused_logits

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_preds += batch_size

            preds = torch.argmax(logits, dim=1)
            correct_preds += torch.sum(preds == labels).item()

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'Loss': f"{running_loss / total_preds:.4f}",
                'Acc': f"{(correct_preds / total_preds) * 100:.2f}%",
                'Stage': 'A' if stage_a_only else 'B',
            })

    epoch_loss = running_loss / total_preds
    epoch_acc = (correct_preds / total_preds) * 100
    return epoch_loss, epoch_acc, all_predictions, all_targets
