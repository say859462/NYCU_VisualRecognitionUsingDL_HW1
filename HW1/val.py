import torch
from tqdm import tqdm
from train import generate_fast_top1_local_view


def validate_one_epoch(model, val_loader, criterion, device, config):
    model.eval()
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    all_predictions, all_targets = [], []

    pbar = tqdm(val_loader, desc="Validating", leave=False, colour="green")

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            local1_views = generate_fast_top1_local_view(
                model=model,
                images=images,
                crop_ratio=config['local_crop_ratio'],
                padding_ratio=config['local_crop_padding_ratio']
            )

            outputs = model.forward_full_local(images, local1_views)
            logits = outputs["fused_logits"]

            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += images.size(0)

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'Loss': f"{running_loss / total_preds:.4f}",
                'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"
            })

    epoch_loss = running_loss / total_preds
    epoch_acc = (correct_preds / total_preds) * 100
    return epoch_loss, epoch_acc, all_predictions, all_targets