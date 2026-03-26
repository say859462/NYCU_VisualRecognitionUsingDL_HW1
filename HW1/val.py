import torch
from tqdm import tqdm
from train import _compute_pmg_loss, _get_stage_weights


def validate_one_epoch(model, val_loader, criterion, device, config, epoch=None):
    model.eval()

    running_loss = 0.0
    total_preds = 0
    correct_preds = 0
    all_predictions = []
    all_targets = []

    stage1_epochs = config.get("pmg_stage1_epochs", 4)
    stage2_epochs = config.get("pmg_stage2_epochs", 4)
    eval_epoch = epoch if epoch is not None else (stage1_epochs + stage2_epochs + 1)
    stage_cfg = _get_stage_weights(eval_epoch, stage1_epochs, stage2_epochs, config)

    pbar = tqdm(val_loader, desc="Validating", leave=False, colour="green")

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model.forward_pmg(images)
            loss = _compute_pmg_loss(outputs, labels, criterion, stage_cfg)
            logits = outputs["concat_logits"] if stage_cfg["concat_weight"] > 0 else outputs["global_logits"]

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
                'Stage': stage_cfg['stage_name'].split('|')[0].strip(),
            })

    epoch_loss = running_loss / total_preds
    epoch_acc = (correct_preds / total_preds) * 100
    return epoch_loss, epoch_acc, all_predictions, all_targets, stage_cfg
