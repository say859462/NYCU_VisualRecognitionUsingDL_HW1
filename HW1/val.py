import torch
from tqdm import tqdm


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    # Lists to store predictions and targets for analysis (e.g. Determine which classes are most confused)
    all_predictions = []
    all_targets = []
    
    # tqdm progress bar
    pbar = tqdm(val_loader, desc="Validating", leave=False, colour="green")
    with torch.no_grad():
        for images, labels in pbar:
            
            bs, n_crops, c, h, w = images.size()
            
            images = images.view(-1, c, h, w).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            outputs_avg = outputs.view(bs, n_crops, -1).mean(dim=1)
            
            loss = criterion(outputs_avg, labels)

            # Loss is averaged over the batch, so we multiply by batch size to get total loss for the batch
            # In case of the last batch which might be smaller, this will still work correctly
            running_loss += loss.item() * bs

            # Take the class with the highest probability as the predicted label
            _, preds = torch.max(outputs_avg, 1)

            correct_preds += torch.sum(preds == labels.data).item()
            total_preds += bs

            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            # tqdm progress bar postfix update
            pbar.set_postfix({
                'Loss': f"{running_loss / total_preds:.4f}",
                'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"
            })

    epoch_loss = running_loss / total_preds
    epoch_acc = (correct_preds / total_preds) * 100  # Percentage

    return epoch_loss, epoch_acc, all_predictions, all_targets
