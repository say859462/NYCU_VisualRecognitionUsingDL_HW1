import torch
from tqdm import tqdm
import torch.nn.utils as nn_utils


def train_one_epoch(model, train_loader, criterions, epoch, optimizer, device, scaler, max_grad_norm=5.0):
    model.train()
    running_loss, correct_preds, total_preds = 0.0, 0, 0

    # 拆解傳入的損失函數字典
    criterion_ce = criterions['ce']
    criterion_supcon = criterions['supcon']

    pbar = tqdm(
        train_loader, desc=f"Training Epoch {epoch}", leave=False, colour="blue")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 獲取 Soft-PMG 的三個輔助頭與融合 Embedding
            out3, out4, out_fused, e_fused = model(images)

            # 1. 深度監督：強迫淺層學會紋理，深層學會語義
            loss3 = criterion_ce(out3, labels)
            loss4 = criterion_ce(out4, labels)
            loss_fused = criterion_ce(out_fused, labels)

            # 2. 柔性對比：在 Embedding 空間推開相異物種
            loss_supcon = criterion_supcon(e_fused, labels)

            # 總損失加權 (核心融合層給予最高權重)
            loss = loss3 + loss4 + 2.0 * loss_fused + 0.5 * loss_supcon

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # 準確率計算以主輸出 (fused) 為準
        _, preds = torch.max(out_fused, 1)
        correct_preds += torch.sum(preds == labels.data).item()
        total_preds += labels.size(0)

        pbar.set_postfix({'Loss': f"{loss.item():.4f}",
                         'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"})

    return running_loss / total_preds, (correct_preds / total_preds) * 100
