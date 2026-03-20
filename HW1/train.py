import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn.utils as nn_utils

def train_one_epoch(model, train_loader, criterion_ce, criterion_ldam, cb_weights, epoch, optimizer, device, scaler, s=15.0, max_grad_norm=2.0):
    model.train()
    running_loss, correct_preds, total_preds = 0.0, 0, 0

    # ⭐ 提前至 Epoch 8 啟動，避免全圖特徵過擬合且保持 LR 動能
    enable_local = epoch > 8 
    if epoch == 9:
        print("\n" + "🔥" * 30 + "\n🚀 [DRW & Attention Zoom] 啟動！擴大視野並強化長尾防禦\n" + "🔥" * 30 + "\n")

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch} (Local: {enable_local})", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits_global, f4_global = model(images)

            if not enable_local:
                loss = criterion_ce(logits_global * s, labels)
                scaler.scale(loss).backward()
            else:
                # --- 優化後的動態裁切 (更廣的視野) ---
                with torch.no_grad():
                    B, C, H, W = images.shape
                    attn_map = f4_global.mean(dim=1)
                    local_imgs = []
                    for i in range(B):
                        amap = attn_map[i]
                        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-6)
                        # ⭐ 降低閾值 (0.2) 以包含更多周邊特徵
                        mask = amap > (amap.mean() + 0.2 * amap.std())
                        nonzero = torch.nonzero(mask)
                        
                        if len(nonzero) == 0:
                            local_imgs.append(F.interpolate(images[i:i+1], size=(384, 384), mode='bilinear')[0])
                            continue

                        y_min, x_min = nonzero.min(dim=0)[0]; y_max, x_max = nonzero.max(dim=0)[0]
                        scale_y, scale_x = H / amap.shape[0], W / amap.shape[1]
                        y_min, y_max = int(y_min * scale_y), int(y_max * scale_y)
                        x_min, x_max = int(x_min * scale_x), int(x_max * scale_x)

                        # ⭐ 增加 Padding (25%) 以保留背景上下文
                        h_box, w_box = y_max - y_min, x_max - x_min
                        y_min = max(0, int(y_min - 0.25 * h_box)); y_max = min(H, int(y_max + 0.25 * h_box))
                        x_min = max(0, int(x_min - 0.25 * w_box)); x_max = min(W, int(x_max + 0.25 * w_box))

                        crop = images[i:i+1, :, y_min:y_max, x_min:x_max]
                        # ⭐ 統一 384x384 避免維度衝突並提速
                        local_imgs.append(F.interpolate(crop, size=(384, 384), mode='bilinear', align_corners=False)[0])
                    x_local = torch.stack(local_imgs)

                # --- 序列化反向傳播 (省顯存) ---
                w_classifier = model.classifier.weight
                loss_global = 0.6 * criterion_ldam(logits_global, labels, w_classifier, cb_weights)
                scaler.scale(loss_global).backward(retain_graph=False)

                logits_local, _ = model(x_local)
                loss_local = 0.4 * criterion_ldam(logits_local, labels, w_classifier, cb_weights)
                scaler.scale(loss_local).backward()
                loss = loss_global + loss_local

        scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer); scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits_global, 1)
        correct_preds += torch.sum(preds == labels.data).item()
        total_preds += labels.size(0)

        pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{(correct_preds / total_preds) * 100:.2f}%"})

    return running_loss / total_preds, (correct_preds / total_preds) * 100