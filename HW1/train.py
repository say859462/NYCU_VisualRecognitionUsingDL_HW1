
from typing import Dict

import torch
from tqdm import tqdm


def _get_stage_weights(epoch: int, stage1_epochs: int, stage2_epochs: int, config: Dict) -> Dict:
    if epoch <= stage1_epochs:
        return {
            "stage_name": "Stage 1 | Global anchor",
            "global_weight": config.get("pmg_stage1_global_weight", 1.0),
            "part2_weight": 0.0,
            "part4_weight": 0.0,
            "concat_weight": 0.0,
            "use_concat_as_main": False,
        }
    if epoch <= stage1_epochs + stage2_epochs:
        return {
            "stage_name": "Stage 2 | Global + coarse branch",
            "global_weight": config.get("pmg_stage2_global_weight", 1.0),
            "part2_weight": config.get("pmg_stage2_part2_weight", 0.25),
            "part4_weight": 0.0,
            "concat_weight": 0.0,
            "use_concat_as_main": False,
        }
    return {
        "stage_name": "Stage 3 | Full PMG + joint consensus fusion",
        "global_weight": config.get("pmg_stage3_global_weight", 0.30),
        "part2_weight": config.get("pmg_stage3_part2_weight", 0.15),
        "part4_weight": config.get("pmg_stage3_part4_weight", 0.10),
        "concat_weight": config.get("pmg_stage3_concat_weight", 1.0),
        "use_concat_as_main": True,
    }


def _compute_pmg_loss(outputs, labels, criterion, stage_cfg):
    loss = outputs["global_logits"].new_tensor(0.0)
    if stage_cfg["global_weight"] > 0:
        loss = loss + stage_cfg["global_weight"] * criterion(outputs["global_logits"], labels)
    if stage_cfg["part2_weight"] > 0:
        loss = loss + stage_cfg["part2_weight"] * criterion(outputs["part2_logits"], labels)
    if stage_cfg["part4_weight"] > 0:
        loss = loss + stage_cfg["part4_weight"] * criterion(outputs["part4_logits"], labels)
    if stage_cfg["concat_weight"] > 0:
        loss = loss + stage_cfg["concat_weight"] * criterion(outputs["concat_logits"], labels)
    return loss


def _get_eval_logits(outputs, stage_cfg):
    if stage_cfg.get("use_concat_as_main", False):
        return outputs["concat_logits"]
    return outputs["global_logits"]


def _compute_batch_acc(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == labels).item()
    return correct, preds


def train_one_epoch(model, train_loader, criterion, epoch, optimizer, device, scaler, config, max_grad_norm: float = 5.0):
    model.train()
    stage_cfg = _get_stage_weights(
        epoch=epoch,
        stage1_epochs=config.get("pmg_stage1_epochs", 8),
        stage2_epochs=config.get("pmg_stage2_epochs", 4),
        config=config,
    )

    running_loss = 0.0
    total = 0
    main_correct = 0
    concat_correct = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model.forward_pmg(images)
            loss = _compute_pmg_loss(outputs, labels, criterion, stage_cfg)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size

        logits_for_main = _get_eval_logits(outputs, stage_cfg)
        batch_main_correct, _ = _compute_batch_acc(logits_for_main, labels)
        main_correct += batch_main_correct
        batch_concat_correct, _ = _compute_batch_acc(outputs["concat_logits"], labels)
        concat_correct += batch_concat_correct

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "main_acc": f"{(main_correct / total) * 100:.2f}%"})

    return {
        "loss": running_loss / max(1, total),
        "main_acc": (main_correct / max(1, total)) * 100,
        "concat_acc": (concat_correct / max(1, total)) * 100,
        "stage_cfg": stage_cfg,
    }
