"""Robustness metrics."""

import torch


@torch.no_grad()
def clean_accuracy(model, loader, device="cuda"):
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images).argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


@torch.no_grad()
def robust_accuracy(model, loader, attack, device="cuda"):
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv = attack(model, images, labels)
        pred = model(adv).argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


@torch.no_grad()
def expected_calibration_error(model, loader, n_bins=15, device="cuda"):
    confidences, accuracies = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        probs = model(images).softmax(dim=1)
        conf, pred = probs.max(dim=1)
        confidences.append(conf)
        accuracies.append((pred == labels).float())
    conf = torch.cat(confidences)
    acc = torch.cat(accuracies)
    ece = torch.zeros(1, device=device)
    for lo in torch.linspace(0, 1, n_bins + 1, device=device)[:-1]:
        hi = lo + 1.0 / n_bins
        mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += mask.float().mean() * (acc[mask].mean() - conf[mask].mean()).abs()
    return float(ece)
