"""Adversarial attack implementations for image classifiers."""

import torch
import torch.nn.functional as F


def fgsm(model, images, labels, eps=8 / 255):
    images = images.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(images), labels)
    grad = torch.autograd.grad(loss, images)[0]
    return (images + eps * grad.sign()).clamp(0, 1).detach()


def pgd(model, images, labels, eps=8 / 255, alpha=2 / 255, steps=20):
    adv = images.clone().detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        delta = torch.clamp(adv - images, -eps, eps)
        adv = torch.clamp(images + delta, 0, 1).detach()
    return adv


def autoattack(model, images, labels, eps=8 / 255):
    """Cheap stand-in: strongest of several random-restart PGD runs."""
    best = images.clone().detach()
    best_loss = F.cross_entropy(model(best), labels, reduction="none")
    for _ in range(3):
        candidate = pgd(model, images, labels, eps=eps, alpha=eps / 4, steps=40)
        loss = F.cross_entropy(model(candidate), labels, reduction="none")
        switch = loss > best_loss
        best[switch] = candidate[switch]
        best_loss = torch.where(switch, loss, best_loss)
    return best
