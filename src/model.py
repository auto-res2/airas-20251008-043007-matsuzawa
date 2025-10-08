"""src/model.py
Backbone definitions *and* wrappers implementing various Test-Time Adaptation
algorithms (Tent, ProxTTA, EATA, AdaNPC).
The wrappers execute adaptation only while the module is in ``eval`` mode so
normal supervised training is unaffected.
"""
from __future__ import annotations
import math, time, copy
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# timm is optional but required for ViT
try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy of softmax predictions."""
    p = F.softmax(x, dim=1)
    return -(p * p.log()).sum(1)


def _select_norm_affine_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return affine scale/bias parameters of BN / GN / LN layers."""
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                params.append(m.weight)
            if m.bias is not None:
                params.append(m.bias)
    return params


def _replace_bn_with_gn(module: nn.Module, gn_groups: int = 32):
    """In-place conversion of BatchNorm2d layers to GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            gn = nn.GroupNorm(
                num_groups=min(gn_groups, num_channels),
                num_channels=num_channels,
                affine=True,
            )
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, gn_groups)


# --------------------------------------------------------------------------- #
# Backbone factory
# --------------------------------------------------------------------------- #

def get_backbone(backbone_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = backbone_name.lower()

    # -------------------------------------------------- ResNet-18 GN (CIFAR)
    if name == "resnet18_gn":
        model = tvm.resnet18(pretrained=pretrained)
        _replace_bn_with_gn(model, 32)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # -------------------------------------------------- ResNet-50 BN
    if name in {"resnet50", "resnet50_bn"}:
        model = tvm.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # -------------------------------------------------- ViT-B/16 LN
    if name in {"vit_b16_ln", "vit-b16-ln"}:
        if timm is None:
            raise RuntimeError("The timm package is required for ViT backbones.")
        model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
        return model

    # Fallback to torchvision models
    if name == "resnet18":
        model = tvm.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "mobilenetv2":
        model = tvm.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    raise ValueError(f"Unknown backbone: {backbone_name}")


# --------------------------------------------------------------------------- #
# Base wrapper for TTA algorithms
# --------------------------------------------------------------------------- #
class _TTAWrapper(nn.Module):
    """Abstract base class for TTA wrappers."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):  # noqa: D401 – simple delegate
        if self.training:
            return self.base_model(x)  # no adaptation during supervised training
        return self._forward_and_adapt(x)

    def _forward_and_adapt(self, x):  # to be implemented by subclasses
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Tent (entropy minimisation on norm parameters)
# --------------------------------------------------------------------------- #
class TentAdaptor(_TTAWrapper):
    def __init__(self, base_model: nn.Module, lr: float = 1e-3, momentum: float = 0.9):
        super().__init__(base_model)
        for p in base_model.parameters():
            p.requires_grad_(False)
        # only norm affine params adaptable
        self.params = _select_norm_affine_parameters(base_model)
        for p in self.params:
            p.requires_grad_(True)
        self.opt = torch.optim.SGD(self.params, lr=lr, momentum=momentum)

    @torch.enable_grad()
    def _forward_and_adapt(self, x):
        self.base_model.train()  # ensure stats update
        logits = self.base_model(x)
        loss = softmax_entropy(logits).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return logits.detach()


# --------------------------------------------------------------------------- #
# ProxTTA – Tent + proximal regulariser on BN params
# --------------------------------------------------------------------------- #
class ProxTTAAdaptor(_TTAWrapper):
    def __init__(self, base_model: nn.Module, lr: float = 1e-3, momentum: float = 0.9, lamb: float = 1e-3):
        super().__init__(base_model)
        for p in base_model.parameters():
            p.requires_grad_(False)
        self.params = _select_norm_affine_parameters(base_model)
        for p in self.params:
            p.requires_grad_(True)
        self.opt = torch.optim.SGD(self.params, lr=lr, momentum=momentum)
        # keep a frozen copy as reference for proximal term
        self._theta0 = [p.clone().detach() for p in self.params]
        self.lamb = lamb

    @torch.enable_grad()
    def _forward_and_adapt(self, x):
        self.base_model.train()
        logits = self.base_model(x)
        ent_loss = softmax_entropy(logits).mean()
        prox_loss = 0.0
        for p, p0 in zip(self.params, self._theta0):
            prox_loss = prox_loss + (p - p0).pow(2).mean()
        loss = ent_loss + self.lamb * prox_loss
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return logits.detach()


# --------------------------------------------------------------------------- #
# EATA – Extended Tent with simple low-confidence thresholding
# --------------------------------------------------------------------------- #
class EATAAdaptor(_TTAWrapper):
    def __init__(self, base_model: nn.Module, lr: float = 1e-3, momentum: float = 0.9, thresh: float = 0.8):
        super().__init__(base_model)
        for p in base_model.parameters():
            p.requires_grad_(False)
        self.params = _select_norm_affine_parameters(base_model)
        for p in self.params:
            p.requires_grad_(True)
        self.opt = torch.optim.SGD(self.params, lr=lr, momentum=momentum)
        self.thresh = thresh

    @torch.enable_grad()
    def _forward_and_adapt(self, x):
        self.base_model.train()
        logits = self.base_model(x)
        preds = F.softmax(logits, dim=1)
        conf = preds.max(1).values  # confidence
        idx = conf >= self.thresh
        if idx.any():
            loss = softmax_entropy(logits[idx]).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        return logits.detach()


# --------------------------------------------------------------------------- #
# AdaNPC – Streaming Fisher, natural grad & probabilistic safety
# --------------------------------------------------------------------------- #
class AdaNPCAdaptor(_TTAWrapper):
    def __init__(self, base_model: nn.Module, beta: float = 0.99, delta: float = 0.1,
                 tau_max: float | None = None, micro_steps: int = 4):
        super().__init__(base_model)
        for p in base_model.parameters():
            p.requires_grad_(True)  # natural update can act on all params
        self.params = [p for p in base_model.parameters() if p.requires_grad]
        self.beta = beta
        self.delta = delta
        self.tau_max = tau_max
        self.k = micro_steps
        self._var: torch.Tensor | None = None  # diagonal Fisher EMA
        self._timer_ema: float | None = None
        # freeze batch-stats to avoid confounders
        self.base_model.eval()
        for m in self.base_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _flatten(self, grads):
        return torch.cat([g.flatten() for g in grads])

    def _reshape_and_apply(self, step):
        idx = 0
        for p in self.params:
            numel = p.numel()
            upd = step[idx: idx + numel].view_as(p)
            p.data.sub_(upd)
            idx += numel

    @torch.enable_grad()
    def _forward_and_adapt(self, x):
        t0 = time.time()
        logits = self.base_model(x)
        loss = softmax_entropy(logits).mean()
        grads = torch.autograd.grad(loss, self.params, retain_graph=False, create_graph=False)
        g_vec = self._flatten(grads)
        # update diagonal Fisher
        if self._var is None:
            self._var = g_vec.pow(2).detach() + 1e-8
        else:
            self._var = self.beta * self._var + (1 - self.beta) * g_vec.pow(2).detach() + 1e-8
        step = g_vec / self._var.sqrt()
        # safety
        deltaL = (step * g_vec).sum()
        varL = (step.pow(2) * self._var).sum().sqrt()
        safe = (deltaL + varL * math.sqrt(2 * math.log(1 / self.delta))) < 0
        if safe:
            eta = 1.0 / max(1, self.k)
            self._reshape_and_apply(eta * step)
        # micro-step scheduler
        elapsed = time.time() - t0
        if self.tau_max is not None:
            self._timer_ema = 0.8 * (self._timer_ema or elapsed) + 0.2 * elapsed
            if self._timer_ema > self.tau_max and self.k > 1:
                self.k //= 2
        return logits.detach()


# --------------------------------------------------------------------------- #
# Factory API exposed to training / evaluation scripts
# --------------------------------------------------------------------------- #

def get_model(cfg: Dict[str, Any], num_classes: int):
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "resnet18")
    pretrained = model_cfg.get("pretrained", False)

    base_model = get_backbone(name, num_classes, pretrained=pretrained)

    tta_type = model_cfg.get("tta", "none").lower()

    if tta_type == "none":
        return base_model

    if tta_type == "tent":
        tent_cfg = model_cfg.get("tent", {})
        return TentAdaptor(
            base_model,
            lr=tent_cfg.get("lr", 1e-3),
            momentum=tent_cfg.get("momentum", 0.9),
        )

    if tta_type == "proxtta":
        p_cfg = model_cfg.get("proxtta", {})
        return ProxTTAAdaptor(
            base_model,
            lr=p_cfg.get("lr", 1e-3),
            momentum=p_cfg.get("momentum", 0.9),
            lamb=p_cfg.get("lamb", 1e-3),
        )

    if tta_type == "eata":
        e_cfg = model_cfg.get("eata", {})
        return EATAAdaptor(
            base_model,
            lr=e_cfg.get("lr", 1e-3),
            momentum=e_cfg.get("momentum", 0.9),
            thresh=e_cfg.get("thresh", 0.8),
        )

    if tta_type == "adanpc":
        a_cfg = model_cfg.get("adanpc", {})
        return AdaNPCAdaptor(
            base_model,
            beta=a_cfg.get("beta", 0.99),
            delta=a_cfg.get("delta", 0.1),
            tau_max=a_cfg.get("tau_max"),
            micro_steps=a_cfg.get("micro_steps", 4),
        )

    raise ValueError(f"Unknown TTA algorithm: {tta_type}")
