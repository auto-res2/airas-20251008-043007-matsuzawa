"""src/model.py
All model definitions **plus** wrappers implementing Test-Time Adaptation (TTA).
Supported TTA methods
---------------------
1. none   – ordinary forward pass (baseline / source model)
2. tent   – entropy minimisation on BN/LN/GN affine parameters (cf. *Tent*)
3. adanpc – our proposed AdaNPC algorithm

Adding a new method simply requires: (i) implementing a new *Adaptor* class,
(ii) registering it in *get_model* below.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torchvision.models as tvm
import timm  # third-party vision models

# ---------------------------------------------------------------------------
# Utility – entropy helper
# ---------------------------------------------------------------------------

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:  # (B, C)
    """Return entropy of softmax distribution of *logits* for each sample."""
    probs = torch.softmax(logits, dim=1)
    logp = torch.log_softmax(logits, dim=1)
    return -(probs * logp).sum(dim=1)


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

def _timm_create(name: str, num_classes: int, pretrained: bool):
    # timm directly allows overriding num_classes so we just forward
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def get_backbone(backbone_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = backbone_name.lower()
    if name == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name in {"vit_base_patch16_224", "vit-b/16", "vit_b16"}:
        # Normalise aliases
        return _timm_create("vit_base_patch16_224", num_classes, pretrained)

    # Fallback: attempt to create via timm for arbitrary names
    try:
        return _timm_create(backbone_name, num_classes, pretrained)
    except Exception as ex:
        raise ValueError(f"Unknown backbone: {backbone_name}") from ex


# ---------------------------------------------------------------------------
# 1) Tent adaptor – entropy minimisation on normalisation layer affine params
# ---------------------------------------------------------------------------
class TentAdaptor(nn.Module):
    """Simplified implementation of Tent (https://arxiv.org/abs/2006.10726).

    *Only* the affine scale/shift parameters of normalisation layers are
    adapted, following the original paper.
    """

    def __init__(self, base_model: nn.Module, lr: float = 1e-3, tau_max: float | None = None):
        super().__init__()
        self.base_model = base_model
        self.tau_max = tau_max

        for p in self.base_model.parameters():
            p.requires_grad_(False)  # freeze all by default

        self.params: List[nn.Parameter] = []
        for m in self.base_model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    m.weight.requires_grad = True
                    self.params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad = True
                    self.params.append(m.bias)

        self.optimizer = torch.optim.SGD(self.params, lr=lr)
        self.timer_ema: float | None = None

    def forward(self, x):
        if torch.is_grad_enabled():
            start = time.time()
            self.base_model.train()  # keep stats updating
            logits = self.base_model(x)
            loss = softmax_entropy(logits).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # micro-step budget control (simply halves LR if over budget)
            if self.tau_max is not None:
                elapsed = time.time() - start
                self.timer_ema = 0.8 * (self.timer_ema or elapsed) + 0.2 * elapsed
                if self.timer_ema > self.tau_max:
                    for g in self.optimizer.param_groups:
                        g["lr"] *= 0.5
            # return *post-update* predictions (detached to avoid autograd above caller)
            with torch.no_grad():
                return self.base_model(x)
        else:
            return self.base_model(x)


# ---------------------------------------------------------------------------
# 2) AdaNPC adaptor – our proposed method
# ---------------------------------------------------------------------------
class AdaNPCAdaptor(nn.Module):
    """Adaptive Natural-gradient & Probabilistically-Certified Test-time Adaptation."""

    def __init__(self, base_model: nn.Module, beta: float = 0.99, delta: float = 0.1,
                 tau_max: float | None = None, micro_steps: int = 4):
        super().__init__()
        self.base_model = base_model
        self.beta = beta
        self.delta = delta
        self.tau_max = tau_max
        self.k = micro_steps

        # collect **all** parameters – normaliser agnostic
        self.params: List[nn.Parameter] = [p for p in self.base_model.parameters() if p.requires_grad]
        self.var: torch.Tensor | None = None  # streaming Fisher diag
        self.timer_ema: float | None = None

    def _flatten_grads(self, grads: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([g.contiguous().view(-1) for g in grads])

    def _apply_step(self, step_vec: torch.Tensor, eta: float):
        idx = 0
        for p in self.params:
            numel = p.numel()
            upd = step_vec[idx: idx + numel].view_as(p)
            p.data.sub_(eta * upd)
            idx += numel

    @torch.enable_grad()
    def _adanpc_update(self, x: torch.Tensor):
        logits = self.base_model(x)
        loss = softmax_entropy(logits).mean()
        grads = torch.autograd.grad(loss, self.params, retain_graph=False, create_graph=False)
        g_vec = self._flatten_grads(list(grads)).detach()

        # update streaming Fisher diag (variance proxy)
        if self.var is None:
            self.var = g_vec.pow(2) + 1e-8
        else:
            self.var = self.beta * self.var + (1 - self.beta) * g_vec.pow(2) + 1e-8

        step = g_vec / torch.sqrt(self.var)
        # safety check via Bernstein inequality
        deltaL = (step * g_vec).sum()
        varL = torch.sqrt((step.pow(2) * self.var).sum())
        threshold = -varL * math.sqrt(2 * math.log(1 / self.delta))
        safe = deltaL < threshold  # want expected loss *decrease*

        if safe:
            eta = 1.0 / max(1, self.k)
            self._apply_step(step, eta)
        return logits  # return pre-update prediction for possible monitoring

    def forward(self, x):
        if torch.is_grad_enabled():
            start = time.time()
            self.base_model.train()  # keep norm layers in train mode for stats
            with torch.enable_grad():
                _ = self._adanpc_update(x)
            # compute predictions *after* the potential parameter update
            with torch.no_grad():
                y_after = self.base_model(x)

            # latency-aware micro-step back-off
            if self.tau_max is not None:
                elapsed = time.time() - start
                self.timer_ema = 0.8 * (self.timer_ema or elapsed) + 0.2 * elapsed
                if self.timer_ema > self.tau_max and self.k > 1:
                    self.k //= 2  # halve micro-steps
            return y_after
        else:
            return self.base_model(x)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_model(cfg: Dict[str, Any], num_classes: int):
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "resnet18")
    pretrained = model_cfg.get("pretrained", False)

    base_model = get_backbone(name, num_classes, pretrained=pretrained)

    tta = model_cfg.get("tta", "none").lower()
    if tta == "none":
        return base_model
    if tta == "tent":
        tent_cfg = model_cfg.get("tent", {})
        lr = tent_cfg.get("lr", 1e-3)
        tau_max = tent_cfg.get("tau_max")
        return TentAdaptor(base_model, lr=float(lr) if lr is not None else 1e-3,
                           tau_max=float(tau_max) if tau_max is not None else None)
    if tta == "adanpc":
        ad_cfg = model_cfg.get("adanpc", {})
        beta_raw = ad_cfg.get("beta", 0.99)
        delta_raw = ad_cfg.get("delta", 0.1)
        tau_max_raw = ad_cfg.get("tau_max")
        micro_steps_raw = ad_cfg.get("micro_steps", 4)
        return AdaNPCAdaptor(base_model,
                             beta=float(beta_raw),
                             delta=float(delta_raw),
                             tau_max=float(tau_max_raw) if tau_max_raw is not None else None,
                             micro_steps=int(micro_steps_raw))

    raise ValueError(f"Unsupported TTA method: {tta}")
