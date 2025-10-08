"""src/model.py
Model definitions together with the *AdaNPC* test-time adaptation wrapper plus
all ablation variants required for **exp-2-ablation-sensitivity**.
"""
from __future__ import annotations
import math, time
from typing import Dict, Any

import torch, torch.nn as nn
import torchvision.models as tvm

__all__ = [
    "get_model",
]

# --------------------------- utility functions ------------------------------ #

def get_backbone(backbone_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = backbone_name.lower()
    if name == "resnet18":
        model = tvm.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif name == "resnet50":
        model = tvm.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


# --------------------------- helper functions ------------------------------- #

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample softmax entropy."""
    p = logits.softmax(dim=1)
    return -(p * logits.log_softmax(dim=1)).sum(dim=1)


# ---------------------- AdaNPC Test-Time Adaptation --------------------------#
class AdaNPCAdaptor(nn.Module):
    """Wraps a *base classifier* to perform on-line AdaNPC adaptation at test
    time.  Behaviour can be modified via *variant* keyword to realise all
    ablation modes required by the experiment.
    """

    def __init__(
        self,
        base_model: nn.Module,
        variant: str = "full",  # one of {full,fixed-fisher,no-safety-filter,no-micro-stepping,sgd-adapter}
        beta: float = 0.99,
        delta: float = 0.1,
        tau_max: float | None = None,
        micro_steps: int = 4,
    ):
        super().__init__()
        self.base_model = base_model
        self.variant = variant.lower()
        self.beta = beta
        self.delta = delta
        self.tau_max = tau_max
        self.k_init = max(1, micro_steps)
        self.k = self.k_init  # will be updated by scheduler

        # Flags derived from variant string ---------------------------------
        self.use_streaming_fisher = self.variant not in {"fixed-fisher"}
        self.use_safety = self.variant not in {"no-safety-filter"}
        self.use_micro_schedule = self.variant not in {"no-micro-stepping"}
        self.use_natural_grad = self.variant not in {"sgd-adapter"}

        # Collect parameters (affine of all normalisers + classifier weight
        # for simplicity).  In this reference implementation we *adapt all*
        # parameters.
        self.params: list[nn.Parameter] = [p for p in self.base_model.parameters() if p.requires_grad]

        # State buffers ------------------------------------------------------
        self._var: list[torch.Tensor] | None = None  # diagonal covariance approximation
        self._timer_ema: float | None = None         # for micro-schedule

    # -------------------------------------------------------------------- #
    # Forward interface
    # -------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor):
        if self.training:
            # During *source* training we do NOT adapt.
            return self.base_model(x)
        else:
            return self._adapt(x)

    # -------------------------------------------------------------------- #
    # Internal adaptation routine
    # -------------------------------------------------------------------- #
    @torch.enable_grad()
    def _adapt(self, x: torch.Tensor):
        t0 = time.time()

        # 1) Forward + unsupervised loss (entropy)
        logits = self.base_model(x)
        loss = softmax_entropy(logits).mean()

        # 2) Compute gradients wrt *all* trainable params
        grads = torch.autograd.grad(loss, self.params, retain_graph=False, create_graph=False)

        # 3) Maintain (or not) streaming Fisher diagonal
        if self._var is None:
            self._var = [g.detach().pow(2) + 1e-8 for g in grads]
        else:
            if self.use_streaming_fisher:
                for v, g in zip(self._var, grads):
                    v.mul_(self.beta).add_(g.detach().pow(2), alpha=(1 - self.beta))
            # else: keep _var frozen after first batch

        # 4) Build update step ------------------------------------------------
        steps: list[torch.Tensor] = []
        for g, v in zip(grads, self._var):
            if self.use_natural_grad:
                steps.append(g / v.sqrt())
            else:  # SGD adapter – no pre-conditioning
                steps.append(g)

        # 5) Safety filter ---------------------------------------------------
        if self.use_safety:
            deltaL_num = torch.stack([(s * g).sum() for s, g in zip(steps, grads)]).sum()
            var_proxy = torch.stack([(s.pow(2) * (v if self.use_natural_grad else 1.0)).sum() for s, v in zip(steps, self._var)]).sum()
            bound = deltaL_num + var_proxy.sqrt() * math.sqrt(2 * math.log(1.0 / self.delta))
            safe = bound < 0
        else:
            safe = True

        # 6) Apply update (possibly micro-stepped) ---------------------------
        if safe:
            eta = 1.0 / max(1, self.k)
            for _ in range(self.k):
                for p, s in zip(self.params, steps):
                    p.data.sub_(eta * s)
        # else: reject update entirely

        # 7) Micro-step scheduler -------------------------------------------
        if self.use_micro_schedule and self.tau_max is not None:
            elapsed = time.time() - t0
            self._timer_ema = 0.8 * (self._timer_ema or elapsed) + 0.2 * elapsed
            if self._timer_ema > self.tau_max and self.k > 1:
                self.k //= 2

        return logits.detach()  # prediction BEFORE adaptation step


# ------------------------- public factory API ------------------------------- #

def get_model(cfg: Dict[str, Any], num_classes: int):
    model_cfg = cfg.get("model", {})
    backbone_name = model_cfg.get("name", "resnet18")
    pretrained = bool(model_cfg.get("pretrained", False))

    base_model = get_backbone(backbone_name, num_classes, pretrained=pretrained)

    if model_cfg.get("tta", "none").lower() == "adanpc":
        adanpc_cfg = model_cfg.get("adanpc", {})
        variant = model_cfg.get("variant", "full")
        model = AdaNPCAdaptor(
            base_model=base_model,
            variant=variant,
            beta=adanpc_cfg.get("beta", 0.99),
            delta=adanpc_cfg.get("delta", 0.1),
            tau_max=adanpc_cfg.get("tau_max"),
            micro_steps=adanpc_cfg.get("micro_steps", 4),
        )
        return model
    else:
        return base_model
