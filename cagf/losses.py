from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyWeightedLoss(nn.Module):

    def __init__(self, num_tasks: int=3, init_log_var: float=0.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_log_var))

    def forward(self, losses: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        if len(losses) != self.log_vars.numel():
            raise ValueError(f'expected {self.log_vars.numel()} losses, got {len(losses)}')
        total = torch.zeros((), device=losses[0].device)
        weights = {}
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
            weights[f'weight_{i}'] = float(precision.item())
        return (total, weights)

def masked_multilabel_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_token = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    per_token = per_token.mean(dim=-1)
    mask_f = mask.float()
    return (per_token * mask_f).sum() / mask_f.sum().clamp(min=1.0)

def masked_lemma_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100)