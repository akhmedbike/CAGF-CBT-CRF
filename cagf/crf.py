from __future__ import annotations
import torch
import torch.nn as nn
NEG_INF = -1000000000.0

class LinearChainCRF(nn.Module):

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _score_sentence(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[torch.arange(batch_size), 0, tags[:, 0]]
        for t in range(1, seq_len):
            emit = emissions[torch.arange(batch_size), t, tags[:, t]]
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (trans + emit) * mask[:, t]
        seq_lens = mask.sum(dim=1).long()
        last_tags = tags[torch.arange(batch_size), seq_lens - 1]
        score = score + self.end_transitions[last_tags]
        return score

    def _forward_algorithm(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, seq_len):
            emit = emissions[:, t].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            broadcast_alpha = alpha.unsqueeze(2)
            next_alpha = torch.logsumexp(broadcast_alpha + trans + emit, dim=1)
            step_mask = mask[:, t].unsqueeze(1)
            alpha = torch.where(step_mask.bool(), next_alpha, alpha)
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gold_score = self._score_sentence(emissions, tags, mask.float())
        log_z = self._forward_algorithm(emissions, mask.float())
        # Normalise by the TOTAL number of valid tokens across the batch, not by
        # batch size. The sentence-level score (log_z - gold_score) is a SUM of
        # per-position log-likelihoods, so dividing by batch_size alone leaves
        # the loss magnitude ~seq_len times larger than per-token CE -- which
        # starves the lemma/grammeme heads through the global clip_grad_norm.
        # Dividing by total tokens makes the CRF loss directly comparable in
        # scale to the per-token cross-entropy used by the other heads.
        n_tokens = mask.float().sum().clamp(min=1.0)
        return (log_z - gold_score).sum() / n_tokens

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        batch_size, seq_len, num_tags = emissions.shape
        mask_bool = mask.bool()
        history = []
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)
            trans = self.transitions.unsqueeze(0)
            next_score = broadcast_score + trans
            best_score, best_idx = next_score.max(dim=1)
            best_score = best_score + emissions[:, t]
            step_mask = mask_bool[:, t].unsqueeze(1)
            score = torch.where(step_mask, best_score, score)
            history.append(best_idx)
        score = score + self.end_transitions.unsqueeze(0)
        seq_lens = mask_bool.sum(dim=1)
        best_paths = []
        for b in range(batch_size):
            length = int(seq_lens[b].item())
            if length == 0:
                best_paths.append([])
                continue
            _, best_last = score[b].max(dim=0)
            best_last = int(best_last.item())
            path = [best_last]
            for t in range(length - 2, -1, -1):
                best_last = int(history[t][b, best_last].item())
                path.append(best_last)
            path.reverse()
            best_paths.append(path)
        return best_paths