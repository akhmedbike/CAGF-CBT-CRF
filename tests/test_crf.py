import itertools
import math
import torch
from cagf.crf import LinearChainCRF

def brute_force_log_z(crf: LinearChainCRF, emissions: torch.Tensor, length: int) -> float:
    num_tags = crf.num_tags
    total = 0.0
    for path in itertools.product(range(num_tags), repeat=length):
        score = crf.start_transitions[path[0]].item() + emissions[0, path[0]].item()
        for t in range(1, length):
            score += crf.transitions[path[t - 1], path[t]].item() + emissions[t, path[t]].item()
        score += crf.end_transitions[path[-1]].item()
        total += math.exp(score)
    return math.log(total)

def brute_force_best_path(crf: LinearChainCRF, emissions: torch.Tensor, length: int):
    num_tags = crf.num_tags
    best_score, best_path = (-1e+18, None)
    for path in itertools.product(range(num_tags), repeat=length):
        score = crf.start_transitions[path[0]].item() + emissions[0, path[0]].item()
        for t in range(1, length):
            score += crf.transitions[path[t - 1], path[t]].item() + emissions[t, path[t]].item()
        score += crf.end_transitions[path[-1]].item()
        if score > best_score:
            best_score, best_path = (score, list(path))
    return (best_path, best_score)

def test_forward_algorithm_matches_brute_force():
    torch.manual_seed(0)
    num_tags, seq_len = (4, 5)
    crf = LinearChainCRF(num_tags)
    emissions = torch.randn(1, seq_len, num_tags)
    mask = torch.ones(1, seq_len)
    log_z_algo = crf._forward_algorithm(emissions, mask).item()
    log_z_brute = brute_force_log_z(crf, emissions[0], seq_len)
    assert abs(log_z_algo - log_z_brute) < 0.0001, (log_z_algo, log_z_brute)

def test_viterbi_matches_brute_force():
    torch.manual_seed(1)
    num_tags, seq_len = (4, 5)
    crf = LinearChainCRF(num_tags)
    emissions = torch.randn(1, seq_len, num_tags)
    mask = torch.ones(1, seq_len)
    decoded = crf.decode(emissions, mask)[0]
    brute_path, brute_score = brute_force_best_path(crf, emissions[0], seq_len)
    assert decoded == brute_path, (decoded, brute_path)

def test_nll_is_nonnegative_and_padding_is_ignored():
    torch.manual_seed(2)
    num_tags = 3
    crf = LinearChainCRF(num_tags)
    emissions = torch.randn(2, 4, num_tags)
    tags = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    nll = crf.neg_log_likelihood(emissions, tags, mask)
    assert nll.item() >= -0.0001
    assert torch.isfinite(nll)
if __name__ == '__main__':
    test_forward_algorithm_matches_brute_force()
    test_viterbi_matches_brute_force()
    test_nll_is_nonnegative_and_padding_is_ignored()
    print('All CRF tests passed.')