import torch

from gvsmc.metrics import effective_sample_size
from gvsmc.weights import (
    hard_randomized_gv_weight,
    normalize_weights,
    sequential_weight_update,
    soft_randomized_gv_weight_from_label_logprobs,
    soft_randomized_gv_weight_from_signed_logodds,
)


def test_hard_randomized_weight():
    target = torch.tensor([0, 1, 1])
    pred = torch.tensor([0, 0, 1])
    got = hard_randomized_gv_weight(target, pred)
    assert torch.equal(got, torch.tensor([1.0, 0.0, 1.0]))


def test_soft_weight_matches_binary_softmax():
    scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([0, 1])
    got = soft_randomized_gv_weight_from_label_logprobs(scores, target)
    expected = torch.sigmoid(torch.tensor([2.0, 2.0]))
    assert torch.allclose(got, expected)


def test_signed_logodds():
    x = torch.tensor([0.0, 2.0, -2.0])
    got = soft_randomized_gv_weight_from_signed_logodds(x)
    assert torch.allclose(got, torch.sigmoid(x))


def test_ess_uniform_is_particle_count():
    w = torch.ones(8)
    assert torch.allclose(effective_sample_size(w), torch.tensor(8.0))


def test_sequential_update_normalizes():
    old = normalize_weights(torch.tensor([1.0, 1.0, 1.0]))
    pot = torch.tensor([1.0, 0.5, 0.25])
    new = sequential_weight_update(old, pot)
    assert torch.allclose(new.sum(), torch.tensor(1.0))
    assert new[0] > new[1] > new[2]
