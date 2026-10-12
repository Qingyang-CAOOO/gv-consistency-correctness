import torch

from gvsmc.sequential import posterior_mass, sequential_importance_update, top_particle_index


def test_sequential_importance_update_multiplies_and_normalizes():
    weights = torch.tensor([0.5, 0.5])
    potentials = torch.tensor([0.8, 0.2])
    updated = sequential_importance_update(weights, potentials)
    assert torch.allclose(updated, torch.tensor([0.8, 0.2]))


def test_posterior_mass_and_top_particle():
    weights = torch.tensor([0.1, 0.7, 0.2])
    mask = torch.tensor([False, True, True])
    assert torch.isclose(posterior_mass(mask, weights), torch.tensor(0.9))
    assert top_particle_index(weights) == 1
