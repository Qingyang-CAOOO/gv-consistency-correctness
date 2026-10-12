import numpy as np

from gvsmc.fusion import PairwiseFusionModel, clipped_logit


def test_clipped_logit_is_antisymmetric():
    p = 0.73
    assert np.isclose(clipped_logit(1.0 - p), -clipped_logit(p))


def test_zero_increment_preserves_prior():
    model = PairwiseFusionModel(
        baseline_generator_coef=1.0,
        full_generator_coef=1.0,
        validator_coef=0.0,
        baseline_c=1.0,
        full_c=1.0,
    )
    prior = [0.6, 0.3, 0.1]
    pairwise = [
        [None, 0.7, 0.8],
        [0.3, None, 0.6],
        [0.2, 0.4, None],
    ]
    fused = model.fused_weights(prior, pairwise)
    assert np.allclose(fused, np.asarray(prior), atol=1e-7)


def test_validator_increment_can_shift_mass():
    model = PairwiseFusionModel(
        baseline_generator_coef=1.0,
        full_generator_coef=1.0,
        validator_coef=1.0,
        baseline_c=1.0,
        full_c=1.0,
    )
    prior = [0.5, 0.5]
    pairwise = [[None, 0.8], [0.2, None]]
    fused = model.fused_weights(prior, pairwise)
    assert fused[0] > 0.5
    assert fused[1] < 0.5
