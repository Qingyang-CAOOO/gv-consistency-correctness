from gvsmc.cft import build_cft_examples


def _record(index, hard, soft, parsed=True, sampled_hard=None):
    if sampled_hard is None:
        sampled_hard = hard
    return {
        "index": index,
        "parse_success": parsed,
        "hard_weight": hard,
        "sampled_hard_weight": sampled_hard,
        "soft_weight": soft,
        "generator_prompt": f"G{index}",
        "raw_generator_response": f"good{index} || bad{index}",
        "validator_prompt": f"V{index}",
        "target_label": "A" if index % 2 == 0 else "B",
    }


def test_hard_cft_filters_and_targets_r():
    rows = [_record(0, 1.0, 0.9), _record(1, 0.0, 0.2)]
    examples = build_cft_examples(rows, mode="hard")

    assert len(examples) == 2
    assert {example.component for example in examples} == {
        "generator",
        "validator",
    }
    assert all(example.trajectory_index == 0 for example in examples)
    assert all(example.weight == 1.0 for example in examples)
    validator = next(x for x in examples if x.component == "validator")
    assert validator.completion == "A"


def test_sampled_hard_uses_sampled_validator_draw():
    rows = [
        _record(0, 1.0, 0.9, sampled_hard=0.0),
        _record(1, 0.0, 0.2, sampled_hard=1.0),
    ]
    examples = build_cft_examples(rows, mode="sampled-hard")

    assert len(examples) == 2
    assert all(example.trajectory_index == 1 for example in examples)
    assert all(example.weight == 1.0 for example in examples)


def test_soft_cft_keeps_all_parsed_with_same_trajectory_weight():
    rows = [
        _record(0, 1.0, 0.9),
        _record(1, 0.0, 0.2),
        _record(2, 1.0, 0.7, parsed=False),
    ]
    examples = build_cft_examples(rows, mode="soft")

    assert len(examples) == 4
    weights_by_index = {}
    for example in examples:
        weights_by_index.setdefault(example.trajectory_index, set()).add(
            example.weight
        )
    assert weights_by_index == {0: {0.9}, 1: {0.2}}
