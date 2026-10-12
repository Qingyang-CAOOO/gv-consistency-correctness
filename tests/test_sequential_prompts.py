from gvsmc.prompts import (
    num_triviaqa_correctness_templates,
    triviaqa_correctness_validator_prompt,
)


def test_correctness_prompt_randomizes_correct_label():
    prompt_a = triviaqa_correctness_validator_prompt(
        "Who wrote Hamlet?",
        "William Shakespeare",
        target_index=0,
        template_index=0,
    )
    prompt_b = triviaqa_correctness_validator_prompt(
        "Who wrote Hamlet?",
        "William Shakespeare",
        target_index=1,
        template_index=0,
    )
    assert "A: The candidate answer is correct." in prompt_a
    assert "B: The candidate answer is incorrect." in prompt_a
    assert "A: The candidate answer is incorrect." in prompt_b
    assert "B: The candidate answer is correct." in prompt_b
    assert num_triviaqa_correctness_templates() >= 4
