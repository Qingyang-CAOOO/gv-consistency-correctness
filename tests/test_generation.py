from gvsmc.generation import (
    GeneratedAnswerPair,
    parse_generated_answer_pair,
    parse_generated_answer_pair_with_mode,
)


def test_parse_generated_answer_pair():
    got = parse_generated_answer_pair("Walter || John")
    assert got == GeneratedAnswerPair("Walter", "John")


def test_parse_generated_answer_pair_uses_first_nonempty_line():
    got = parse_generated_answer_pair("\n  Paris || Rome  \nextra continuation")
    assert got == GeneratedAnswerPair("Paris", "Rome")


def test_parse_generated_answer_pair_rejects_bad_format():
    assert parse_generated_answer_pair("Paris") is None
    assert parse_generated_answer_pair("Paris || Rome || Berlin") is None
    assert parse_generated_answer_pair(" || Rome") is None


def test_parse_generated_answer_pair_strips_echoed_answer_prefix_and_quotes():
    got = parse_generated_answer_pair('Answer:  "South Carolina" || "Virginia"')
    assert got == GeneratedAnswerPair("South Carolina", "Virginia")


def test_parse_generated_answer_pair_skips_standalone_answer_marker():
    pair, mode = parse_generated_answer_pair_with_mode(
        "Answer:\n* The Morning Chronicle || The London Gazette"
    )
    assert pair == GeneratedAnswerPair(
        "The Morning Chronicle", "The London Gazette"
    )
    assert mode == "delimiter_line"


def test_parse_generated_answer_pair_accepts_labeled_lines():
    pair, mode = parse_generated_answer_pair_with_mode(
        "**Correct Answer:** Chevrolet ||\n**Misleading Answer:** Toyota"
    )
    assert pair == GeneratedAnswerPair("Chevrolet", "Toyota")
    assert mode == "labeled_lines"


def test_parse_generated_answer_pair_accepts_labels_without_delimiter():
    pair, mode = parse_generated_answer_pair_with_mode(
        "**Correct:** Ambidextrous\n**Misleading:** Right-handed"
    )
    assert pair == GeneratedAnswerPair("Ambidextrous", "Right-handed")
    assert mode == "labeled_lines"


def test_parse_generated_answer_pair_rejects_incomplete_and_ambiguous():
    pair, mode = parse_generated_answer_pair_with_mode("Answer: Harrods ||")
    assert pair is None
    assert mode == "incomplete_delimiter_pair"

    pair, mode = parse_generated_answer_pair_with_mode(
        'Answer: || "The Cat in the Hat" || "The Wizard of Oz"'
    )
    assert pair is None
    assert mode == "ambiguous_multiple_delimiters"
