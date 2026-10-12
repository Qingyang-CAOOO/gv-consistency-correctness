from gvsmc.evaluation import matches_any_gold, normalize_qa_answer


def test_normalize_qa_answer():
    assert normalize_qa_answer("The Keanu Reeves!") == "keanu reeves"


def test_matches_any_gold():
    gold = ["Keanu Reeves", "Keanu Charles Reeves"]
    assert matches_any_gold("the Keanu Reeves", gold)
    assert not matches_any_gold("Tom Cruise", gold)
