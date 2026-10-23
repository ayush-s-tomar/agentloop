from agent.grounding import check_report, extract_numbers


def note(answer, question="q"):
    return {"question": question, "answer": answer}


def test_grounded_numbers_pass():
    r = check_report("Adoption grew 46% in 2025.", [note("Developers shipped 46% more code in 2025.")], "topic")
    assert r["checked"] == 2 and r["missing_count"] == 0


def test_invented_number_is_flagged():
    r = check_report("Costs fall 40% by 2030.", [note("The target year is 2030.")], "topic")
    assert r["missing"] == ["40"] and r["missing_count"] == 1


def test_small_integers_are_ignored():
    r = check_report("Three of the 5 firms agreed.", [], "topic")
    assert r["checked"] == 0


def test_thousands_separator_is_normalised():
    r = check_report("Budget of USD 1,000 per unit.", [note("Each unit costs 1000 dollars.")], "topic")
    assert r["checked"] == 1 and r["missing_count"] == 0


def test_list_numbering_is_not_a_figure():
    assert extract_numbers("## 1. Overview\n2. Next step") == []


def test_decimals_are_kept():
    assert extract_numbers("USD 0.05 per call") == ["0.05"]


def test_never_raises_on_bad_input():
    r = check_report(None, None, "", None)
    assert r["checked"] == 0 and r["missing_count"] == 0