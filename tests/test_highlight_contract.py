from agent.grounding import check_report


def test_missing_figures_are_normalised_for_highlighting():
    r = check_report("Costs rise 1,500 euros and 40%.", [], "topic")
    assert "1500" in r["missing"] and "40" in r["missing"]


def test_missing_list_is_capped_but_count_is_full():
    report = " ".join(str(100 + i) for i in range(15))
    r = check_report(report, [], "topic")
    assert len(r["missing"]) == 10 and r["missing_count"] == 15