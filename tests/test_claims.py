from agent.grounding import check_claims

NOTES = [{"question": "q", "answer": "Startups must classify each AI system into a risk tier before launching products in Europe."}]


def test_supported_sentence_not_flagged():
    r = check_claims("Startups must classify every AI system into a risk tier before launching.", NOTES, "EU AI Act")
    assert r["sentences"] == 1 and r["weak_count"] == 0


def test_drifting_sentence_is_flagged():
    r = check_claims("Quantum batteries delivered fantastic revenue growth across Mongolian markets yesterday.", NOTES, "EU AI Act")
    assert r["weak_count"] == 1 and "Quantum" in r["weak"][0]


def test_gap_statements_and_short_sentences_are_skipped():
    r = check_claims("The notes do not provide concrete case studies of sandbox success.\nShort one here.", NOTES, "t")
    assert r["sentences"] == 0


def test_headings_and_bullets_are_cleaned():
    r = check_claims("## Overview\n- Startups must classify every AI system into a risk tier before launching.", NOTES, "t")
    assert r["sentences"] == 1


def test_never_raises():
    assert check_claims(None, None, None, None)["sentences"] == 0