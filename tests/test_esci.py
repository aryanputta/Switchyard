from pathlib import Path

from switchyard.esci import ESCI_GAIN, Candidate, load_esci_jsonl

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "esci_sample.jsonl"


def test_gain_mapping():
    assert ESCI_GAIN == {"E": 3, "S": 2, "C": 1, "I": 0}


def test_candidate_gain_from_label():
    assert Candidate("p", "t", label="E").gain == 3
    assert Candidate("p", "t", label="I").gain == 0


def test_candidate_text_leads_with_title_and_brand():
    c = Candidate("p", "Title Here", brand="BrandX", description="desc")
    assert c.text.startswith("Title Here BrandX")


def test_load_sample_is_candidate_reranking():
    queries = load_esci_jsonl(SAMPLE)
    assert len(queries) == 8
    for q in queries:
        # every query carries its own candidate set with graded labels
        assert q.candidates
        assert set(q.relevance) == {c.product_id for c in q.candidates}
        assert any(c.label == "E" for c in q.candidates)


def test_relevance_map_uses_graded_gains():
    queries = load_esci_jsonl(SAMPLE)
    q1 = next(q for q in queries if q.query_id == "q1")
    assert q1.relevance["p11"] == 3  # Exact
    assert q1.relevance["p14"] == 0  # Irrelevant
