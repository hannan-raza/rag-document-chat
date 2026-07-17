"""Tests for the Module 2 dataset-picker (app.datasets.pick_datasets).

Two tiers:
- Unit tests mock the catalog + LLM, so they're fast/deterministic/free and
  pin down the DEFENSIVE PARSING and PER-USER ISOLATION logic.
- One integration test hits the real LLM with a fake in-memory catalog (no DB).
  It's the regression guard for the compound-question bug: a question mixing a
  data part with a non-data part must still pick the relevant dataset. Only a
  real model exercises the prompt, so a mock could never have caught it.
  Auto-skipped when OPENAI_API_KEY is unset.
"""
import os
import pytest

import app.datasets as ds

CATALOG = [
    {"id": 2, "source": "products.csv",
     "columns": ["order_id", "customer_name", "city", "product", "quantity", "price"]},
    {"id": 5, "source": "employees.csv",
     "columns": ["emp_id", "name", "department", "salary"]},
]


@pytest.fixture
def fake_catalog(monkeypatch):
    """pick_datasets sees a fixed two-dataset catalog owned by the user; no DB."""
    monkeypatch.setattr(ds, "list_datasets_meta", lambda user_id: CATALOG)


def _mock_reply(monkeypatch, reply):
    monkeypatch.setattr(ds, "generate", lambda *a, **k: reply)


# --- unit: defensive parsing + isolation (mocked LLM) ----------------------

def test_clean_ids_are_parsed(fake_catalog, monkeypatch):
    _mock_reply(monkeypatch, "2, 5")
    assert ds.pick_datasets("u", "q") == [2, 5]


def test_none_reply_returns_empty(fake_catalog, monkeypatch):
    _mock_reply(monkeypatch, "none")
    assert ds.pick_datasets("u", "q") == []


def test_unowned_ids_are_dropped(fake_catalog, monkeypatch):
    # 99 isn't the user's; 5 and 2 are. 99 must be discarded (isolation guard).
    _mock_reply(monkeypatch, "99, 5, 2")
    assert ds.pick_datasets("u", "q") == [5, 2]


def test_order_preserved_and_deduped(fake_catalog, monkeypatch):
    _mock_reply(monkeypatch, "5, 2, 2, 5")
    assert ds.pick_datasets("u", "q") == [5, 2]


def test_chatty_reply_yields_empty(fake_catalog, monkeypatch):
    # a non-numeric/explanatory reply must not smuggle an id through
    _mock_reply(monkeypatch, "I think dataset 2 is relevant")
    assert ds.pick_datasets("u", "q") == []


def test_empty_catalog_short_circuits(monkeypatch):
    # user owns nothing -> [] and the LLM is never called
    monkeypatch.setattr(ds, "list_datasets_meta", lambda user_id: [])

    def _boom(*a, **k):
        raise AssertionError("generate() must not be called for an empty catalog")

    monkeypatch.setattr(ds, "generate", _boom)
    assert ds.pick_datasets("u", "q") == []


# --- integration: real LLM, fake catalog (regression for the compound bug) --

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"),
                    reason="needs a real LLM provider (OPENAI_API_KEY)")
def test_compound_question_still_picks_dataset(fake_catalog):
    # "who founded the company" (non-data) + "which product..." (data).
    # Regression: the picker used to answer 'none' for compound questions.
    q = "Who founded the company, and which product generates the most revenue?"
    picked = ds.pick_datasets("u", q)
    assert 2 in picked, f"expected products.csv (id 2) to be picked, got {picked}"
