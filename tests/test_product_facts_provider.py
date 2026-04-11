from __future__ import annotations

from app.providers.product_facts_provider import ProductFactsProvider


def test_prepare_name_candidates_strips_packaging_noise():
    candidates = ProductFactsProvider._prepare_name_candidates("Coca Cola Zero 1.5L (Pack of 6)")

    assert candidates[0] == "Coca Cola Zero 1.5L (Pack of 6)"
    assert "Coca Cola Zero" in candidates


def test_prepare_name_candidates_keeps_short_name_as_is():
    candidates = ProductFactsProvider._prepare_name_candidates("Nutella")

    assert candidates == ["Nutella"]
