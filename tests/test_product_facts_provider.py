from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.providers.product_facts_provider import ProductFactsProvider


class FakeHttpClient:
    def __init__(self, *, off_data=None, obf_data=None):
        self.off_data = off_data
        self.obf_data = obf_data
        self.urls: list[str] = []

    async def get_json(self, url: str, params=None, headers=None):
        self.urls.append(url)
        if "openfoodfacts" in url:
            return self.off_data
        if "openbeautyfacts" in url:
            return self.obf_data
        return None


class FakeCache:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: dict):
        self._store[key] = value


def test_prepare_name_candidates_strips_packaging_noise():
    candidates = ProductFactsProvider._prepare_name_candidates("Coca Cola Zero 1.5L (Pack of 6)")

    assert candidates[0] == "Coca Cola Zero"
    assert "Coca Cola Zero" in candidates


def test_prepare_name_candidates_keeps_short_name_as_is():
    candidates = ProductFactsProvider._prepare_name_candidates("Nutella")

    assert candidates == ["Nutella"]


def test_prepare_name_candidates_strips_buy_marketplace_noise():
    candidates = ProductFactsProvider._prepare_name_candidates("Tomate SICAM ElsaBuy")

    assert candidates[0] == "Tomate SICAM"
    assert all("buy" not in candidate.lower() for candidate in candidates)


def test_prepare_name_candidates_strips_dash_by_connector_noise():
    candidates = ProductFactsProvider._prepare_name_candidates("TOMATE SICAM - by")

    assert candidates[0] == "TOMATE SICAM"
    assert all(" by" not in candidate.lower() for candidate in candidates)


def test_prepare_name_candidates_trims_text_after_separator_dash():
    candidates = ProductFactsProvider._prepare_name_candidates("Chocolate Biscuits MAJOR - Ayshek")

    assert candidates[0] == "Chocolate Biscuits MAJOR"
    assert all("ayshek" not in candidate.lower() for candidate in candidates)


def test_prepare_name_candidates_strips_nestle_noise_token():
    candidates = ProductFactsProvider._prepare_name_candidates("Chocolate Biscuits MAJOR Nestle")

    assert candidates[0] == "Chocolate Biscuits MAJOR"
    assert all("nestle" not in candidate.lower() for candidate in candidates)


def test_prepare_name_candidates_strips_gr_and_g_weight_tokens():
    candidates = ProductFactsProvider._prepare_name_candidates("TOMATE SICAM 400GR")

    assert candidates[0] == "TOMATE SICAM"
    assert all("400gr" not in candidate.lower() for candidate in candidates)

    candidates_g = ProductFactsProvider._prepare_name_candidates("TOMATE SICAM 400g")
    assert candidates_g[0] == "TOMATE SICAM"


def test_fetch_by_name_food_category_skips_obf():
    off_payload = {
        "products": [
            {
                "product_name": "Nutella",
                "brands": "Ferrero",
                "ingredients": [{"text": "sugar", "id": "en:sugar"}],
            }
        ]
    }
    http = FakeHttpClient(off_data=off_payload, obf_data={"products": []})
    provider = ProductFactsProvider(http, SimpleNamespace(off_user_agent="test"), FakeCache())

    result = asyncio.run(provider.fetch_by_name("Nutella", preferred_category="food"))

    assert result is not None
    assert result.category == "food"
    assert any("openfoodfacts" in url for url in http.urls)
    assert all("openbeautyfacts" not in url for url in http.urls)


def test_fetch_by_name_cosmetic_category_skips_off():
    obf_payload = {
        "products": [
            {
                "product_name": "Shampoo X",
                "brands": "BrandX",
                "ingredients": [{"text": "aqua"}],
            }
        ]
    }
    http = FakeHttpClient(off_data={"products": []}, obf_data=obf_payload)
    provider = ProductFactsProvider(http, SimpleNamespace(off_user_agent="test"), FakeCache())

    result = asyncio.run(provider.fetch_by_name("Shampoo X", preferred_category="cosmetic"))

    assert result is not None
    assert result.category == "cosmetic"
    assert any("openbeautyfacts" in url for url in http.urls)
    assert all("openfoodfacts" not in url for url in http.urls)


def test_pick_best_product_prefers_name_relevance_over_ingredient_count():
    provider = ProductFactsProvider(FakeHttpClient(), SimpleNamespace(off_user_agent="test"), FakeCache())
    products = [
        {
            "product_name": "Sidi Ali Eau Minerale",
            "brands": "Sidi Ali",
            "ingredients": [{"text": "water"}] * 12,
        },
        {
            "product_name": "Thon entier a l'huile vegetale",
            "brands": "Sidi Ali",
            "ingredients": [{"text": "thon"}, {"text": "huile"}, {"text": "sel"}],
        },
    ]

    best = provider._pick_best_product(products, query="Thon entier Sidi Ali")

    assert best["product_name"] == "Thon entier a l'huile vegetale"
