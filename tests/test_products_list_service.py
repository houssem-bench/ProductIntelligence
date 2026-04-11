from __future__ import annotations

import asyncio

from app.models.schemas import ProductAnalysis
from app.services.products_list_service import ProductsListService


class FakeGrokService:
    def __init__(self, payload):
        self.payload = payload

    async def enrich_products_list(self, *, products_list_payload, analyses):
        return self.payload


def test_products_list_service_builds_base_payload_when_no_grok_enrichment():
    service = ProductsListService(FakeGrokService(payload=None))
    analyses = [
        ProductAnalysis(
            product_id="1",
            source="barcode",
            confidence=0.9,
            category="food",
            name="Product A",
            brand="Brand A",
            ingredients=["sulfate", "formaldehyde"],
            additives=[],
            debug={},
        )
    ]

    payload = asyncio.run(service.build_products_list(analyses))

    assert "products_list" in payload
    assert len(payload["products_list"]) == 1

    product = payload["products_list"][0]
    assert product["product_id"] == "1"
    assert product["product_usage"] == "food"
    assert product["exposure_type"] == "oral"
    assert product["ingredient_list"] == [
        {"name": "sulfate"},
        {"name": "formaldehyde"},
    ]


def test_products_list_service_merges_only_non_empty_enrichment_fields():
    enriched_payload = {
        "products_list": [
            {
                "product_id": "1",
                "product_usage": "cosmetics",
                "exposure_type": "skin",
                "ingredient_list": [
                    {
                        "name": "sulfate",
                        "code": "CAS223",
                        "dose": "2%",
                        "product_prevalence": "Found in 85% of shampoos",
                        "additional_info": "Irritant at high concentration",
                    },
                    {
                        "name": "formaldehyde",
                        "code": "",
                    },
                ],
            }
        ]
    }

    service = ProductsListService(FakeGrokService(payload=enriched_payload))
    analyses = [
        ProductAnalysis(
            product_id="1",
            source="lens",
            confidence=0.75,
            category="cosmetic",
            name="Product B",
            brand="Brand B",
            ingredients=["sulfate", "formaldehyde"],
            additives=[],
            debug={},
        )
    ]

    payload = asyncio.run(service.build_products_list(analyses))
    product = payload["products_list"][0]

    assert product["product_usage"] == "cosmetics"
    assert product["exposure_type"] == "skin"
    assert product["ingredient_list"][0]["name"] == "sulfate"
    assert product["ingredient_list"][0]["code"] == "CAS223"
    assert product["ingredient_list"][0]["dose"] == "2%"
    assert product["ingredient_list"][0]["product_prevalence"] == "Found in 85% of shampoos"
    assert product["ingredient_list"][0]["additional_info"] == "Irritant at high concentration"
    assert product["ingredient_list"][1] == {"name": "formaldehyde"}


def test_products_list_service_skips_products_without_ingredients():
    service = ProductsListService(FakeGrokService(payload=None))
    analyses = [
        ProductAnalysis(
            product_id="1",
            source="failed",
            confidence=0.0,
            category="unknown",
            name="Product C",
            brand=None,
            ingredients=[],
            additives=[],
            debug={},
        )
    ]

    payload = asyncio.run(service.build_products_list(analyses))

    assert payload == {"products_list": []}
