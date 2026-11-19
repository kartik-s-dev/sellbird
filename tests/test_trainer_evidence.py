from harvey.trainer import (
    _claim_has_source_quote,
    _filter_evidence_claims,
    _validate_icp_claims,
    _validate_product_claims,
)


def test_exact_source_quote_accepts_product_claim():
    source = {"page": "Zoho CRM helps businesses manage customer relationships."}
    result = _validate_product_claims(
        {
            "evidence": {
                "product_name": {
                    "claim": "Zoho CRM",
                    "quote": "Zoho CRM helps businesses manage customer relationships.",
                }
            }
        },
        source,
    )
    assert result["product_name"] == "Zoho CRM"


def test_missing_source_quote_rejects_product_claim():
    source = {"page": "Zoho CRM helps businesses manage customer relationships."}
    result = _validate_product_claims(
        {
            "evidence": {
                "product_name": {
                    "claim": "Zoho Mail",
                    "quote": "Zoho Mail provides advanced email security.",
                }
            }
        },
        source,
    )
    assert result["product_name"] == ""


def test_missing_evidence_rejects_all_product_claims():
    result = _validate_product_claims(
        {"product_name": "Invented Product"},
        {"page": "Some website text."},
    )
    assert result == {}


def test_product_list_filters_claims_without_matching_quotes():
    source = {"page": "Automate sales workflows and manage customer records."}
    result = _validate_product_claims(
        {
            "evidence": {
                "key_benefits": [
                    {
                        "claim": "Automate sales workflows",
                        "quote": "Automate sales workflows",
                    },
                    {
                        "claim": "Guaranteed 300% revenue growth",
                        "quote": "Guaranteed 300% revenue growth",
                    },
                ]
            }
        },
        source,
    )
    assert result["key_benefits"] == ["Automate sales workflows"]


def test_quote_matching_ignores_case_and_repeated_whitespace():
    assert _claim_has_source_quote(
        "Claim",
        "ZOHO   CRM",
        "Zoho CRM helps businesses.",
    )


def test_quote_matching_rejects_empty_or_malformed_values():
    assert not _claim_has_source_quote("", "some quote", "some quote")
    assert not _claim_has_source_quote("claim", "", "some quote")
    assert not _claim_has_source_quote("claim", None, "some quote")


def test_icp_rejects_unsupported_titles_and_industries():
    source = {"page": "Designed for healthcare organisations."}
    result = _validate_icp_claims(
        {
            "evidence": {
                "industries": [
                    {
                        "claim": "Healthcare",
                        "quote": "Designed for healthcare organisations.",
                    },
                    {
                        "claim": "Retail",
                        "quote": "Designed for retail companies.",
                    },
                ],
                "titles": [
                    {
                        "claim": "Practice Manager",
                        "quote": "Practice Manager",
                    }
                ],
            }
        },
        source,
    )
    assert result["industries"] == ["Healthcare"]
    assert result["titles"] == []


def test_icp_missing_evidence_returns_empty_result():
    result = _validate_icp_claims(
        {"industries": ["Healthcare"], "titles": ["CEO"]},
        {"page": "Some website text."},
    )
    assert result == {}


def test_malformed_list_evidence_is_rejected():
    assert _filter_evidence_claims(
        "not a list",
        "Website text",
    ) == []
