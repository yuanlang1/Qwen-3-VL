import json

import pytest

from sportsqa_offline_sampling.query_bank import QueryBankGenerator


def test_query_bank_parser_accepts_exactly_seven_distinct_queries():
    payload = {
        "sport_type": "basketball",
        "sport_type_confidence": "high",
        "query_bank": [f"find visible evidence number {index}" for index in range(7)],
    }
    assert QueryBankGenerator._parse(json.dumps(payload))["query_bank"] == payload["query_bank"]


def test_query_bank_parser_rejects_duplicate_or_invalid_unknown_confidence():
    duplicate = {
        "sport_type": "basketball",
        "sport_type_confidence": "high",
        "query_bank": ["same query"] * 7,
    }
    with pytest.raises(ValueError):
        QueryBankGenerator._parse(json.dumps(duplicate))
    unknown = {
        "sport_type": "unknown",
        "sport_type_confidence": "medium",
        "query_bank": [f"query {index}" for index in range(7)],
    }
    with pytest.raises(ValueError):
        QueryBankGenerator._parse(json.dumps(unknown))
