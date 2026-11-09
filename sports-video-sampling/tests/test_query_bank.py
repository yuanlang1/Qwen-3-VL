import json
from types import SimpleNamespace

import pytest
from PIL import Image

from sports_video_sampling.query_bank import QueryBankGenerator, select_query_prefix


def bank():
    return {"sport_type": "basketball", "sport_type_confidence": "high", "query_bank": [f"visible evidence {index}" for index in range(7)], "focus_nouns": ["shooter", "defender's hand"]}


def test_prompt_preserves_sports_rule_evidence_and_visible_only_sport_inference():
    prompt = QueryBankGenerator.build_prompt("Was contact made before the shot?")
    for text in ("sports-video question answering", "only visible evidence", "filenames", "answer options", "metadata", "referee responses", "potential contact", "legality judgments", "\"focus_nouns\""):
        assert text in prompt
    assert "Was contact made before the shot?" in prompt


@pytest.mark.parametrize("change", [
    {"sport_type": "unknown", "sport_type_confidence": "high"},
    {"sport_type_confidence": "certain"},
    {"query_bank": ["same"] * 7},
    {"query_bank": ["a"] * 6},
    {"query_bank": [None] + [f"q {i}" for i in range(6)]},
    {"query_bank": "1234567"},
    {"query_bank": ["word " * 21] + [f"q {i}" for i in range(6)]},
])
def test_response_rejects_invalid_banks(change):
    with pytest.raises(ValueError):
        QueryBankGenerator.parse_response(json.dumps({**bank(), **change}))


def test_unknown_low_and_markdown_json_are_accepted():
    payload = {**bank(), "sport_type": "unknown", "sport_type_confidence": "low"}
    assert QueryBankGenerator.parse_response("```json\n" + json.dumps(payload) + "\n```") == payload
    assert select_query_prefix(payload["query_bank"], 3) == payload["query_bank"][:3]
    with pytest.raises(ValueError):
        select_query_prefix(payload["query_bank"], 2)


@pytest.mark.parametrize("nouns", [None, "basketball", [], ["ball"], [f"player {i}" for i in range(9)],
                                    ["ball", None], ["ball", " "], ["  Shot Clock ", "shot  clock"],
                                    ["ball", "one two three four five"], ["ball", "123"]])
def test_invalid_focus_nouns_are_rejected(nouns):
    with pytest.raises(ValueError, match="focus_nouns"):
        QueryBankGenerator.parse_response(json.dumps({**bank(), "focus_nouns": nouns}))


def test_generate_sends_only_question_and_previews_and_retries_invalid_response(monkeypatch):
    generator = object.__new__(QueryBankGenerator)
    generator.model, generator.max_retries = "test-model", 2
    calls = []

    valid = {**bank(), "focus_nouns": [" shooter ", "defender's hand"]}
    missing = bank()
    del missing["focus_nouns"]

    def create(**kwargs):
        calls.append(kwargs)
        response = json.dumps(missing if len(calls) == 1 else valid)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])

    generator.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("sports_video_sampling.query_bank.time.sleep", lambda _: None)
    result = generator.generate_from_previews("Who has the ball?", [Image.new("RGB", (8, 8))], [5])
    assert result["preview_frame_indices"] == [5]
    assert result["focus_nouns"] == ["shooter", "defender's hand"]
    assert len(calls) == 2
    content = calls[-1]["messages"][0]["content"]
    assert len(content) == 3
    assert content[0]["text"] == QueryBankGenerator.build_prompt("Who has the ball?")
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
