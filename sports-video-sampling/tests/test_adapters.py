import json

import pytest

from sports_video_sampling.adapters.qwen import convert_qwen
from sports_video_sampling.cli import main
from sports_video_sampling.records import load_records


def test_two_dataset_formats_normalize_to_the_same_core(tmp_path, capsys):
    variants = {
        "sportsqa": {"qa_id": 7, "video": "clip.mp4", "question": "Who has the ball?", "answer": "SECRET", "ans_cls": 123},
        "qwen": {"id": 7, "video": ["clip.mp4"], "conversations": [
            {"from": "human", "value": "<video>\nWho has the ball?"},
            {"from": "gpt", "value": "SECRET"},
        ]},
    }
    records = []
    for format_name, row in variants.items():
        source, output = tmp_path / f"{format_name}.json", tmp_path / f"{format_name}.jsonl"
        source.write_text(json.dumps([row]), encoding="utf-8")
        main(["normalize", "--format", format_name, "--input", str(source), "--output", str(output)])
        assert "SECRET" not in output.read_text(encoding="utf-8")
        records.append(load_records(output)[0])
    assert [(record.id, record.video, record.question) for record in records] == [("7", "clip.mp4", "Who has the ball?")] * 2


@pytest.mark.parametrize("changes", [
    {"video": ["a.mp4", "b.mp4"]},
    {"image": "photo.png"},
    {"conversations": [{"from": "human", "value": "one"}, {"from": "human", "value": "two"}]},
    {"conversations": [{"from": "human", "value": "<video><video> question"}]},
])
def test_qwen_adapter_rejects_ambiguous_context(changes):
    row = {"id": "x", "video": "a.mp4", "conversations": [{"from": "human", "value": "<video> question"}]}
    with pytest.raises(ValueError):
        convert_qwen({**row, **changes})


def test_normalize_rejects_extension_that_would_be_misread_as_an_array(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match=".jsonl"):
        main(["normalize", "--format", "sportsqa", "--input", str(source), "--output", str(tmp_path / "output.json")])
