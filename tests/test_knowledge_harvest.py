from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from knowledge_harvest import consolidate, extract  # noqa: E402
from knowledge_harvest.forum import (  # noqa: E402
    DiscourseForum,
    cutoff_datetime,
    html_to_text,
    previous_markdown,
)


class FakeHttp:
    def __init__(self, routes):
        self.routes = routes

    def get_json(self, url):
        for key, value in self.routes.items():
            if url.endswith(key):
                return value
        raise AssertionError(f"unexpected url {url}")


def _post(number, created, cooked, version=1, post_id=None):
    return {
        "id": post_id or 100 + number, "post_number": number, "post_type": 1,
        "created_at": created, "cooked": cooked, "version": version,
        "username": f"u{number}", "staff": False, "actions_summary": [],
    }


def test_html_to_text_keeps_code_and_list_structure():
    text = html_to_text("<p>Use <b>20000</b> trees</p><ul><li>lr 0.001</li></ul><pre><code>a = 1\nb = 2</code></pre>")
    assert "Use 20000 trees" in text
    assert "- lr 0.001" in text
    assert "a = 1\nb = 2" in text


def test_previous_markdown_reads_pre_edit_body():
    revision = {"body_changes": {"side_by_side_markdown":
        '<table class="markdown"><tr><td class="--previous">old <del>corr 0.02</del> &amp; more</td>'
        '<td class="--current">new <ins>corr 0.03</ins></td></tr></table>'}}
    assert previous_markdown(revision) == "old corr 0.02 & more"


def test_topic_keeps_only_text_from_before_cutoff():
    cutoff = cutoff_datetime("2024-01-19")
    topic = {
        "id": 7, "slug": "t", "title": "Targets", "created_at": "2023-01-01T00:00:00Z",
        "post_stream": {"stream": [101, 102, 103], "posts": [
            _post(1, "2023-01-01T00:00:00Z", "<p>first</p>"),
            _post(2, "2023-02-01T00:00:00Z", "<p>edited later</p>", version=2),
            _post(3, "2024-02-01T00:00:00Z", "<p>after cutoff</p>"),
        ]},
    }
    revision = {"created_at": "2024-03-01T00:00:00Z", "body_changes": {"side_by_side_markdown":
        '<td class="--previous">as of cutoff</td><td class="--current">edited later</td>'}}
    forum = DiscourseForum("https://f", FakeHttp({"/t/7.json": topic, "/posts/102/revisions/2.json": revision}))

    document = forum.topic_document(7, cutoff, "tournament")

    assert [p["text"] for p in document["posts"]] == ["first", "as of cutoff"]
    assert document["posts"][1]["text_status"] == "rolled_back"


def test_verify_item_drops_quotes_not_in_source_and_mismatched_numbers():
    item = {"category": "targets", "evidence": [
        {"quote": "mean corr 0.0284 and sharpe 1.142", "value": "0.0284", "setting": "validation"},
        {"quote": "corr 0.05 live", "value": "0.05", "setting": "live"},
        {"quote": "mean corr 0.0284 and sharpe 1.142", "value": "0.9", "setting": "validation"},
    ], "counter_evidence": [{"quote": "invented objection"}]}

    verified = extract.verify_item(item, "Result: mean  corr 0.0284 and Sharpe 1.142 on validation.")

    assert [ev["value"] for ev in verified["evidence"]] == ["0.0284"]
    assert verified["evidence_dropped_unverified"] == 2
    assert verified["counter_evidence"] == []


def test_grade_rules():
    def item(author, setting=None, staff=False, stance="recommend", counter=()):
        evidence = [{"setting": setting}] if setting else []
        return {"authors": [author], "evidence": evidence, "staff": staff, "stance": stance,
                "counter_evidence": list(counter), "doc_id": author}

    assert consolidate.grade([item("a", "live")])["grade"] == "B"
    assert consolidate.grade([item("a", "live"), item("b", "validation")])["grade"] == "A"
    assert consolidate.grade([item("a", "validation"), item("b", "backtest")])["grade"] == "A"
    assert consolidate.grade([item("a", "validation")])["grade"] == "B"
    assert consolidate.grade([item("a", "in_sample", staff=True)])["grade"] == "C"
    assert consolidate.grade([item("a", "in_sample")])["grade"] == "D"
    assert consolidate.grade([item("a"), item("b", stance="caution")])["disputed"] is True


def test_render_splits_long_documents_into_chunks_with_header():
    document = {"title": "T", "url": "u", "created_at": "2023-01-01T00:00:00Z", "posts": [
        {"post_number": n, "author": "a", "staff": False, "created_at": "2023-01-01", "likes": 0,
         "text": "x" * 20_000} for n in (1, 2)]}
    chunks = extract.render(document)
    assert len(chunks) == 2
    assert all(chunk.startswith("Title: T") for chunk in chunks)
