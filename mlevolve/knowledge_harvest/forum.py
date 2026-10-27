"""Discourse forum reader that returns each topic as it stood before a cutoff.

Only posts created before the cutoff are kept. A post edited after the cutoff
is rolled back to its last version before the cutoff through the public
revision history; if that history cannot be read the post is dropped.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Iterator

from .http import CachedHttp

_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
               "pre", "blockquote", "tr", "table", "hr", "aside", "details", "summary"}
_SKIP_TAGS = {"script", "style", "svg"}
_POST_CHUNK = 20


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def cutoff_datetime(cutoff: str) -> datetime:
    """A cutoff date means 00:00 UTC of that day; earlier posts are kept."""
    return datetime.fromisoformat(cutoff).replace(tzinfo=timezone.utc)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._pre = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "pre":
            self._pre += 1
            self.parts.append("\n```\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "img":
            alt = dict(attrs).get("alt")
            if alt:
                self.parts.append(f"[image: {alt}]")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag == "pre":
            self._pre = max(0, self._pre - 1)
            self.parts.append("\n```\n")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data if self._pre else re.sub(r"\s+", " ", data))


def html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    parser.feed(markup or "")
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def previous_markdown(revision: dict[str, Any]) -> str | None:
    """Return the post body before ``revision`` from Discourse's side-by-side diff."""
    table = (revision.get("body_changes") or {}).get("side_by_side_markdown") or ""
    match = re.search(r'<td class="--previous">(.*?)</td>', table, flags=re.S)
    if match is None:
        return None
    body = re.sub(r"<ins>.*?</ins>", "", match.group(1), flags=re.S)
    body = re.sub(r"</?(del|span)[^>]*>", "", body)
    return html.unescape(body).strip()


class DiscourseForum:
    def __init__(self, base_url: str, http: CachedHttp):
        self.base_url = base_url.rstrip("/")
        self.http = http

    def topics(self, category_slug: str, category_id: int, cutoff: datetime) -> Iterator[dict]:
        """Yield every topic of a category created before the cutoff."""
        seen: set[int] = set()
        page = 0
        while True:
            data = self.http.get_json(
                f"{self.base_url}/c/{category_slug}/{category_id}/l/latest.json?page={page}"
            )
            listed = (data.get("topic_list") or {}).get("topics") or []
            if not listed:
                return
            for topic in listed:
                if topic["id"] in seen:
                    continue
                seen.add(topic["id"])
                if parse_time(topic["created_at"]) < cutoff:
                    yield topic
            page += 1

    def _posts(self, topic_id: int) -> tuple[dict, list[dict]]:
        topic = self.http.get_json(f"{self.base_url}/t/{topic_id}.json")
        stream = topic["post_stream"]["stream"]
        posts = {post["id"]: post for post in topic["post_stream"]["posts"]}
        missing = [post_id for post_id in stream if post_id not in posts]
        for start in range(0, len(missing), _POST_CHUNK):
            query = "&".join(f"post_ids[]={post_id}" for post_id in missing[start:start + _POST_CHUNK])
            extra = self.http.get_json(f"{self.base_url}/t/{topic_id}/posts.json?{query}")
            for post in extra["post_stream"]["posts"]:
                posts[post["id"]] = post
        return topic, [posts[post_id] for post_id in stream if post_id in posts]

    def _text_at_cutoff(self, post: dict, cutoff: datetime) -> tuple[str | None, str]:
        version = int(post.get("version") or 1)
        if version <= 1:
            return html_to_text(post.get("cooked", "")), "original"
        for number in range(2, version + 1):
            try:
                revision = self.http.get_json(
                    f"{self.base_url}/posts/{post['id']}/revisions/{number}.json"
                )
            except Exception:
                return None, "revision_unreadable"
            if parse_time(revision["created_at"]) >= cutoff:
                body = previous_markdown(revision)
                return (body, "rolled_back") if body else (None, "revision_unreadable")
        return html_to_text(post.get("cooked", "")), "edited_before_cutoff"

    def topic_document(self, topic_id: int, cutoff: datetime, category: str) -> dict | None:
        topic, posts = self._posts(topic_id)
        kept: list[dict] = []
        dropped = 0
        for post in posts:
            if post.get("post_type") != 1 or post.get("deleted_at") or post.get("hidden"):
                continue
            if parse_time(post["created_at"]) >= cutoff:
                continue
            text, status = self._text_at_cutoff(post, cutoff)
            if text is None:
                dropped += 1
                continue
            kept.append({
                "post_number": post["post_number"],
                "author": post.get("username"),
                "staff": bool(post.get("staff") or post.get("admin") or post.get("moderator")),
                "created_at": post["created_at"],
                "likes": sum(a.get("count", 0) for a in post.get("actions_summary") or [] if a.get("id") == 2),
                "text_status": status,
                "text": text,
            })
        if not kept:
            return None
        return {
            "source": "forum",
            "doc_id": f"forum:{topic_id}",
            "url": f"{self.base_url}/t/{topic.get('slug', topic_id)}/{topic_id}",
            "title": topic.get("title", ""),
            "category": category,
            "created_at": topic["created_at"],
            "posts": kept,
            "posts_dropped_unreadable_revision": dropped,
        }
