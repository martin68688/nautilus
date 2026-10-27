"""Official repositories read at their last commit before the cutoff."""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime
from pathlib import PurePosixPath

from .http import CachedHttp

_TEXT_SUFFIXES = {".md", ".py", ".ipynb", ".rst", ".txt"}
_MAX_FILE_BYTES = 20_000_000  # notebooks carry outputs; the text kept is capped below
_MAX_TEXT_CHARS = 200_000


def last_commit_before(http: CachedHttp, repo: str, cutoff: datetime) -> dict:
    until = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    commits = http.get_json(f"https://api.github.com/repos/{repo}/commits?until={until}&per_page=1")
    if not commits:
        raise RuntimeError(f"{repo} has no commit before {until}")
    commit = commits[0]
    return {"sha": commit["sha"], "date": commit["commit"]["committer"]["date"]}


def _notebook_text(raw: bytes) -> str:
    notebook = json.loads(raw.decode("utf-8"))
    parts = []
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        parts.append(f"```python\n{source}\n```" if cell.get("cell_type") == "code" else source)
    return "\n\n".join(parts)


def repo_documents(http: CachedHttp, repo: str, cutoff: datetime) -> tuple[dict, list[dict]]:
    """Return (commit, documents) for the repository snapshot before the cutoff.

    The tarball is only read in memory; nothing from it is executed or written.
    """
    commit = last_commit_before(http, repo, cutoff)
    archive = http.get_bytes(f"https://codeload.github.com/{repo}/tar.gz/{commit['sha']}")
    documents = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.size > _MAX_FILE_BYTES:
                continue
            path = PurePosixPath(*PurePosixPath(member.name).parts[1:])
            if path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            raw = tar.extractfile(member).read()
            try:
                text = _notebook_text(raw) if path.suffix == ".ipynb" else raw.decode("utf-8")
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if len(text.strip()) < 200 or len(text) > _MAX_TEXT_CHARS:
                continue
            documents.append({
                "source": "repo",
                "doc_id": f"repo:{repo}:{path}",
                "url": f"https://github.com/{repo}/blob/{commit['sha']}/{path}",
                "title": f"{repo}/{path}",
                "category": repo,
                "created_at": commit["date"],
                "posts": [{
                    "post_number": 1,
                    "author": repo.split("/")[0],
                    "staff": True,
                    "created_at": commit["date"],
                    "likes": 0,
                    "text_status": "repo_snapshot",
                    "text": text,
                }],
            })
    return commit, documents
