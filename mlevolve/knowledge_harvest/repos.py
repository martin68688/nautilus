"""Official repositories read at their last commit before the cutoff."""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path, PurePosixPath

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


def first_added_dates(repo: str, sha: str, git_dir: Path) -> dict[str, str]:
    """Date each path was first added in the history up to ``sha``.

    A snapshot's commit date says nothing about when a page was written; this
    keeps 2020 advice from looking like January 2024 advice. Only commit
    metadata and trees are fetched (no file contents).
    """
    clone = git_dir / repo.replace("/", "__")
    if not clone.exists():
        git_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout",
                        f"https://github.com/{repo}.git", str(clone)], check=True)
    log = subprocess.run(["git", "-C", str(clone), "log", "--diff-filter=A", "--name-only",
                          "--format=@%cI", sha], check=True, capture_output=True, text=True).stdout
    dates: dict[str, str] = {}
    current = ""
    for line in log.splitlines():
        if line.startswith("@"):
            current = line[1:]
        elif line.strip():
            dates[line.strip()] = current  # log runs newest first: the last write is the oldest add
    return dates


def _norm_text(text: str) -> str:
    """Key for duplicate pages: copies often differ only after a long shared opening."""
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower()[:4000].encode("utf-8")).hexdigest()


def repo_documents(http: CachedHttp, repo: str, cutoff: datetime,
                   git_dir: Path | None = None) -> tuple[dict, list[dict]]:
    """Return (commit, documents) for the repository snapshot before the cutoff.

    The tarball is only read in memory; nothing from it is executed or written.
    Pages with identical text under several paths are kept once (oldest date).
    """
    commit = last_commit_before(http, repo, cutoff)
    added = first_added_dates(repo, commit["sha"], git_dir) if git_dir is not None else {}
    archive = http.get_bytes(f"https://codeload.github.com/{repo}/tar.gz/{commit['sha']}")
    documents = []
    seen: dict[str, dict] = {}
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
            date = added.get(str(path), commit["date"])
            digest = _norm_text(text)
            if digest in seen:
                if date < seen[digest]["created_at"]:
                    seen[digest]["created_at"] = seen[digest]["posts"][0]["created_at"] = date
                continue
            document = {
                "source": "repo",
                "doc_id": f"repo:{repo}:{path}",
                "url": f"https://github.com/{repo}/blob/{commit['sha']}/{path}",
                "title": f"{repo}/{path}",
                "category": repo,
                "created_at": date,
                "posts": [{
                    "post_number": 1,
                    "author": repo.split("/")[0],
                    "staff": True,
                    "created_at": date,
                    "likes": 0,
                    "text_status": "repo_snapshot",
                    "text": text,
                }],
            }
            seen[digest] = document
            documents.append(document)
    return commit, documents
