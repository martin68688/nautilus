"""Canonical Draft-role helpers.

Draft slots are unique scheduling identities. Multiple slots may share the
same behavioral role, so behavioral checks must not rely on one exact string.
"""

from __future__ import annotations


NOVEL_DRAFT_ROLES = frozenset(
    {
        "novel_exploration",
        "novel_exploration_a",
        "novel_exploration_b",
        "replacement_draft",
    }
)


def is_novel_draft_role(role: object) -> bool:
    return str(role or "") in NOVEL_DRAFT_ROLES


def canonical_draft_role(role: object) -> str:
    value = str(role or "")
    return "novel_exploration" if value in NOVEL_DRAFT_ROLES else value
