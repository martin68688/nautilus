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


TRANSFER_PORTFOLIO_DRAFT_ROLE_TO_SEAT = {
    "memory_transfer_comprehensive_best": "comprehensive_best",
    "memory_transfer_backbone_diversity_1": "backbone_diversity_1",
    "memory_transfer_backbone_diversity_2": "backbone_diversity_2",
    "memory_transfer_descriptor_fusion_diversity": (
        "descriptor_fusion_diversity"
    ),
}

TRANSFER_PORTFOLIO_DRAFT_ROLES = tuple(
    TRANSFER_PORTFOLIO_DRAFT_ROLE_TO_SEAT
)


def is_novel_draft_role(role: object) -> bool:
    return str(role or "") in NOVEL_DRAFT_ROLES


def canonical_draft_role(role: object) -> str:
    value = str(role or "")
    if value in TRANSFER_PORTFOLIO_DRAFT_ROLE_TO_SEAT:
        return "memory_transfer"
    return "novel_exploration" if value in NOVEL_DRAFT_ROLES else value


def is_memory_transfer_draft_role(role: object) -> bool:
    """Whether one scheduling identity has Memory Transfer behavior."""

    return canonical_draft_role(role) == "memory_transfer"


def transfer_portfolio_seat_for_role(role: object) -> str | None:
    """Return the frozen portfolio seat bound to a tournament Draft role."""

    return TRANSFER_PORTFOLIO_DRAFT_ROLE_TO_SEAT.get(str(role or ""))


def memory_retrieval_draft_role(role: object) -> str:
    """Keep a tournament slot distinct while canonicalizing legacy roles.

    Behavioral prompt checks intentionally collapse all Transfer portfolio
    slots to ``memory_transfer``.  Memory retrieval cannot do that because the
    stage-aware layer uses the scheduling identity to bind the corresponding
    frozen portfolio seat.
    """

    value = str(role or "")
    if value in TRANSFER_PORTFOLIO_DRAFT_ROLE_TO_SEAT:
        return value
    return canonical_draft_role(value)
