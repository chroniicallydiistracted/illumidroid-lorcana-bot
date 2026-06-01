"""Fail-closed guard for Tier-A belief-search remediation.

Belief-guided clean-label training is intentionally unavailable until the full
hidden-zone, shared-tree path passes the Tier-A release gate. Diagnostic search
and non-belief training remain available while the remediation lands.
"""

from __future__ import annotations


TIER_A_CLEAN_LABEL_BELIEF_TRAINING_ENABLED = False


class TierARemediationIncompleteError(RuntimeError):
    """Raised when an unsafe belief-guided sample-writing path is requested."""


def require_tier_a_clean_label_belief_training_ready(*, use_belief: bool,
                                                     context: str) -> None:
    """Reject belief-guided sample writing until the Tier-A release gate passes."""
    if use_belief and not TIER_A_CLEAN_LABEL_BELIEF_TRAINING_ENABLED:
        raise TierARemediationIncompleteError(
            f"{context}: belief-guided clean-label training is blocked while Tier-A "
            "belief-search remediation is incomplete. Use a non-belief diagnostic "
            "run only; do not write belief-search training labels until the Tier-A "
            "release gate passes."
        )
