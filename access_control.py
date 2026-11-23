"""Fail-closed helpers for optional shared-deployment access control."""
from __future__ import annotations

import os
from collections.abc import Mapping


def shared_auth_required(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the deployment operator requires authenticated app access."""
    values = os.environ if environ is None else environ
    return str(values.get("FACE_AUTH_REQUIRED", "")).strip().casefold() in {
        "1", "true", "yes", "on"
    }


def allowed_email_addresses(environ: Mapping[str, str] | None = None) -> set[str]:
    """Read the explicit comma-separated allowlist without accepting wildcards."""
    values = os.environ if environ is None else environ
    raw = str(values.get("FACE_ALLOWED_EMAILS", ""))
    return {
        value.strip().casefold()
        for value in raw.split(",")
        if value.strip() and "@" in value and "*" not in value
    }


def is_authorized_email(email: str, allowlist: set[str]) -> bool:
    """Require an exact normalized email match; an empty list denies everyone."""
    normalized = str(email or "").strip().casefold()
    return bool(normalized and normalized in allowlist)


def access_decision(
    *,
    required: bool,
    logged_in: bool,
    email: str,
    email_verified: bool,
    allowlist: set[str],
) -> str:
    """Return the gate outcome used by the Streamlit page guard."""
    if not required:
        return "disabled"
    if not allowlist:
        return "setup_required"
    if not logged_in:
        return "login_required"
    if not email_verified or not is_authorized_email(email, allowlist):
        return "not_authorized"
    return "allowed"
