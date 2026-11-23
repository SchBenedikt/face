from access_control import (
    access_decision,
    allowed_email_addresses,
    is_authorized_email,
    shared_auth_required,
)


def test_shared_auth_is_opt_in_and_parses_explicit_true_values():
    assert not shared_auth_required({})
    assert shared_auth_required({"FACE_AUTH_REQUIRED": "1"})
    assert shared_auth_required({"FACE_AUTH_REQUIRED": " TRUE "})
    assert not shared_auth_required({"FACE_AUTH_REQUIRED": "false"})


def test_allowlist_requires_exact_email_and_rejects_wildcards():
    allowlist = allowed_email_addresses({
        "FACE_ALLOWED_EMAILS": " Alice@example.com,*,bob@example.com,invalid "
    })
    assert allowlist == {"alice@example.com", "bob@example.com"}
    assert is_authorized_email(" ALICE@example.com ", allowlist)
    assert not is_authorized_email("mallory@example.com", allowlist)
    assert not is_authorized_email("", allowlist)
    assert not is_authorized_email("alice@example.com", set())


def test_allowlist_empty_when_missing():
    assert allowed_email_addresses({}) == set()


def test_access_decision_fails_closed_for_missing_login_and_unlisted_users():
    approved = {"alice@example.com"}
    assert access_decision(
        required=False, logged_in=False, email="", email_verified=False, allowlist=set()
    ) == "disabled"
    assert access_decision(
        required=True, logged_in=True, email="alice@example.com",
        email_verified=True, allowlist=set()
    ) == "setup_required"
    assert access_decision(
        required=True, logged_in=False, email="", email_verified=False,
        allowlist=approved
    ) == "login_required"
    assert access_decision(
        required=True, logged_in=True, email="mallory@example.com",
        email_verified=True, allowlist=approved
    ) == "not_authorized"
    assert access_decision(
        required=True, logged_in=True, email="ALICE@example.com",
        email_verified=True, allowlist=approved
    ) == "allowed"
    assert access_decision(
        required=True, logged_in=True, email="alice@example.com",
        email_verified=False, allowlist=approved
    ) == "not_authorized"
