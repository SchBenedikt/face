"""Web import rejects unapproved and non-public destinations."""
from __future__ import annotations

import socket
import asyncio
import importlib.util
from pathlib import Path
import urllib.robotparser

from scraper_safety import approved_public_url


def _resolver(addresses):
    return lambda host, port, type=socket.SOCK_STREAM: [
        (socket.AF_INET, type, 6, "", (address, port)) for address in addresses
    ]


def test_approved_public_host_is_allowed():
    allowed, reason = approved_public_url(
        "https://example.test/gallery",
        {"example.test"},
        resolver=_resolver(["93.184.216.34"]),
    )

    assert allowed, reason


def test_unapproved_host_and_private_ip_are_rejected():
    unapproved, _ = approved_public_url(
        "https://cdn.example.test/photo.jpg",
        {"example.test"},
        resolver=_resolver(["93.184.216.34"]),
    )
    private, reason = approved_public_url(
        "http://example.test/photo.jpg",
        {"example.test"},
        resolver=_resolver(["127.0.0.1"]),
    )

    assert not unapproved
    assert not private
    assert "private" in reason


def test_credentials_nonstandard_ports_and_mixed_dns_answers_are_rejected():
    credential_url, _ = approved_public_url(
        "https://user:pass@example.test/",
        {"example.test"},
        resolver=_resolver(["93.184.216.34"]),
    )
    port_url, _ = approved_public_url(
        "https://example.test:8443/",
        {"example.test"},
        resolver=_resolver(["93.184.216.34"]),
    )
    mixed_answers, _ = approved_public_url(
        "https://example.test/",
        {"example.test"},
        resolver=_resolver(["93.184.216.34", "10.0.0.1"]),
    )

    assert not credential_url
    assert not port_url
    assert not mixed_answers


def test_crawler_applies_robots_disallow_rules(monkeypatch):
    from old_image_scraper import AdvancedImageScraper

    def allow_public(url, hosts):
        assert hosts == {"example.test"}
        return True, ""

    monkeypatch.setattr("old_image_scraper.approved_public_url", allow_public)
    scraper = AdvancedImageScraper()
    scraper.approved_hosts = {"example.test"}
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(["User-agent: *", "Disallow: /private"])
    scraper.robots_parser = parser

    public_result = asyncio.run(scraper._url_is_allowed("https://example.test/gallery"))
    blocked_result = asyncio.run(scraper._url_is_allowed("https://example.test/private/photo.jpg"))

    assert public_result == (True, "")
    assert not blocked_result[0]
    assert "robots.txt" in blocked_result[1]


def test_legacy_bulk_crawler_entry_point_is_disabled(capsys):
    script = Path(__file__).resolve().parents[1] / "image-download.py"
    spec = importlib.util.spec_from_file_location("legacy_image_download", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    asyncio.run(module.main())

    output = capsys.readouterr().out
    assert "is disabled" in output
    assert "network-safety rules" in output
