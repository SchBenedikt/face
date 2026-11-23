"""Web import rejects unapproved and non-public destinations."""
from __future__ import annotations

import socket
import asyncio
import importlib.util
from pathlib import Path
import urllib.robotparser

import pytest

from scraper_safety import HostRequestBudget, approved_public_url


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
    monkeypatch.setattr(
        "old_image_scraper.public_addresses_for_host",
        lambda host, port: ["93.184.216.34"],
    )
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


def test_http_connector_resolves_only_the_pinned_approved_address():
    from old_image_scraper import _PinnedHostResolver

    resolver = _PinnedHostResolver("example.test", "93.184.216.34")
    result = asyncio.run(resolver.resolve("example.test", 443))
    assert result[0]["host"] == "93.184.216.34"
    assert result[0]["family"] == socket.AF_INET

    try:
        asyncio.run(resolver.resolve("internal.example.test", 443))
    except OSError as exc:
        assert "only permits the approved hostname" in str(exc)
    else:
        raise AssertionError("unapproved hosts must not resolve through the pinned connector")


@pytest.mark.parametrize("script_name", ["image.py", "image-download.py"])
def test_legacy_bulk_crawler_entry_point_is_disabled(capsys, script_name):
    script = Path(__file__).resolve().parents[1] / script_name
    spec = importlib.util.spec_from_file_location("legacy_image_download", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    asyncio.run(module.main())

    output = capsys.readouterr().out
    assert "is disabled" in output
    assert "Web import" in output


def test_host_request_budget_stops_at_limit_and_tracks_hosts_separately():
    async def run():
        budget = HostRequestBudget(max_requests=2, min_interval=0)
        assert await budget.acquire("example.test")
        assert await budget.acquire("example.test")
        assert not await budget.acquire("example.test")
        assert await budget.acquire("other.test")

    asyncio.run(run())


def test_cancelled_budget_wait_does_not_consume_another_request():
    async def run():
        budget = HostRequestBudget(max_requests=2, min_interval=1)
        assert await budget.acquire("example.test")
        pending = asyncio.create_task(budget.acquire("example.test"))
        await asyncio.sleep(0)
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        budget.min_interval = 0
        assert await budget.acquire("example.test")
        assert not await budget.acquire("example.test")

    asyncio.run(run())


def test_directory_discovery_is_disabled():
    from old_image_scraper import AdvancedImageScraper

    scraper = AdvancedImageScraper()
    assert asyncio.run(scraper.find_images_from_directories("https://example.test")) == []
    assert asyncio.run(scraper._discover_common_pages("https://example.test")) == set()


def test_multiple_urls_share_one_host_image_budget(monkeypatch):
    import old_image_scraper as module

    async def no_wait(_duration):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", no_wait)
    scraper = module.AdvancedImageScraper()
    requested = []

    async def fake_scrape(url, limit, *, _reset_budget=True):
        requested.append(limit)
        return {"success": True, "url": url, "images_found": limit, "images_downloaded": limit}

    monkeypatch.setattr(scraper, "scrape_website", fake_scrape)
    results = asyncio.run(scraper.scrape_multiple_websites(
        ["https://example.test/a", "https://example.test/b", "https://example.test/c"],
        max_images_per_site=3,
    ))

    assert requested == [3]
    assert results[1]["images_found"] == 0
    assert "budget was exhausted" in results[1]["error"]
