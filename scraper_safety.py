"""Network safety checks for the optional image crawler."""
from __future__ import annotations

import ipaddress
import asyncio
import socket
import time
from urllib.parse import urlsplit


class HostRequestBudget:
    """Bound and pace outbound requests per hostname for one crawl session."""

    def __init__(self, max_requests: int = 1200, min_interval: float = 0.2):
        self.max_requests = max(1, int(max_requests))
        self.min_interval = max(0.0, float(min_interval))
        self._counts: dict[str, int] = {}
        self._last_request: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, host: str) -> bool:
        key = str(host).casefold().rstrip(".")
        async with self._lock:
            count = self._counts.get(key, 0)
            if count >= self.max_requests:
                return False
            now = time.monotonic()
            delay = self.min_interval - (now - self._last_request.get(key, now))
            if delay > 0:
                await asyncio.sleep(delay)
            self._counts[key] = count + 1
            self._last_request[key] = time.monotonic()
            return True

    def reset(self) -> None:
        self._counts.clear()
        self._last_request.clear()


def approved_public_url(
    url: str,
    approved_hosts: set[str],
    *,
    resolver=socket.getaddrinfo,
) -> tuple[bool, str]:
    """Allow only explicit HTTP(S) hosts that resolve exclusively to public IPs."""
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").rstrip(".").casefold()
        if parsed.scheme.casefold() not in {"http", "https"}:
            return False, "Only HTTP and HTTPS URLs are allowed."
        if parsed.username or parsed.password:
            return False, "URLs with embedded credentials are not allowed."
        if not host or host not in {item.rstrip(".").casefold() for item in approved_hosts}:
            return False, "The URL host was not explicitly approved."
        if parsed.scheme.casefold() == "http" and parsed.port not in (None, 80):
            return False, "HTTP is allowed only on port 80."
        if parsed.scheme.casefold() == "https" and parsed.port not in (None, 443):
            return False, "HTTPS is allowed only on port 443."

        try:
            addresses = {ipaddress.ip_address(host)}
        except ValueError:
            addresses = {
                ipaddress.ip_address(result[4][0].split("%", 1)[0])
                for result in resolver(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
            }
        if not addresses or any(not address.is_global for address in addresses):
            return False, "The URL resolves to a private or special-use network address."
        return True, ""
    except (OSError, ValueError) as exc:
        return False, f"The URL host could not be safely resolved: {exc}"
