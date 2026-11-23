"""Network safety checks for the optional image crawler."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit


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
