#!/usr/bin/env python3
"""Disabled legacy crawler entry point.

Use the Streamlit app's **Web import** page, which enforces per-host
authorization, robots.txt rules, request budgets, and network address checks.
"""
from __future__ import annotations

import asyncio


async def main() -> None:
    print(
        "The legacy image crawler is disabled because it does not enforce the "
        "current authorization and network-safety rules. Use the app's Web import page."
    )


if __name__ == "__main__":
    asyncio.run(main())
