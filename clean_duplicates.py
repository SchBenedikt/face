#!/usr/bin/env python3
"""Report or safely remove byte-identical image files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import IMAGES_DIR
from image_quality_manager import clean_image_duplicates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find byte-identical image files. Cleanup creates a local recovery "
            "archive for indexed faces and reconciles the vector database."
        ),
        epilog=(
            "Examples:\n"
            "  python clean_duplicates.py\n"
            "  python clean_duplicates.py --dir data/images --report\n"
            "  python clean_duplicates.py --dir data/images --clean"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        default=str(IMAGES_DIR),
        help=f"Image directory to scan (default: {IMAGES_DIR})",
    )
    parser.add_argument("--clean", action="store_true", help="Remove reviewed exact duplicates")
    parser.add_argument("--report", action="store_true", help="Print the detailed duplicate report")
    parser.add_argument("--yes", action="store_true", help="Confirm cleanup without an interactive prompt")
    parser.add_argument("--quiet", action="store_true", help="Reduce informational output")
    args = parser.parse_args(argv)

    image_directory = Path(args.dir).expanduser()
    if not image_directory.is_dir():
        print(f"Error: image directory does not exist: {image_directory}", file=sys.stderr)
        return 1

    if args.clean and not args.yes:
        try:
            response = input(
                f"Remove only byte-identical duplicates from {image_directory}? Type DELETE to continue: "
            ).strip()
        except EOFError:
            response = ""
        if response != "DELETE":
            print("Cleanup cancelled.")
            return 0

    try:
        result = clean_image_duplicates(str(image_directory), dry_run=not args.clean)
    except Exception as exc:  # noqa: BLE001 - return a readable CLI error
        print(f"Cleanup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    stats = result["removal_stats"]
    if args.report and not args.quiet:
        print(result["report"])
    if not args.quiet:
        print(f"Duplicate groups: {stats['total_groups']}")
        print(f"Files to remove: {stats['files_to_remove']}")
        print(f"Estimated space: {stats['bytes_to_free'] / (1024 * 1024):.2f} MiB")
        if args.clean:
            print(f"Files removed: {stats.get('files_removed', 0)}")
            print(f"Errors: {len(stats['errors'])}")
            if stats.get("recovery_archives"):
                print("Recovery archives:")
                for path in stats["recovery_archives"]:
                    if path:
                        print(f"  {path}")
        elif stats["files_to_remove"]:
            print("Run with --clean to remove these exact duplicates after confirmation.")
        else:
            print("No byte-identical duplicates found.")

    if stats["errors"]:
        for error in stats["errors"]:
            print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
