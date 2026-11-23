# Repository artifact and history cleanup

The current `main` tree is protected by `.gitignore` and
`.github/scripts/check_repository_artifacts.py`. The guard rejects generated
database files, backups, environment files, and files larger than 10 MiB. It
runs in CI on pushes and pull requests. `python .github/scripts/check_repository_artifacts.py`
also verifies the checked-out tree locally.

## History audit

The audit on 2026-11-23 found no database, OS, bulk-URL-dump, or backup files
in the current `main` tree. The local clone still has about 933 MiB of packed
objects across its fetched refs. The reachable `main` history contains about
1.04 GB of unique file data, including about 985 MB in common photo formats.
The file contents were not reviewed for personal data, consent, or licensing,
so this report does not claim that historical images are safe to redistribute.

Removing files from the current tree does not remove old blobs from Git
history. A history rewrite changes commit IDs. Updating `main` alone also
does not reclaim objects that remain reachable from other branches, tags, or
open pull requests.

## Coordinated rewrite procedure

Do this only after the repository owner approves the exact set of paths and
refs to rewrite and all contributors agree to a short push freeze.

1. Inventory open pull requests, all branch and tag refs, and any automation
   that depends on their commit IDs. Decide which refs must be retained and
   which historical image/data paths are approved for removal. Never remove
   historical photos solely because they are large.
2. Make a fresh mirror clone and fetch all refs. Create a private, access-
   restricted backup outside the working repository with
   `git bundle create ../face-precleanup.bundle --all`, then record and verify
   its SHA-256 checksum. The bundle may contain sensitive images and must be
   stored and shared accordingly.
3. Rewrite only the approved paths and refs in that isolated clone with a
   reviewed history-filtering tool. Run the test suite and artifact guard on
   the rewritten default branch; verify the retained refs and repository size.
4. Before publishing, confirm the expected current remote commit has not
   changed. Update only the approved refs using an explicit
   `--force-with-lease=<ref>:<expected-old-commit>` value. Do not use an
   unconditional force push or overwrite unreviewed pull-request refs.
5. Tell contributors that old commit IDs are obsolete and that their clones
   must be recreated or carefully rebased. Remove or rewrite any remaining
   refs that still keep the old objects reachable; otherwise GitHub may keep
   the old data in repository storage.

The repository owner should retain the private backup until the rewritten
repository and all required branches have been verified. The backup should
then be deleted according to the owner's data-retention policy.
