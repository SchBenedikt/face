# Repository artifact and history cleanup

## Current protection

The CI artifact guard checks tracked files in the current tree. It rejects
known generated database/OS artifacts, database and backup suffixes, private
environment/key files, and files larger than 10 MiB. It does not inspect older
Git objects. Run it before pushing:

```bash
python .github/scripts/check_repository_artifacts.py
git ls-files
```

The current `main` tree passes that guard. The local Git object pack is still
about 933 MiB, so files removed in earlier commits may remain in history. A
history rewrite is a separate coordinated operation; a passing tree check does
not shrink existing clones or the hosted object store.

## Coordinated history rewrite

Before rewriting history:

1. Announce a write freeze and coordinate with every contributor and bot that
   can push branches or tags. Record the exact artifact paths after a private
   review; do not publish file contents or sensitive provenance.
2. Create and verify a mirror backup outside the working checkout:

   ```bash
   git clone --mirror https://github.com/SchBenedikt/face.git face-history-backup.git
   git -C face-history-backup.git fsck --full
   git -C face-history-backup.git count-objects -vH
   ```

3. Make a second disposable mirror, then use `git filter-repo` with the reviewed
   exact paths. Include every branch and tag that will be force-pushed. Keep the
   original mirror untouched until the hosted repository and a fresh clone
   have been verified.
4. In the filtered mirror, inspect all refs and reachable object sizes, run the
   artifact guard from a fresh checkout, and confirm the removed paths are no
   longer present in reachable history. Record the before/after pack sizes.
5. Force-push the rewritten refs only after repository-owner approval and the
   write freeze. Tell contributors to discard old clones and re-clone; merging
   old branches can reintroduce the removed history.
6. After the push, verify `main`, the intended branch/tag set, a fresh clone,
   the artifact guard, and GitHub Actions. Retain the verified mirror backup
   until the team confirms the rewrite is stable.

Do not delete the backup as part of the rewrite. Coordinate any eventual backup
retention or removal separately.
