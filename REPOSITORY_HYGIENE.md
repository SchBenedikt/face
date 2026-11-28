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

The current `main` tree passes that guard. The local Git object pack is
933.47 MiB. A metadata-only scan of reachable refs confirms large historical
image and generated URL-list objects remain. Their contents were not inspected,
so this does not establish consent, privacy, or license compatibility. Keep
exact affected paths private and review them before a rewrite. A history rewrite
is a separate coordinated operation; a passing tree check does not shrink
existing clones or the hosted object store.

## Start with an empty local index

The app creates its ChromaDB collection on first startup. To verify the empty
database path without touching an existing library, point `FACE_DATA_DIR` at a
new directory before launching the app:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python setup_env.py
export FACE_DATA_DIR="$(mktemp -d)"
python -c 'from config import VECTOR_DB_PATH, COLLECTION_NAME; from vector_store import FaceVectorStore; store = FaceVectorStore(); stats = store.get_collection_stats(); print("Initialized {} at {}; faces={}".format(COLLECTION_NAME, VECTOR_DB_PATH, stats["total_faces"]))'
streamlit run app.py
```

The new index starts with zero faces. Add images only after confirming you have
the right to process them. Keep the temporary data directory if you want to
retain that library; remove it only when you intend to delete that local data.

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
