# Testing guide

The regression suite uses synthetic images and temporary ChromaDB collections.
It does not include personal photographs or evaluate identity accuracy on a
representative population.

## Requirements

- Python 3.11 or 3.12
- Dependencies from `requirements.txt`
- `pytest`

## Run checks

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests
python .github/scripts/check_repository_artifacts.py
```

GitHub Actions runs these checks on pushes to `main` and pull requests targeting
`main`, on Ubuntu with Python 3.11 and 3.12.

See [Repository hygiene](REPOSITORY_HYGIENE.md) for the artifact guard's scope
and a coordinated history-cleanup procedure.

## Coverage

The suite covers local upload and search flows, embedding pipeline compatibility,
search ranking and thresholds, stable navigation, gallery filtering and
pagination, person-group suggestions, duplicate deletion and reconciliation,
portable archive integrity/restore, and batch updates.

The suite validates crawler authorization, URL/robots policy, DNS pinning,
redirect refusal, per-host budgets, retry/backoff behavior, cancellation cleanup,
and saved source provenance, plus the OIDC/email-allowlist decision logic. It
does not connect to a live identity provider or crawl external websites. The
privacy deletion tests exercise index, source-image, derived-data, and archive removal against
temporary local stores. Large-scale face accuracy is not currently validated.
Similarity scores are ranking signals, not identity probabilities.
