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

## Coverage

The suite covers local upload and search flows, embedding pipeline compatibility,
search ranking and thresholds, stable navigation, gallery filtering and
pagination, person-group suggestions, duplicate deletion and reconciliation,
portable archive integrity/restore, and batch updates.

Web collection, external deployment authentication, and large-scale face
accuracy are not currently validated by the suite. Similarity scores are
ranking signals, not identity probabilities.
