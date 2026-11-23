# Project structure and runtime behavior

This file describes the current application source. It is not a security or compliance certification.

## Face search flow

1. `app.py` accepts an uploaded query image and displays detected faces.
2. `utils.py` loads images and applies EXIF orientation handling.
3. `face_recognition_engine.py` detects faces, reuses aligned detector crops, and extracts one Buffalo_L embedding per face.
4. `vector_store.py` stores normalized embeddings and metadata in ChromaDB, then ranks indexed faces by cosine similarity.
5. Search covers the configured local collection. It does not search public websites.

The detector tries the optional `face_recognition` package first when installed. DeepFace backends are then tried in configured order, followed by the OpenCV Haar fallback. The active search score is cosine similarity; it is not a probability of identity.

## Main modules

- `app.py` — Streamlit pages for face search, galleries, names, scraping, settings, and batch indexing.
- `face_recognition_engine.py` — Face detection, crop handling, Buffalo_L embedding extraction, and image batch processing.
- `vector_store.py` — ChromaDB collections, pipeline-version checks, similarity search, grouping, and person metadata.
- `similarity.py` — Normalization and distance conversion helpers.
- `face_grouping.py` — Clustering helpers for reviewed identity groups.
- `evaluate_face_search.py` — Retrieval and threshold metrics for labeled, identity-folder datasets.
- `settings_store.py` — Validated JSON settings stored under the configured data directory.
- `fast_process.py` — CLI batch indexing for `IMAGES_DIR`; `--update` refreshes that directory.
- `fast_config.py` — Shared data paths, supported image formats, and batch size. It does not select a different model.
- `image-download.py` — Interactive image crawler. Running it contacts external websites.
- `clean_duplicates.py` — Reports or removes image-file variants; it does not validate face identities.
- `image_quality_manager.py` and `duplicate_detector.py` — Image quality and duplicate utilities.

## Configuration and data

- `config.py` defines `FACE_DATA_DIR`, image directories, ChromaDB location, detector order, Buffalo_L model, and the default cosine threshold.
- Without `FACE_DATA_DIR`, runtime files live under `data/`.
- Face embeddings are 512-dimensional for the active Buffalo_L pipeline.
- A detector, crop, or embedding-pipeline change increments `EMBEDDING_PIPELINE_VERSION`. Reprocess each image directory before searching its old vectors again.
- The app has no built-in consent manager, encryption layer, authentication system, anonymization workflow, or legal-compliance checker. Protect the data directory with operating-system access controls and process only images you are authorized to use.

## Dependencies and checks

- `requirements.txt` contains the Streamlit app runtime dependencies.
- `requirements-optional.txt` contains optional dlib and MediaPipe packages.
- `tests/` contains the current pytest suite and local stubs.
- `README.md`, `FEATURES.md`, and `TESTING_GUIDE.md` describe setup, features, and validation guidance.
