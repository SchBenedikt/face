# Feature status

This matrix describes behavior present in the current `main` branch. A feature
marked experimental is available, but should not be treated as reliable for
high-stakes decisions or unrestricted use.

| Area | Status | Current behavior and limits |
| --- | --- | --- |
| Local face search | Implemented | Searches indexed local images using the configured embedding model and cosine ranking. Index compatibility records the selected model, DeepFace package version, detector settings, preprocessing ID, and vector dimension/normalization contract. Scores are not identity probabilities. |
| Image processing | Implemented | Imports user-selected local image folders, detects faces, and stores embeddings and source paths in the local ChromaDB index. |
| First-run diagnostics | Implemented with limits | Settings checks required dependencies, writable local storage, ChromaDB availability, and index compatibility. The embedding model is loaded only when the user starts the explicit model check; there is no guided consented sample wizard. |
| Face library | Implemented | Searches by filename, face ID, or person name; filters by face size; sorts, paginates, and supports manual deletion. CSV and versioned ZIP exports are available. ZIP restore requires included embeddings and a matching embedding pipeline. |
| Person labels | Implemented with limits | A user can assign a label to a selected face. Unlabeled identity groups are suggestions and require review before assignment. Per-face assignment and removal history records old and new person fields and timestamps. The current session can undo its latest reviewed group assignment; changed records are skipped to protect later edits. Actor identity is local and unauthenticated. |
| Duplicate management | Implemented with recovery | Detects byte-identical image files, previews and rechecks the selected set, writes a local portable recovery archive for indexed faces and originals before cleanup, removes associated vector rows, and offers in-app restore. Recovery archives are biometric data and remain until restored or removed by the user. |
| Attribute analysis | Experimental | Optional model estimates for age, gender, emotion, and ethnicity can be wrong and are not verified personal attributes. |
| Web image import | Experimental | Requires explicit authorization confirmation, accepts at most 10 entered URLs and 500 images per hostname, paces and caps outbound requests per host, checks robots.txt, disables redirects, follows linked pages and sitemaps without guessed directory scans, restricts requests to the entered exact hostname, and rejects non-public IP addresses. The legacy bulk crawler is disabled. DNS rebinding and actual legal consent are not independently verified. |
| Shared or internet-facing deployment | Unsupported | The app has no built-in user authentication or per-user authorization. Run it locally; a shared deployment needs an authenticated access layer in front of it. |
| Portable archive export and restore | Implemented with limits | ZIP archives can include selected face records, provenance metadata, optional original images, and optional embeddings. A SHA-256 manifest protects archive entries. Restore validates the pipeline and refuses duplicate face IDs. Downloads are capped at 128 MiB; demographic model outputs and thumbnails are not exported. |
| Live camera and mobile clients | Planned | No supported live-camera workflow or mobile application is included. |

## Score semantics

Search scores are clipped cosine-similarity ranking values in the range 0–1.
They are not calibrated probabilities and do not establish identity. The
legacy `basic`, `enhanced`, and `premium` names in `FaceSimilarityCalculator`
are compatibility arguments; all use the same cosine score. The active app
search uses `FaceVectorStore` and does not blend unvalidated distance metrics.

## Evaluation

The README records reproducible diagnostic samples for LFW and CFP. Those
measurements are limited to their datasets and sampling procedures; they do not
establish performance on a user's collection or justify high-stakes use.
