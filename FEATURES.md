# Feature status

This matrix describes behavior present in the current `main` branch. A feature
marked experimental is available, but should not be treated as reliable for
high-stakes decisions or unrestricted use.

| Area | Status | Current behavior and limits |
| --- | --- | --- |
| Local face search | Implemented | Searches indexed local images using the configured embedding model and cosine ranking. Collection and face metadata record model family/name/package version, detector settings, preprocessing version, embedding dimension, L2 normalization, and index schema version. Scores are not identity probabilities. |
| Image processing | Implemented | Imports user-selected local image folders, detects faces, and stores embeddings and source paths in the local ChromaDB index. |
| First-run setup | Implemented with limits | The empty state links to local import; the app reports the completed image/face counts and links to search. Settings checks dependencies, writable storage, ChromaDB and pipeline readiness; model weights load only after an explicit check. There is no bundled consented sample wizard. |
| Face library | Implemented | Searches by filename, face ID, or person name; filters by face size; sorts, paginates, and supports manual deletion. CSV and versioned ZIP exports are available. ZIP restore requires included embeddings and a matching embedding pipeline. |
| Person labels | Implemented with limits | A user can assign a label to a selected face. Unlabeled identity groups are suggestions and require review before assignment. The legacy automatic-propagation API is a safe no-op. Assignment rejects unconfirmed reassignment, inconsistent fields for one person ID, and duplicate records with the same name and birth details; people who share a name but have different details remain separate. Per-face history records old/new fields, actor, source, suggested score and timestamp. The current session can undo its latest reviewed group assignment; changed records are skipped to protect later edits. In local mode the actor is recorded as `local-user`; optional OIDC mode records the authenticated email. |
| Duplicate management | Implemented with recovery | Detects byte-identical image files, previews and rechecks the selected set, writes a local portable recovery archive for indexed faces and originals before cleanup, removes associated vector rows, and offers in-app restore. Recovery archives are biometric data and remain until restored or removed by the user. |
| Attribute analysis | Experimental | Optional model estimates for age, gender, emotion, and ethnicity can be wrong and are not verified personal attributes. |
| Web image import | Experimental | Crawler APIs refuse to start without explicit authorization confirmation. Each downloaded image gets a source URL, collection timestamp, and confirmation record; provenance is shown in the image gallery and copied to indexed face metadata. The importer accepts at most 10 entered URLs and 500 images per hostname, paces and caps outbound requests per host, checks robots.txt, disables redirects, follows linked pages and sitemaps without guessed directory scans, restricts requests to the entered exact hostname, rejects non-public IP addresses, and pins browser and HTTP clients to the validated public address. The legacy bulk crawler is disabled. Actual legal consent is not independently verified. |
| Shared or internet-facing deployment | Implemented with configuration | Optional fail-closed Streamlit OIDC login and exact email allowlist gate all app pages and data actions when `FACE_AUTH_REQUIRED=1`. Operators must configure OIDC secrets and `FACE_ALLOWED_EMAILS`; default local mode is unauthenticated and should remain bound to localhost. |
| Biometric data deletion | Implemented with limits | Settings clears the searchable face index, cached thumbnails, duplicate-file path/hash metadata, image provenance sidecars, derived embedding files, and local recovery archives. A second option removes source images referenced by the index and all images in the app's managed folders. There is no automatic expiration; files in other folders remain unless referenced by the index and the user selects source-image deletion. |
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
