# Feature status

This matrix describes behavior present in the current `main` branch. A feature
marked experimental is available, but should not be treated as reliable for
high-stakes decisions or unrestricted use.

| Area | Status | Current behavior and limits |
| --- | --- | --- |
| Local face search | Implemented | Searches indexed local images using the configured embedding model and cosine ranking. Index compatibility records the selected model, DeepFace package version, detector settings, preprocessing ID, and vector dimension/normalization contract. Scores are not identity probabilities. |
| Image processing | Implemented | Imports user-selected local image folders, detects faces, and stores embeddings and source paths in the local ChromaDB index. |
| Face library | Implemented | Searches by filename, face ID, or person name; filters by face size; sorts, paginates, and supports manual deletion. CSV and versioned ZIP exports are available. ZIP restore requires included embeddings and a matching embedding pipeline. |
| Person labels | Implemented | A user can assign a label to a selected face. Unlabeled identity groups are suggestions and require review before assignment. |
| Duplicate management | Implemented | Detects duplicate image files, previews candidate copies, and updates indexed face records when selected copies are removed. |
| Attribute analysis | Experimental | Optional model estimates for age, gender, emotion, and ethnicity can be wrong and are not verified personal attributes. |
| Web image import | Experimental | An optional crawler is available. Only collect images from sites and for purposes you are authorized to use. See the privacy and crawler restrictions in the README. |
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
