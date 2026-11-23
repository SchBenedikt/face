# 🔍 Advanced Face Recognition Search System

<div align="center">

![Python](https://img.shields.io/badge/python-v3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.49+-red.svg)
![Issues](https://img.shields.io/github/issues/SchBenedikt/face.svg)

**Local face search with Buffalo_L embeddings, aligned face crops, and a ChromaDB index**

[📖 Documentation](#-documentation) • [⚡ Quick Start](#-quick-start) • [🎯 Features](#-key-features) • [🧠 Models and Search](#-models-and-search) • [🛠️ Installation](#-installation) • [🤝 Contributing](#-contributing)

[Feature status and limitations](FEATURES.md) • [Testing and CI](TESTING_GUIDE.md)

</div>

## 📚 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🌟 What Makes This Special?](#-what-makes-this-special)
- [🎯 Key Features](#-key-features)
- [🧠 Models and Search](#-models-and-search)
- [🏗️ Architecture](#-architecture)
- [🛠️ Installation](#-installation)
- [📥 Data Processing Workflow](#-data-processing-workflow)
- [🎯 Use Cases](#-use-cases)
- [⚠️ Privacy and limitations](#-privacy-and-limitations)
- [🤝 Contributing](#-contributing)

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/SchBenedikt/face.git
cd face

# Install dependencies
pip install -r requirements.txt

# Set up the environment
python setup_env.py

# Launch the web interface
streamlit run app.py
```

**🎉 That's it!** Open your browser to `http://localhost:8501` and start exploring.

By default, Streamlit binds to `127.0.0.1`, so the interface is available only
from the machine running it. Do not change `server.address` to a network-facing
address or expose the port through a proxy unless that deployment adds
authentication and authorization for viewing, exporting, editing, and deleting
biometric data. Streamlit itself does not provide application-level access
control here.

### Performance and hardware

The app preserves the machine's CUDA visibility so a compatible GPU-enabled
DeepFace runtime can use the GPU. The default lightweight DeepFace install uses
ONNX Runtime; TensorFlow or PyTorch backends require their corresponding
DeepFace extras and compatible GPU libraries. Set `FACE_FORCE_CPU=1` before
launching the app to force CPU mode. CPU thread limits default to `1`, and can
be overridden with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, or `OPENBLAS_NUM_THREADS`.

When the face detector or embedding preprocessing changes, the app marks old
vectors as incompatible and excludes them from similarity results. In **Batch
Face Processing**, run **Update All Images** for **Images and subfolders** and
for **Scraped Images** to rebuild the complete searchable index. During that
update, only already reprocessed images are searched. Index identity includes
the selected model, DeepFace package version, detector settings, preprocessing
version, and the 512-dimensional L2-normalized vector contract. Export a
portable ZIP with embeddings before upgrading the recognition runtime or
rebuilding an index if you need a recoverable snapshot.

## 🌟 What Makes This Special?

The application provides local face search over images you choose to index:

- **🧠 Face Search**: Buffalo_L embeddings from DeepFace-detected faces
- **⚡ Batch Processing**: Index new images or refresh a selected image directory
- **🎯 Search Filtering**: Adjustable cosine similarity threshold and result filters
- **🔍 Vector Search**: Similarity search across the locally indexed image collection
- **📊 Face Analysis**: Optional age, gender, emotion, and ethnicity estimates
- **🌐 Image Crawler**: Optional interactive image collection from websites
- **👥 Person Management**: Assign names to selected faces and review identity-group suggestions
- **📦 Portable backups**: Export the current gallery page or whole index as a versioned ZIP. Original images and biometric embeddings are separate opt-in choices. Embeddings are required to restore a searchable index; restore accepts only a matching pipeline and refuses existing face IDs. ZIP downloads are capped at 128 MiB.

## 🎯 Key Features

### 🔍 **Advanced Search Engine**
- **Similarity Search**: Find faces similar to uploaded images
- **Result Filters**: Filter displayed results by cosine similarity, face size, or assigned name
- **Search Presets**: Balanced, High Precision, and High Recall adjust the cosine threshold and result count
- **Real-time Filtering**: Dynamic result filtering by similarity, size, and assigned name

### 🧠 **AI-Powered Analysis**
- **Facial Recognition**: Buffalo_L embeddings with RetinaFace-first detection and aligned face crops
- **Attribute Detection**: Age, gender, emotion, ethnicity analysis
- **Search Scores**: Cosine similarity is shown as a ranking score, not a match probability
- **Search Threshold**: Defaults to 0.40 cosine similarity; High Precision adds 0.15 and High Recall subtracts 0.10

### ⚡ **High-Performance Processing**
- **Batch Processing**: Process image directories in configurable batches
- **Incremental Updates**: Skip already indexed images or refresh a directory
- **Smart Caching**: Cache face thumbnails and selected image processing results
- **Local Index**: Keep embeddings and image paths in the configured local data directory

### 🧩 **People and grouping**
- **Identity groups**: Unlabeled faces are grouped by cosine similarity (Buffalo_L starting threshold: 0.35) and assigned only after review
- **Name management**: Assigning a name labels only the selected face; similar faces can be reviewed and assigned together from the groups page
- **Calibrated similarity**: ChromaDB's squared L2 distance is converted to cosine similarity correctly

### 🌐 **Data Collection & Management**
- **Web Scraping**: Optional interactive image crawler
- **Duplicate Detection**: Find image-file variants before indexing
- **Metadata Management**: Store image paths, face boxes, and assigned person names
- **Image Cleanup**: Find and remove duplicate image-file variants

## 🧠 Models and Search

### Face detection and embeddings

- DeepFace detection tries RetinaFace, MTCNN, and OpenCV first, retaining aligned crops for embeddings. Optional dlib and Haar detectors are fallbacks. This keeps the primary embedding path consistent when optional packages differ across machines.
- Face crops from DeepFace detectors are aligned and reused for both embedding extraction and the result preview.
- The active recognition pipeline extracts one 512-dimensional embedding per detected face using the model selected in **Settings**. Buffalo_L is the default; ArcFace is available for pose-heavy collections. Each index version includes its model, and switching models requires **Update All Images** before searching. The app does not mix vectors from different model spaces.
- Embeddings are unit-normalized and stored in the local ChromaDB collection. Search converts ChromaDB distances to cosine similarity and ranks indexed images.
- On a balanced LFW people sample (299 embedded images, 62 identities, seed 2026), Buffalo_L reached 99.7% Top-1 and 99.7% Top-5 retrieval (one error); FaceNet512 reached 98.3% Top-1 and 99.3% Top-5 (five errors). Across all 576 same-identity and 43,975 different-identity pairs from the same images, Buffalo_L at 0.40 had 87.2% same-person recall and 0/43,975 different-person false accepts. At 0.30 it had 95.5% recall and 10/43,975 false accepts. FaceNet512 at 0.60 had 82.8% recall and 58/43,975 false accepts. For reviewed identity grouping with margin 0.04, threshold 0.35 produced 70 clusters, 94.8% same-identity pair recall, and 0/43,975 false pairs; threshold 0.30 raised recall to 96.5% but introduced 25 false pairs. These matched measurements support Buffalo_L, a 0.40 search threshold, and a separate 0.35 group-suggestion threshold. Pair observations share images and are correlated; they are not independent probabilities or the official cross-validated LFW protocol, and they do not establish accuracy on other collections.
- In an earlier LFW center-crop sample (299 images, seed 42), both models reached 99.3% Top-1; Buffalo_L had two errors and FaceNet512 had two errors. A separate 200-pair full-frame sample (seed 42) gave Buffalo_L 187/200 same-person accepts and 0/200 different-person accepts at 0.30, versus FaceNet512 185/200 and 0/200 at 0.50. These samples differ in crop and pair construction; use them as supporting diagnostics, not match-probability guarantees.
- To repeat the matched retrieval comparison, install optional `scikit-learn` and run `.venv/bin/python evaluate_face_search.py --lfw-people --max-images 300 --seed 2026`. Both evaluators use the active model selected in **Settings** unless `--model` explicitly overrides it. Pair evaluation: `.venv/bin/python evaluate_lfw_pairs.py --pairs-per-class 200`; specify `--thresholds` to inspect other operating points. LFW downloads are cached outside the repository.
- To measure vector-store performance separately from face detection and recognition, run `.venv/bin/python benchmark_vector_search.py`. With 512-dimensional synthetic vectors in a temporary local Chroma index, this environment measured 1,000-vector search at 8.6 ms p50 / 13.1 ms p95 and 10,000-vector search at 25.3 / 33.7 ms; their index builds took 0.43 and 5.25 seconds. A matched 10,000-vector run (same seed and 500-row batches) improved build time from 20.41 to 7.86 seconds after batching stale-row cleanup. A 100,000-vector index built in 81.8 seconds and searched at 166.4 / 189.3 ms p50 / p95 (50 results, 100 queries). These are one-run local measurements; random vectors do not establish identity accuracy and timings depend on hardware and storage.
- Search covers only images indexed in this application's database; it does not search the public web.

### Index updates

Changes to face detection, alignment, or embedding preprocessing create a new pipeline version. The app excludes older vectors until their source images are reprocessed. Use **Batch Face Processing → Update All Images** for each image directory. Back up the database before a large reindex.

Changing from FaceNet512 to Buffalo_L invalidates the old embeddings. Reprocess each image directory before searching. A saved, unchanged FaceNet512 default threshold of 0.65 migrates to Buffalo_L's 0.40 starting threshold; other saved thresholds are retained.

When reprocessing a face, existing person labels follow a one-to-one match between old and new boxes in the same image, using at least 0.35 IoU. This preserves names when detector ordering changes; unmatched or ambiguous labels are not transferred, and a reused face ID with a nonmatching box is logged as a warning.

These LFW diagnostics do not establish web-scale retrieval performance or a comparison with PimEyes. Search quality depends on the source images, face size, pose, lighting, and indexed collection.

A larger, seed-fixed CFP frontal-profile diagnostic sampled 50 same and 50 different pairs from each of the 10 official folds (1,000 requested pairs). The current detection/embedding pipeline scored 995 pairs from 1,696 of 1,699 usable images. ArcFace reached AUC 0.861 / EER 22.8%; Buffalo_L reached AUC 0.796 / EER 28.7%. At the 0.40 search threshold, ArcFace accepted 110/499 genuine pairs and 1/496 different-person pairs; Buffalo_L accepted 1/499 and 0/496. On the matched 299-image LFW sample (62 identities, seed 2026), ArcFace reached 98.0% Top-1 / 99.3% Top-5, with 83.2% same-person recall and 0.273% different-person false accepts at 0.40. Buffalo_L reached 99.7% Top-1 / Top-5, with 87.2% recall and 0/43,975 false accepts. The pose robustness and frontal precision trade off; this evidence does not support a single global model switch.

A controlled ArcFace preprocessing pilot sampled 10 same and 10 different pairs from each CFP fold (seed 20261123). On the same detections and 199 valid pairs, DeepFace's ArcFace-specific pixel normalization scored AUC 0.867 / EER 21.6%, while the current `base` normalization scored AUC 0.869 / EER 21.1%. At cosine 0.40, they accepted 21/99 and 26/99 genuine pairs, respectively, with 0/100 different pairs accepted by either. This nested pilot does not support changing ArcFace preprocessing; crop and pose handling remain the stronger areas to investigate. Its small pair sample does not establish deployment-wide false-accept rates.

Grouping thresholds also depend on the model. With ArcFace, the existing 0.35 threshold merged 62 identities into 35 groups, with only 6.8% pairwise precision and 16.9% different-person false pairs. At 0.55, pairwise precision was 100% with 64.8% recall, but produced 127 groups. Keep identity groups under review and do not reuse Buffalo_L's grouping threshold for ArcFace.

To repeat or expand that comparison, extract the CFP dataset and run `.venv/bin/python evaluate_cfp_pairs.py /path/to/cfp-dataset --pairs-per-class 50 --seed 20261123 --models Buffalo_L ArcFace --thresholds 0.30,0.35,0.40,0.45,0.50 --embedding-cache /path/to/cfp-embeddings.npz --output /path/to/cfp-results.json`. This samples 50 same and 50 different pairs from each official split and reports overall and per-fold verification metrics plus threshold operating points. Use `--pairs-per-class 0` for all 7,000 official pairs. The optional cache contains derived face embeddings; keep it local and remove it when no longer needed. Without the cache, the evaluator runs the app's detector/alignment path once per image and compares embeddings from each requested model.

## 🛠️ Installation

### Prerequisites
- **Python 3.11 or 3.12**. CI covers Ubuntu x86_64 on both versions; other operating systems and architectures are not currently release-tested.
- **8 GB RAM recommended** for small collections. Larger collections and model loading need more memory.
- **CPU is supported**. A compatible GPU runtime is optional and requires separately installed model backends.

### Quick Install
```bash
# Clone and setup
git clone https://github.com/SchBenedikt/face.git
cd face

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies (numpy 2, ChromaDB 1.x, DeepFace with ONNX backend)
pip install -r requirements.txt

# Optional: browser binaries for the web scraper
python -m playwright install chromium

# Optional: dlib-based detection and the CLI/live scripts
pip install -r requirements-optional.txt

# Initialize system
python setup_env.py
```

### Development tools
```bash
# Install the runtime and test dependencies
pip install -r requirements.txt pytest

# Run the regression suite and repository artifact guard
python -m pytest -q tests
python .github/scripts/check_repository_artifacts.py
```

The same checks run on pushes and pull requests through GitHub Actions.

## 📥 Data Processing Workflow

### Add and index images

1. Place images in `data/images` (or in `images` under the configured `FACE_DATA_DIR`).
2. Run `python fast_process.py` to index new images. Use `python fast_process.py --update` to reprocess that directory after changing the recognition pipeline.
3. To process scraped images or another directory, use **Batch Face Processing** in the Streamlit app, select the directory, and choose **Process New Images Only** or **Update All Images**.
4. Start the interface with `streamlit run app.py`, open **Face Search**, and upload an image containing a face.

The legacy `image-download.py` crawler is disabled because it does not enforce
the current network-safety rules. Use **Web import** in the Streamlit app. It
requires an explicit authorization confirmation, applies a finite image cap,
honors the entered hostname and its `robots.txt` rules, and blocks private or
special-use network addresses. Only the exact hostname you enter is crawled;
CDNs and other subdomains require separate authorization and entry.

### Remove duplicate image files

`clean_duplicates.py` reports image-file variants by default. Review the report before deleting anything:

```bash
python clean_duplicates.py --dir data/images --report
```

To remove detected image-file duplicates after review, add `--clean`. This cleanup concerns image files; it does not verify embedding quality or identity matches.

### Settings and data location

`config.py` defines the active detector order, Buffalo_L as the default model, the default search threshold (0.40), and the default group-suggestion threshold (0.35). Settings can select ArcFace, which uses a model-specific index version and a 0.55 group-suggestion starting threshold. `FACE_DATA_DIR` relocates the images and database together. The CLI batch processor handles `IMAGES_DIR`; use the app's batch page to update additional directories.

## 🎯 Use Cases

<table>
<tr>
<td width="50%">

### 🔬 **Research & Development**
- Computer vision research and algorithm development
- Retrieval experiments on a labeled dataset with documented metrics
- Academic projects and scientific papers
- AI model performance evaluation

### 📚 **Educational**
- Teaching computer vision and deep learning concepts
- Student projects and assignments
- AI/ML course demonstrations and workshops
- Technical training and skill development

</td>
<td width="50%">

### 🏢 **Professional**
- Digital asset management and organization
- Photo journalism and media production
- Historical photo analysis and archiving
- Photo archive prototypes using images you are authorized to process

</td>
</tr>
</table>

## 🏗️ Architecture

```
Streamlit UI → image loader and face detector → aligned face crop
            → Buffalo_L embedding → ChromaDB cosine-ranked search
```

Image crawling is a separate optional workflow. Search runs against the local ChromaDB collection and indexed image paths.

### Core Components

- **`app.py`** - Main Streamlit web application with modern UI
- **`face_recognition_engine.py`** - Face detection, aligned crops, and Buffalo_L embeddings
- **`vector_store.py`** - ChromaDB database management and similarity search
- **`image_quality_manager.py`** - Image quality checks used by cleanup workflows
- **`fast_process.py`** - High-performance batch processing engine

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive tests for new features
- Update documentation for any API changes
- Ensure backward compatibility

## 📚 Documentation

- **[Feature List](FEATURES.md)** - Detailed feature documentation
- **[Testing Guide](TESTING_GUIDE.md)** - Testing procedures and benchmarks

## ⚠️ Privacy and limitations

- Images, face embeddings, names, and source metadata are stored under the configured local data directory. The default server bind is local-only. The optional image crawler makes requests to external websites.
- The app has no built-in consent tracking, legal-compliance checks, or demographic-bias evaluation. You are responsible for deciding which images you may process and for following applicable rules.
- The app does not send indexed images or embeddings to a project-operated cloud service. Model packages may download model files during setup, and web import contacts the URLs you enter. Review third-party model and website terms before use.
- Removing a face record does not by itself erase the original image file or every derived copy. Manage the configured data directory and backups as sensitive data.
- Face search can return false matches and miss correct matches. Cosine similarity is a ranking score, not a probability or identity confirmation. Do not rely on it for high-stakes decisions.
- The app searches only its indexed local collection; it is not a public-web face search service.
- No formal accuracy or fairness benchmark is currently published for this project.

## 🚀 Roadmap

- [ ] **Real-time Processing**: Live camera integration and streaming
- [ ] **Mobile App**: iOS/Android companion applications
- [ ] **Cloud Deployment**: Docker containerization and cloud deployment
- [ ] **API Gateway**: RESTful API for third-party integrations
- [ ] **Advanced Analytics**: Enhanced reporting and visualization dashboard
- [ ] **Multi-language Support**: Interface internationalization

## 📏 Performance and Evaluation

This repository has LFW diagnostic measurements but no benchmark against PimEyes or a 75,000-face collection. The app searches images indexed in the local database. Runtime and retrieval quality depend on the hardware and collection; treat similarity scores as ranking signals and review matches manually.

To evaluate a local labeled collection, put images in one subdirectory per identity and run:

```bash
python evaluate_face_search.py data/demo_images
```

The report includes face-detection coverage, Top-1/Top-5 retrieval, and same-identity recall versus different-identity false-accept rate at selected cosine thresholds. The included 32-image sample returned 100% Top-1 and Top-5 retrieval. Images with zero or multiple detected faces are excluded from identity scoring. This small-data evaluation is not a substitute for an external benchmark.

## 🏆 Community & Stats

- **🍴 Active Development**: Contributing to open-source innovation
- **🐛 Issue Tracking**: Responsive maintenance and bug fixes
- **👥 Contributors**: Welcome developers of all skill levels

## 📄 License

There is currently no `LICENSE` file in this repository. Check the repository owner's licensing terms before redistribution.

## 🙏 Acknowledgments

Special thanks to the amazing open-source community:

- **[DeepFace](https://github.com/serengil/deepface)** - Facial analysis framework
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[ChromaDB](https://www.trychroma.com/)** - Vector database for embeddings
- **[OpenCV](https://opencv.org/)** - Computer vision library

## 🌐 Community & Support

- **[GitHub Discussions](https://github.com/SchBenedikt/face/discussions)** - Ask questions and share ideas
- **[Issues](https://github.com/SchBenedikt/face/issues)** - Report bugs and request features
- **[Wiki](https://github.com/SchBenedikt/face/wiki)** - Comprehensive documentation

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star! ⭐**

*Made with ❤️ for the Computer Vision and AI community*

[🔝 Back to Top](#-advanced-face-recognition-search-system)

</div>
