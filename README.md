# 🔍 Advanced Face Recognition Search System

<div align="center">

![Python](https://img.shields.io/badge/python-v3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.49+-red.svg)
![Issues](https://img.shields.io/github/issues/SchBenedikt/face.svg)

**Local face search with Facenet512 embeddings, aligned face crops, and a ChromaDB index**

[📖 Documentation](#-documentation) • [⚡ Quick Start](#-quick-start) • [🎯 Features](#-key-features) • [🧠 Models and Search](#-models-and-search) • [🛠️ Installation](#-installation) • [🤝 Contributing](#-contributing)

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
update, only already reprocessed images are searched.

## 🌟 What Makes This Special?

The application provides local face search over images you choose to index:

- **🧠 Face Search**: Facenet512 embeddings from DeepFace-detected faces
- **⚡ Batch Processing**: Index new images or refresh a selected image directory
- **🎯 Search Filtering**: Adjustable cosine similarity threshold and result filters
- **🔍 Vector Search**: Similarity search across the locally indexed image collection
- **📊 Face Analysis**: Optional age, gender, emotion, and ethnicity estimates
- **🌐 Image Crawler**: Optional interactive image collection from websites
- **👥 Person Management**: Intelligent name assignment, reviewed identity groups and automatic propagation

## 🎯 Key Features

### 🔍 **Advanced Search Engine**
- **Similarity Search**: Find faces similar to uploaded images
- **Result Filters**: Filter displayed results by cosine similarity, face size, or assigned name
- **Smart Modes**: High Precision, High Recall, or Balanced search algorithms
- **Real-time Filtering**: Dynamic result filtering by similarity, size, and assigned name

### 🧠 **AI-Powered Analysis**
- **Facial Recognition**: Facenet512 embeddings with RetinaFace-first detection and aligned face crops
- **Attribute Detection**: Age, gender, emotion, ethnicity analysis
- **Search Scores**: Cosine similarity is shown as a ranking score, not a match probability
- **Search Threshold**: Defaults to 0.70 cosine similarity; adjust it to trade precision for recall

### ⚡ **High-Performance Processing**
- **Batch Processing**: Process image directories in configurable batches
- **Incremental Updates**: Skip already indexed images or refresh a directory
- **Smart Caching**: Cache face thumbnails and selected image processing results
- **Local Index**: Keep embeddings and image paths in the configured local data directory

### 🧩 **Personen & Gruppierung**
- **Identitätsgruppen**: Unbenannte Gesichter werden per Cosine-Ähnlichkeit gruppiert (Facenet512-Schwelle ~0,68) und erst nach Review zugewiesen
- **Namensverwaltung**: Manuelle Zuordnung, automatische Verbreitung mit Ambiguitätsschutz
- **Kalibrierte Ähnlichkeit**: ChromaDBs quadrierte L2-Distanz wird korrekt in Cosine umgerechnet

### 🌐 **Data Collection & Management**
- **Web Scraping**: Optional interactive image crawler
- **Duplicate Detection**: Find image-file variants before indexing
- **Metadata Management**: Store image paths, face boxes, and assigned person names
- **Image Cleanup**: Find and remove duplicate image-file variants

## 🧠 Models and Search

### Face detection and embeddings

- The optional `face_recognition` package is tried first when installed. DeepFace detection then tries RetinaFace, MTCNN, and OpenCV; Haar cascades are the final fallback.
- Face crops from DeepFace detectors are aligned and reused for both embedding extraction and the result preview.
- The active recognition pipeline extracts one 512-dimensional Facenet512 embedding per detected face. Model candidates and ensemble helpers elsewhere in the code are legacy configuration; search does not combine four models.
- Embeddings are unit-normalized and stored in the local ChromaDB collection. Search converts ChromaDB distances to cosine similarity and ranks indexed images.
- The default search threshold is 0.70 cosine similarity. This is a starting threshold, not a calibrated probability that two images show the same person. DeepFace's Facenet512 verification threshold is a cosine distance of 0.30, corresponding to cosine similarity 0.70 for normalized vectors. See [DeepFace's threshold configuration](https://github.com/serengil/deepface/blob/master/deepface/config/threshold.py).
- Search covers only images indexed in this application's database; it does not search the public web.

### Index updates

Changes to face detection, alignment, or embedding preprocessing create a new pipeline version. The app excludes older vectors until their source images are reprocessed. Use **Batch Face Processing → Update All Images** for each image directory. Back up the database before a large reindex.

This repository has no measured LFW accuracy, web-scale retrieval benchmark, or PimEyes comparison. Search quality depends on the source images, face size, pose, lighting, and indexed collection.

## 🛠️ Installation

### Prerequisites
- **Python 3.11+** (recommended for best performance)
- **8GB+ RAM** (for large datasets)
- **GPU support** (optional, but recommended for faster processing)

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
# Install test and formatting tools
pip install pytest black flake8

# Run tests
python -m pytest tests/
```

## 📥 Data Processing Workflow

### Add and index images

1. Place images in `data/images` (or in `images` under the configured `FACE_DATA_DIR`).
2. Run `python fast_process.py` to index new images. Use `python fast_process.py --update` to reprocess that directory after changing the recognition pipeline.
3. To process scraped images or another directory, use **Batch Face Processing** in the Streamlit app, select the directory, and choose **Process New Images Only** or **Update All Images**.
4. Start the interface with `streamlit run app.py`, open **Face Search**, and upload an image containing a face.

The `image-download.py` crawler is interactive. Run `python image-download.py`, review its prompts, then index the downloaded images with the steps above. It makes requests to external websites; follow their access rules.

### Remove duplicate image files

`clean_duplicates.py` reports image-file variants by default. Review the report before deleting anything:

```bash
python clean_duplicates.py --dir data/images --report
```

To remove detected image-file duplicates after review, add `--clean`. This cleanup concerns image files; it does not verify embedding quality or identity matches.

### Settings and data location

`config.py` defines the active detector order, Facenet512 model, default search threshold (0.70), and ChromaDB path. `FACE_DATA_DIR` relocates the images and database together. The CLI batch processor handles `IMAGES_DIR`; use the app's batch page to update additional directories.

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
            → Facenet512 embedding → ChromaDB cosine-ranked search
```

Image crawling is a separate optional workflow. Search runs against the local ChromaDB collection and indexed image paths.

### Core Components

- **`app.py`** - Main Streamlit web application with modern UI
- **`face_recognition_engine.py`** - Face detection, aligned crops, and Facenet512 embeddings
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

- Images and face embeddings are stored under the configured local data directory. The optional image crawler makes requests to external websites.
- The app has no built-in consent tracking, legal-compliance checks, or demographic-bias evaluation. You are responsible for deciding which images you may process and for following applicable rules.
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

This repository does not publish a benchmark against LFW, PimEyes, or a 75,000-face collection. The app searches images indexed in the local database. Runtime and retrieval quality depend on the hardware and collection; treat similarity scores as ranking signals and review matches manually.

## 🏆 Community & Stats

- **⭐ Open Source**: Join our community of developers and researchers
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
