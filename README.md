# Vector Database from Scratch (NumPy & Gradio)

A high-performance vector database built completely from scratch using **Python** and **NumPy**, featuring exact brute-force search, a custom IVF-Flat (Inverted File Flat) index with K-Means clustering, CRUD API (insert, search, delete), automated recall vs. speed benchmarking, and an interactive **Gradio** web UI.

Zero external indexing libraries: All distance metrics, vector normalization, K-Means clustering, centroid assignment, posting lists, and top-k search are written using NumPy operations.

## Features
- **Pure NumPy Implementation**: Cosine similarity search without external vector DB dependencies.
- **Indexing Options**:
  - **Exact Brute Force**: Ground truth nearest neighbor search.
  - **IVF-Flat Indexing**: Inverted File index with K-Means clustering for fast approximate search.
- **Interactive UI**: Gradio application (`app.py`) for live document embedding, insertion, and semantic search using `sentence-transformers`.
- **Benchmarking**: Included benchmark script to evaluate search speedup vs. recall accuracy over 50,000+ vectors.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark
```bash
python vector_db.py
```

### 3. Launch the Gradio Web App
```bash
python app.py
```
