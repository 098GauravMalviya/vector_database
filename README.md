# Vector Database from Scratch (NumPy & Gradio)

A custom lightweight vector database implemented purely using **NumPy**, featuring exact brute-force search, approximate IVF-Flat indexing, and an interactive **Gradio** web UI.

## Features
- **Pure NumPy Implementation**: Cosine similarity search without external vector DB dependencies.
- **Indexing Options**:
  - **Exact Brute Force**: Ground truth nearest neighbor search.
  - **IVF-Flat Indexing**: Inverted File index with K-Means clustering for fast approximate search.
- **Interactive UI**: Gradio application (`app.py`) for live document embedding, insertion, and semantic search using `sentence-transformers`.
- **Benchmarking**: Included benchmark script to evaluate search speedup vs. recall accuracy.

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
