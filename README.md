# vector_database
A high-performance vector database completely from scratch using Python and NumPy, featuring exact brute-force search, a custom IVF-Flat (Inverted File Flat) index with custom K-Means clustering, CRUD API (insert, search, delete), automated recall vs speed benchmarking over 50,000+ vectors, and an interactive Streamlit visualizer.
Zero external indexing libraries: All distance metrics, vector normalization, K-Means clustering, centroid assignment, posting lists, and top-k search will be written manually using only basic numpy operations.
Deletion Semantics: In IVF-Flat, deletion involves looking up the posting list for the vector ID and removing it from the posting list array/dict. If we also provide HNSW, we will explain the trade-offs of tombstoning vs graph re-wiring.
Dataset Size: The benchmark will run on 50,000 vectors (synthetic + real text embeddings d=384) with 500 query vectors to evaluate exact Ground Truth vs IVF-Flat.

Proposed Architecture & Design
1. Mathematical Fundamentals & Distance Metric
2.  Custom Indexing Implementations  (BruteForceIndex)
   1. Custom IVF-Flat Index (IVFFlatIndex)
      1. Training Phase (fit)
      2. Inverted File Posting Lists (posting_lists)
      3. Search Phase (search(query, k, nprobe)):
      4. CRUD Operations:
3.Text Embedding & Synthetic Generator (src/text_encoder.py & src/synthetic_data.py)
4. Automated Benchmark Suite (benchmark.py)
5. Interactive Visual UI (app.py Streamlit App)

Verification Plan
1. Automated Tests
Run pytest tests/test_vector_db.py to verify:
BruteForceIndex exact dot product top-k matches.
IVFFlatIndex clustering and lookup.
insert() adds vector to posting list and makes it searchable.
delete() removes vector from index and confirms it no longer appears in search results.
50,000 vector index initialization & memory efficiency.

2. Benchmark Execution
Run python benchmark.py:
Build 50,000 vector ground truth index + IVF-Flat index (n_clusters = 256).
Compute top-10 recall and latency over 500 query vectors for nprobe from 1 to 256.
Verify Recall@10 increases monotonically with nprobe up to 1.0 (100%).

3. Manual & Visual Verification
Run streamlit run app.py to test:
Semantic text search on real text queries.
Cluster visualization and probed cluster highlight.
Real-time insertion and deletion of documents.
