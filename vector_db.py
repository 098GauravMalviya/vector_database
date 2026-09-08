import numpy as np
import time
from collections import defaultdict

class VectorDatabase:
    """
    A pure-NumPy Vector Database implementing Exact (Brute Force) and 
    Approximate (IVF-Flat) nearest neighbor search using cosine similarity.
    """
    def __init__(self, d, n_clusters=100):
        self.d = d
        self.vectors = {} # mapping id to normalized vector
        self.n_clusters = n_clusters
        
        # IVF-Flat structures
        self.centroids = None
        # Mapping from cluster_id to list of vector_ids
        self.inverted_index = defaultdict(list)
        # Mapping from vector_id to cluster_id
        self.vector_to_cluster = {}
        
        self._next_id = 0
        
    def _normalize(self, v):
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        # avoid division by zero
        norm = np.where(norm == 0, 1, norm)
        return v / norm

    def insert(self, vector, vector_id=None):
        """Insert a single vector into the DB."""
        if vector_id is None:
            vector_id = self._next_id
            self._next_id += 1
            
        vector = np.array(vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector[np.newaxis, :]
            
        # Pre-normalize for fast cosine similarity (dot product)
        vector = self._normalize(vector)[0]
        self.vectors[vector_id] = vector
        
        # If the IVF index is already built, assign to the nearest cluster
        if self.centroids is not None:
            similarities = np.dot(self.centroids, vector)
            cluster_id = np.argmax(similarities)
            self.inverted_index[cluster_id].append(vector_id)
            self.vector_to_cluster[vector_id] = cluster_id
            
        return vector_id

    def insert_batch(self, vectors, vector_ids=None):
        """Utility to insert multiple vectors at once."""
        vectors = np.array(vectors, dtype=np.float32)
        vectors = self._normalize(vectors)
        
        if vector_ids is None:
            vector_ids = list(range(self._next_id, self._next_id + len(vectors)))
            self._next_id += len(vectors)
            
        for vid, v in zip(vector_ids, vectors):
            self.vectors[vid] = v

    def delete(self, vector_id):
        """Delete a vector by id."""
        if vector_id in self.vectors:
            del self.vectors[vector_id]
            # Remove from IVF index if built
            if self.centroids is not None and vector_id in self.vector_to_cluster:
                cluster_id = self.vector_to_cluster[vector_id]
                self.inverted_index[cluster_id].remove(vector_id)
                del self.vector_to_cluster[vector_id]
            return True
        return False

    def brute_force_search(self, query_vector, top_k=10):
        """
        Exact nearest neighbor search using cosine similarity.
        O(N) time complexity. This is our ground truth.
        """
        if not self.vectors:
            return []
            
        query_vector = np.array(query_vector, dtype=np.float32)
        query_vector = self._normalize(query_vector)
        
        if query_vector.ndim == 1:
            query_vector = query_vector[np.newaxis, :]
            
        # Extract all vectors and their ids
        ids = list(self.vectors.keys())
        matrix = np.array([self.vectors[i] for i in ids])
        
        # Calculate exact dot product with entire dataset
        similarities = np.dot(query_vector, matrix.T)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(ids[i], similarities[i]) for i in top_indices]

    def build_ivf_index(self, max_iters=20):
        """Build the IVF-Flat index using K-Means clustering."""
        if not self.vectors:
            return
            
        ids = list(self.vectors.keys())
        matrix = np.array([self.vectors[i] for i in ids])
        
        n_samples = len(matrix)
        if n_samples < self.n_clusters:
            self.n_clusters = max(1, n_samples)
            
        # 1. Initialize centroids randomly from the dataset
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = matrix[random_indices].copy()
        
        # 2. Run K-Means iterations
        for _ in range(max_iters):
            # Compute similarities to centroids
            similarities = np.dot(matrix, self.centroids.T)
            
            # Assign points to closest centroid
            cluster_assignments = np.argmax(similarities, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            counts = np.zeros(self.n_clusters)
            
            for i, cluster_id in enumerate(cluster_assignments):
                new_centroids[cluster_id] += matrix[i]
                counts[cluster_id] += 1
                
            # Handle empty clusters by keeping old centroid
            for c in range(self.n_clusters):
                if counts[c] > 0:
                    new_centroids[c] /= counts[c]
                else:
                    new_centroids[c] = self.centroids[c]
                    
            self.centroids = self._normalize(new_centroids)
            
        # 3. Build inverted index based on final centroids
        self.inverted_index = defaultdict(list)
        self.vector_to_cluster = {}
        
        similarities = np.dot(matrix, self.centroids.T)
        cluster_assignments = np.argmax(similarities, axis=1)
        
        for vid, cid in zip(ids, cluster_assignments):
            self.inverted_index[cid].append(vid)
            self.vector_to_cluster[vid] = cid

    def ivf_flat_search(self, query_vector, top_k=10, n_probes=1):
        """
        Approximate nearest neighbor search using IVF-Flat.
        Searches the nearest `n_probes` clusters instead of the whole dataset.
        """
        if self.centroids is None:
            raise ValueError("IVF index is not built. Call build_ivf_index() first.")
            
        query_vector = np.array(query_vector, dtype=np.float32)
        query_vector = self._normalize(query_vector)
        
        if query_vector.ndim == 1:
            query_vector = query_vector[np.newaxis, :]
            
        # 1. Find nearest centroids
        centroid_similarities = np.dot(query_vector, self.centroids.T)[0]
        
        # Pick top n_probes clusters to search within
        n_probes = min(n_probes, self.n_clusters)
        best_clusters = np.argsort(centroid_similarities)[::-1][:n_probes]
        
        # 2. Collect candidate vectors from the selected clusters (buckets)
        candidate_ids = []
        for cluster_id in best_clusters:
            candidate_ids.extend(self.inverted_index[cluster_id])
            
        if not candidate_ids:
            return []
            
        # 3. Perform exact search over the candidates ONLY
        candidate_matrix = np.array([self.vectors[i] for i in candidate_ids])
        
        similarities = np.dot(query_vector, candidate_matrix.T)[0]
        
        top_k = min(top_k, len(candidate_ids))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(candidate_ids[i], similarities[i]) for i in top_indices]


def generate_synthetic_data(n_samples, d, n_clusters, random_state=42):
    """Generates a dataset of clustered vectors to simulate real embeddings."""
    np.random.seed(random_state)
    
    # Generate random centroids
    centroids = np.random.randn(n_clusters, d)
    centroids = centroids / np.linalg.norm(centroids, axis=-1, keepdims=True)
    
    # Generate vectors around centroids
    vectors = []
    samples_per_cluster = n_samples // n_clusters
    
    for c in centroids:
        # Add gaussian noise to the centroid to create a cluster
        cluster_vectors = c + np.random.randn(samples_per_cluster, d) * 0.15
        vectors.extend(cluster_vectors)
        
    # Handle remainder if not exactly divisible
    if len(vectors) < n_samples:
        rem = n_samples - len(vectors)
        c = centroids[0]
        cluster_vectors = c + np.random.randn(rem, d) * 0.15
        vectors.extend(cluster_vectors)
        
    return np.array(vectors)

def run_benchmark():
    n_db_vectors = 50000
    n_query_vectors = 500
    d = 128
    n_clusters = 100
    top_k = 10
    n_probes = 5 # Number of clusters to probe during approx search
    
    print(f"Generating synthetic clustered dataset of {n_db_vectors} vectors (dim={d})...")
    # Underlying data has 20 natural clusters, we'll partition it into 100 for IVF
    db_vectors = generate_synthetic_data(n_db_vectors, d, n_clusters=20)
    
    print(f"Generating {n_query_vectors} query vectors...")
    query_vectors = np.random.randn(n_query_vectors, d)
    
    db = VectorDatabase(d=d, n_clusters=n_clusters)
    
    print("Inserting vectors into Database...")
    db.insert_batch(db_vectors)
    
    print(f"Building IVF-Flat index (K-Means with {n_clusters} clusters)...")
    t0 = time.time()
    db.build_ivf_index(max_iters=15)
    print(f"Index built in {time.time() - t0:.2f} seconds.")
    
    print("\n--- Running Benchmark ---")
    
    # Brute Force Search (Ground Truth)
    print("Running Exact Brute Force Search...")
    t0 = time.time()
    bf_results = []
    for q in query_vectors:
        res = db.brute_force_search(q, top_k=top_k)
        bf_results.append([r[0] for r in res])
    bf_time = time.time() - t0
    print(f"Brute Force took: {bf_time:.4f} seconds.")
    
    # IVF-Flat Search (Approximate)
    print(f"Running Approx IVF-Flat Search (probing top {n_probes} clusters)...")
    t0 = time.time()
    ivf_results = []
    for q in query_vectors:
        res = db.ivf_flat_search(q, top_k=top_k, n_probes=n_probes)
        ivf_results.append([r[0] for r in res])
    ivf_time = time.time() - t0
    print(f"IVF-Flat took: {ivf_time:.4f} seconds.")
    
    # Calculate Speedup and Recall
    speedup = bf_time / ivf_time
    print(f"\nResults:")
    print(f"Speedup Factor: {speedup:.2f}x faster")
    
    # Recall @ k
    total_matches = 0
    for bf_res, ivf_res in zip(bf_results, ivf_results):
        # Count how many of the exact top-k are in the approx top-k
        matches = set(bf_res).intersection(set(ivf_res))
        total_matches += len(matches)
        
    recall = total_matches / (n_query_vectors * top_k)
    print(f"Recall@{top_k}: {recall * 100:.2f}% (Accuracy of the approximation)")
    
    print("\n--- Testing API (Insert/Delete) ---")
    new_vec = np.random.randn(d)
    v_id = db.insert(new_vec)
    print(f"Inserted new vector with ID {v_id}. DB Size: {len(db.vectors)}")
    db.delete(v_id)
    print(f"Deleted vector with ID {v_id}. DB Size: {len(db.vectors)}")

if __name__ == '__main__':
    run_benchmark()
