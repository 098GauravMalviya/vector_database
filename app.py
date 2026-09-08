import gradio as gr
from vector_db import VectorDatabase
from sentence_transformers import SentenceTransformer
import time

print("Loading embedding model (this may take a few seconds on startup)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a sample corpus to pre-load
sample_corpus = [
    "Machine learning relies heavily on linear algebra and calculus.",
    "I need to walk my dog in the park.",
    "Artificial intelligence is transforming software engineering.",
    "The weather is really nice today.",
    "Vector databases are essential for modern AI applications.",
    "I love programming in Python.",
    "Retrieval-Augmented Generation (RAG) uses vectors to find context.",
    "The stock market saw a massive increase in tech shares.",
    "I bought some fresh vegetables from the farmers market.",
    "Neural networks process data through multiple layers of nodes."
]

# Initialize Database (MiniLM outputs 384 dimensional vectors)
db = VectorDatabase(d=384, n_clusters=3)
corpus = []

# Preload data
print("Pre-loading sample data into the database...")
for text in sample_corpus:
    emb = model.encode(text)
    db.insert(emb, vector_id=len(corpus))
    corpus.append(text)

# Build the IVF index
db.build_ivf_index(max_iters=10)
print("Database ready!")

def insert_document(text):
    if text.strip() == "":
        return "Please enter valid text."
    
    emb = model.encode(text)
    idx = len(corpus)
    corpus.append(text)
    db.insert(emb, vector_id=idx)
    
    # Periodically rebuild index if we get enough new documents
    if len(corpus) % 5 == 0:
        db.n_clusters = max(3, len(corpus) // 5)
        db.build_ivf_index()
        
    return f"Inserted successfully! Total documents in database: {len(corpus)}"

def search(query, method):
    if not corpus:
        return "Database is empty."
    if query.strip() == "":
        return "Please enter a query."
        
    emb = model.encode(query)
    
    t0 = time.perf_counter()
    if method == "Approximate (IVF-Flat)":
        # Search using approx index
        n_probes = max(1, db.n_clusters // 2)
        results = db.ivf_flat_search(emb, top_k=3, n_probes=n_probes)
    else:
        # Search using brute force ground truth
        results = db.brute_force_search(emb, top_k=3)
        
    t1 = time.perf_counter()
    
    search_time_ms = (t1 - t0) * 1000
    
    output = f"Search completed in {search_time_ms:.3f} ms using {method}.\n"
    output += "-" * 50 + "\n\n"
    
    for rank, (idx, score) in enumerate(results, 1):
        output += f"#{rank} | Score: {score:.4f}\n{corpus[idx]}\n\n"
        
    return output

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🚀 NumPy Vector Database Demo")
    gr.Markdown("A custom vector database built from scratch using purely NumPy. It supports both **O(N) Brute Force** searching and **Approximate IVF-Flat** indexing.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Insert Data")
            new_doc = gr.Textbox(label="Document Text", placeholder="Type a sentence to add to the database...")
            insert_btn = gr.Button("Insert into Vector DB", variant="secondary")
            insert_status = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column():
            gr.Markdown("### 2. Search Data")
            query = gr.Textbox(label="Search Query", placeholder="What are you looking for?")
            search_method = gr.Radio(
                ["Exact (Brute Force)", "Approximate (IVF-Flat)"], 
                label="Search Algorithm", 
                value="Exact (Brute Force)"
            )
            search_btn = gr.Button("Search", variant="primary")
            search_results = gr.Textbox(label="Top Results", lines=10, interactive=False)

    insert_btn.click(fn=insert_document, inputs=new_doc, outputs=insert_status)
    search_btn.click(fn=search, inputs=[query, search_method], outputs=search_results)

if __name__ == "__main__":
    interface.launch(server_port=7860, share=False)
