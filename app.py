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
corpus = {}

# Preload data
print("Pre-loading sample data into the database...")
for idx, text in enumerate(sample_corpus):
    emb = model.encode(text)
    db.insert(emb, vector_id=idx)
    corpus[idx] = text

# Build the IVF index
db.build_ivf_index(max_iters=10)
print("Database ready!")

def get_doc_choices():
    return [f"ID {v_id}: {text[:40]}..." for v_id, text in corpus.items() if text is not None]

def insert_document(text):
    if text.strip() == "":
        return "Please enter valid text.", gr.update(choices=get_doc_choices())
    
    # Next available ID
    idx = max(corpus.keys()) + 1 if corpus else 0
    emb = model.encode(text)
    corpus[idx] = text
    db.insert(emb, vector_id=idx)
    
    # Periodically rebuild index if we get enough new documents
    if len(corpus) % 5 == 0:
        db.n_clusters = max(3, len(corpus) // 5)
        db.build_ivf_index()
        
    return f"Inserted successfully! Document ID: {idx}. Total active documents: {len(db.vectors)}", gr.update(choices=get_doc_choices())

def delete_document(selected_doc):
    if not selected_doc:
        return "Please select a document to delete.", gr.update(choices=get_doc_choices())
    
    try:
        # Extract doc ID from choice string e.g. "ID 3: ..."
        doc_id = int(selected_doc.split(":")[0].replace("ID ", "").strip())
    except Exception:
        return "Invalid document selection.", gr.update(choices=get_doc_choices())

    if doc_id in db.vectors:
        db.delete(doc_id)
        deleted_text = corpus.pop(doc_id, "Unknown")
        return f"Deleted Document ID {doc_id} ('{deleted_text[:30]}...'). Total active documents: {len(db.vectors)}", gr.update(choices=get_doc_choices(), value=None)
    else:
        return f"Document ID {doc_id} not found.", gr.update(choices=get_doc_choices())

def search(query, method):
    if not db.vectors:
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
        doc_text = corpus.get(idx, "[Deleted Document]")
        output += f"#{rank} | Doc ID: {idx} | Score: {score:.4f}\n{doc_text}\n\n"
        
    return output

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🚀 NumPy Vector Database Demo")
    gr.Markdown("A custom vector database built from scratch using purely NumPy. Supports **Insert**, **Delete**, **Brute Force**, and **IVF-Flat** indexing.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Insert Data")
            new_doc = gr.Textbox(label="Document Text", placeholder="Type a sentence to add to the database...")
            insert_btn = gr.Button("Insert into Vector DB", variant="secondary")
            insert_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("### 2. Delete Data")
            delete_dropdown = gr.Dropdown(choices=get_doc_choices(), label="Select Document to Delete")
            delete_btn = gr.Button("Delete Document", variant="stop")
            delete_status = gr.Textbox(label="Deletion Status", interactive=False)
            
        with gr.Column():
            gr.Markdown("### 3. Search Data")
            query = gr.Textbox(label="Search Query", placeholder="What are you looking for?")
            search_method = gr.Radio(
                ["Exact (Brute Force)", "Approximate (IVF-Flat)"], 
                label="Search Algorithm", 
                value="Exact (Brute Force)"
            )
            search_btn = gr.Button("Search", variant="primary")
            search_results = gr.Textbox(label="Top Results", lines=12, interactive=False)

    insert_btn.click(fn=insert_document, inputs=new_doc, outputs=[insert_status, delete_dropdown])
    delete_btn.click(fn=delete_document, inputs=delete_dropdown, outputs=[delete_status, delete_dropdown])
    search_btn.click(fn=search, inputs=[query, search_method], outputs=search_results)

if __name__ == "__main__":
    interface.launch(server_port=7860, share=False)
