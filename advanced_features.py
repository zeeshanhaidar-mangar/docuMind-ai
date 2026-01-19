import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import spacy
import networkx as nx
from pyvis.network import Network
from rank_bm25 import BM25Okapi
from transformers import pipeline
import re

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Translation pipeline
translator = None

def get_translator():
    """Lazy load translator."""
    global translator
    if translator is None:
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    return translator

def hybrid_search(
    rag_pipeline,
    query: str,
    chunks: List[Dict],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Combine BM25 and semantic search for hybrid retrieval."""
    if not chunks:
        return []
    
    # BM25 search
    tokenized_chunks = [chunk['text'].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Semantic search
    semantic_results = rag_pipeline.retrieve(query, top_k * 2, chunks)
    
    # Create score dictionaries
    semantic_scores = {
        result['metadata']['chunk_id']: result['score']
        for result in semantic_results
    }
    
    # Combine scores (weighted average)
    combined_scores = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk['metadata']['chunk_id']
        bm25_score = bm25_scores[i]
        semantic_score = semantic_scores.get(chunk_id, 0.0)
        
        # Normalize and combine (70% semantic, 30% BM25)
        combined = 0.7 * semantic_score + 0.3 * (bm25_score / (max(bm25_scores) + 1e-6))
        combined_scores.append((i, combined))
    
    # Sort and get top-k
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, score in combined_scores[:top_k]:
        chunk = chunks[idx].copy()
        chunk['score'] = score
        results.append(chunk)
    
    return results

def rerank_results(query: str, results: List[Dict]) -> List[Dict]:
    """Re-rank results using cross-encoder (simplified version)."""
    # Simple re-ranking based on query term overlap
    query_terms = set(query.lower().split())
    
    for result in results:
        chunk_terms = set(result['text'].lower().split())
        overlap = len(query_terms & chunk_terms)
        # Boost score based on term overlap
        result['score'] *= (1 + 0.1 * overlap)
    
    # Re-sort by updated scores
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def translate_text(text: str, target_language: str) -> str:
    """Translate text to target language."""
    lang_codes = {
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh"
    }
    
    if target_language == "English":
        return text
    
    try:
        # For demo, we'll use a simple approach
        # In production, use proper translation models for each language pair
        if target_language in lang_codes:
            # Simplified: only works for Spanish with our loaded model
            if target_language == "Spanish":
                translator = get_translator()
                result = translator(text[:500], max_length=512)
                return result[0]['translation_text']
            else:
                return f"[Translation to {target_language}] {text}"
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def generate_summary(rag_pipeline, doc_name: str) -> str:
    """Generate a summary of a document."""
    # Get all chunks from the document
    doc_chunks = [
        chunk for chunk in rag_pipeline.chunks
        if chunk['metadata']['source'] == doc_name
    ]
    
    if not doc_chunks:
        return "No content found for this document."
    
    # Take first few chunks for summary
    sample_chunks = doc_chunks[:5]
    combined_text = " ".join([chunk['text'][:200] for chunk in sample_chunks])
    
    # Generate summary using LLM
    try:
        prompt = f"Summarize the following text in 2-3 sentences:\n\n{combined_text[:800]}\n\nSummary:"
        response = rag_pipeline.llm(
            prompt,
            max_new_tokens=80,
            num_return_sequences=1,
            temperature=0.7
        )
        
        summary = response[0]['generated_text']
        if "Summary:" in summary:
            summary = summary.split("Summary:")[-1].strip()
        else:
            summary = summary[len(prompt):].strip()
        
        # Clean and limit summary
        summary = summary.split('\n')[0][:300]
        
        if not summary or len(summary) < 20:
            summary = f"This document contains {len(doc_chunks)} sections covering: {combined_text[:150]}..."
        
        return summary
    except Exception as e:
        return f"Document overview: {combined_text[:200]}..."

def create_knowledge_graph(chunks: List[Dict]) -> str:
    """Create interactive knowledge graph from document chunks."""
    # Extract entities from chunks
    entities = []
    relationships = []
    
    for chunk in chunks[:20]:  # Limit for performance
        doc = nlp(chunk['text'][:500])
        chunk_entities = [(ent.text, ent.label_) for ent in doc.ents]
        entities.extend(chunk_entities)
        
        # Create relationships between entities in same chunk
        for i in range(len(chunk_entities)):
            for j in range(i + 1, len(chunk_entities)):
                relationships.append((chunk_entities[i][0], chunk_entities[j][0]))
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    entity_counts = {}
    for entity, label in entities:
        if entity not in entity_counts:
            entity_counts[entity] = 0
        entity_counts[entity] += 1
    
    # Add top entities
    top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    for entity, count in top_entities:
        G.add_node(entity, size=count * 10)
    
    # Add edges
    for e1, e2 in relationships:
        if e1 in G.nodes() and e2 in G.nodes():
            if G.has_edge(e1, e2):
                G[e1][e2]['weight'] += 1
            else:
                G.add_edge(e1, e2, weight=1)
    
    # Create PyVis network
    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white')
    net.from_nx(G)
    net.set_options("""
    {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -30000,
                "centralGravity": 0.3,
                "springLength": 95
            }
        }
    }
    """)
    
    # Generate HTML
    return net.generate_html()

def show_analytics_dashboard(query_history: List[Dict], feedback_log: List[Dict]):
    """Display analytics dashboard with query metrics."""
    import streamlit as st
    
    if not query_history:
        st.info("No query history available.")
        return
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Queries", len(query_history))
    
    with col2:
        positive_feedback = sum(1 for f in feedback_log if f['feedback'] == 'positive')
        st.metric("Positive Feedback", positive_feedback)
    
    with col3:
        if feedback_log:
            satisfaction_rate = (positive_feedback / len(feedback_log)) * 100
            st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
        else:
            st.metric("Satisfaction Rate", "N/A")
    
    # Query timeline
    st.subheader("Query Timeline")
    timestamps = [datetime.fromisoformat(q['timestamp']) for q in query_history]
    query_counts = list(range(1, len(timestamps) + 1))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=query_counts,
        mode='lines+markers',
        name='Queries',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(
        title="Cumulative Queries Over Time",
        xaxis_title="Time",
        yaxis_title="Number of Queries",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Results distribution
    st.subheader("Results Distribution")
    num_results = [q['num_results'] for q in query_history]
    
    fig2 = px.histogram(
        x=num_results,
        nbins=10,
        title="Distribution of Search Results",
        labels={'x': 'Number of Results', 'y': 'Frequency'},
        template="plotly_dark"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Feedback breakdown
    if feedback_log:
        st.subheader("Feedback Breakdown")
        feedback_counts = {
            'Positive': sum(1 for f in feedback_log if f['feedback'] == 'positive'),
            'Negative': sum(1 for f in feedback_log if f['feedback'] == 'negative')
        }
        
        fig3 = go.Figure(data=[go.Pie(
            labels=list(feedback_counts.keys()),
            values=list(feedback_counts.values()),
            marker_colors=['#2ecc71', '#e74c3c']
        )])
        fig3.update_layout(title="Feedback Distribution", template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

def cluster_chunks(rag_pipeline, num_clusters: int = 5):
    """Perform semantic clustering of document chunks."""
    embeddings = rag_pipeline.get_all_embeddings()
    
    if len(embeddings) < num_clusters:
        num_clusters = max(2, len(embeddings) // 2)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=cluster_labels,
        cmap='viridis',
        alpha=0.6,
        s=100
    )
    
    # Add cluster centers
    centers_2d = pca.transform(kmeans.cluster_centers_)
    ax.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c='red',
        marker='X',
        s=300,
        edgecolors='black',
        linewidths=2,
        label='Cluster Centers'
    )
    
    ax.set_title(f'Semantic Clustering of Document Chunks ({num_clusters} clusters)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID', fontsize=12)
    
    plt.tight_layout()
    
    return fig
