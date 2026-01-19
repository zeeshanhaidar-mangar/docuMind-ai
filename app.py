import streamlit as st
import os
import json
from datetime import datetime
from rag_utils import RAGPipeline
from advanced_features import (
    generate_summary,
    create_knowledge_graph,
    show_analytics_dashboard,
    cluster_chunks,
    rerank_results,
    translate_text,
    hybrid_search
)

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Q&A System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()
if 'feedback_log' not in st.session_state:
    st.session_state.feedback_log = []
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []

# Title and description
st.title("📚 Advanced RAG Question-Answering System")
st.markdown("""
Upload documents (PDF/TXT), ask questions, and get intelligent answers with source citations.
Features include hybrid search, multi-language support, knowledge graphs, and analytics.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Search method selection
    search_method = st.radio(
        "Search Method",
        ["Semantic Only", "Hybrid (BM25 + Semantic)"],
        help="Choose between pure semantic search or hybrid approach"
    )
    
    # Number of chunks to retrieve
    top_k = st.slider("Number of chunks to retrieve", 3, 10, 5)
    
    # Language support
    target_language = st.selectbox(
        "Output Language",
        ["English", "Spanish", "French", "German", "Chinese"],
        help="Translate answers to your preferred language"
    )
    
    # Content filters
    st.subheader("📋 Content Filters")
    filter_keyword = st.text_input("Filter by keyword (optional)")
    min_length = st.number_input("Min chunk length", 0, 1000, 0)
    max_length = st.number_input("Max chunk length", 0, 5000, 5000)
    
    # Advanced features toggle
    st.subheader("🔧 Advanced Features")
    use_reranking = st.checkbox("Use Re-ranking", value=True)
    show_context = st.checkbox("Highlight Context", value=True)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload & Query", "📊 Analytics", "🕸️ Knowledge Graph", "🔍 Clustering"])

with tab1:
    # File upload section
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    # Save file temporarily
                    file_path = os.path.join("temp", file.name)
                    os.makedirs("temp", exist_ok=True)
                    
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Process document
                    success = st.session_state.rag_pipeline.process_document(
                        file_path, 
                        file.name
                    )
                    
                    if success:
                        st.session_state.uploaded_docs.append({
                            'name': file.name,
                            'upload_time': datetime.now().isoformat()
                        })
                
                st.success(f"✅ Processed {len(uploaded_files)} document(s)")
    
    # Display uploaded documents
    if st.session_state.uploaded_docs:
        st.subheader("Uploaded Documents")
        for doc in st.session_state.uploaded_docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📄 {doc['name']}")
            with col2:
                if st.button(f"Summarize", key=f"sum_{doc['name']}"):
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(
                            st.session_state.rag_pipeline,
                            doc['name']
                        )
                        st.info(f"**Summary:** {summary}")
    
    # Query section
    st.header("2. Ask Questions")
    
    # Multi-document query option
    query_scope = st.radio(
        "Query Scope",
        ["All Documents", "Specific Document"],
        horizontal=True
    )
    
    specific_doc = None
    if query_scope == "Specific Document" and st.session_state.uploaded_docs:
        specific_doc = st.selectbox(
            "Select Document",
            [doc['name'] for doc in st.session_state.uploaded_docs]
        )
    
    query = st.text_area("Enter your question:", height=100)
    
    if st.button("🔍 Search", type="primary"):
        if query and st.session_state.rag_pipeline.chunks:
            with st.spinner("Searching and generating answer..."):
                # Apply filters
                filtered_chunks = st.session_state.rag_pipeline.chunks
                
                if filter_keyword:
                    filtered_chunks = [
                        c for c in filtered_chunks 
                        if filter_keyword.lower() in c['text'].lower()
                    ]
                
                filtered_chunks = [
                    c for c in filtered_chunks 
                    if min_length <= len(c['text']) <= max_length
                ]
                
                if specific_doc:
                    filtered_chunks = [
                        c for c in filtered_chunks 
                        if c['metadata']['source'] == specific_doc
                    ]
                
                # Perform search
                if search_method == "Hybrid (BM25 + Semantic)":
                    results = hybrid_search(
                        st.session_state.rag_pipeline,
                        query,
                        filtered_chunks,
                        top_k
                    )
                else:
                    results = st.session_state.rag_pipeline.retrieve(
                        query,
                        top_k,
                        filtered_chunks
                    )
                
                # Re-rank if enabled
                if use_reranking and results:
                    results = rerank_results(query, results)
                
                # Generate answer
                answer = st.session_state.rag_pipeline.generate_answer(
                    query,
                    results
                )
                
                # Translate if needed
                if target_language != "English":
                    answer = translate_text(answer, target_language)
                
                # Display answer
                st.subheader("💡 Answer")
                if show_context and results:
                    # Highlight context in answer
                    highlighted_answer = answer
                    for result in results[:2]:
                        snippet = result['text'][:50]
                        highlighted_answer = highlighted_answer.replace(
                            snippet,
                            f"**{snippet}**"
                        )
                    st.markdown(highlighted_answer)
                else:
                    st.write(answer)
                
                # Display sources
                if results:
                    st.subheader("📎 Sources")
                    for i, result in enumerate(results, 1):
                        with st.expander(
                            f"Source {i}: {result['metadata']['source']} "
                            f"(Similarity: {result['score']:.3f})"
                        ):
                            st.write(result['text'])
                            st.caption(
                                f"Chunk {result['metadata']['chunk_id']} | "
                                f"Length: {len(result['text'])} chars"
                            )
                
                # Feedback section
                st.subheader("📝 Feedback")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("👍 Helpful"):
                        feedback_entry = {
                            'query': query,
                            'answer': answer,
                            'feedback': 'positive',
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.feedback_log.append(feedback_entry)
                        st.session_state.query_history.append({
                            'query': query,
                            'timestamp': datetime.now().isoformat(),
                            'num_results': len(results)
                        })
                        st.success("Thanks for your feedback!")
                
                with col2:
                    if st.button("👎 Not Helpful"):
                        feedback_entry = {
                            'query': query,
                            'answer': answer,
                            'feedback': 'negative',
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.feedback_log.append(feedback_entry)
                        st.session_state.query_history.append({
                            'query': query,
                            'timestamp': datetime.now().isoformat(),
                            'num_results': len(results)
                        })
                        st.warning("Thanks for your feedback. We'll improve!")
        else:
            st.warning("Please upload documents and enter a question.")

with tab2:
    st.header("📊 Query Analytics Dashboard")
    if st.session_state.query_history:
        show_analytics_dashboard(
            st.session_state.query_history,
            st.session_state.feedback_log
        )
    else:
        st.info("No queries yet. Start asking questions to see analytics!")

with tab3:
    st.header("🕸️ Knowledge Graph Visualization")
    if st.session_state.rag_pipeline.chunks:
        if st.button("Generate Knowledge Graph"):
            with st.spinner("Building knowledge graph..."):
                graph_html = create_knowledge_graph(
                    st.session_state.rag_pipeline.chunks
                )
                st.components.v1.html(graph_html, height=600, scrolling=True)
    else:
        st.info("Upload documents to generate a knowledge graph.")

with tab4:
    st.header("🔍 Semantic Clustering")
    if st.session_state.rag_pipeline.chunks:
        num_clusters = st.slider("Number of clusters", 2, 10, 5)
        if st.button("Perform Clustering"):
            with st.spinner("Clustering chunks..."):
                fig = cluster_chunks(
                    st.session_state.rag_pipeline,
                    num_clusters
                )
                st.pyplot(fig)
    else:
        st.info("Upload documents to perform clustering analysis.")

# Footer
st.markdown("---")
st.markdown(
    "Built with Streamlit, Sentence Transformers, FAISS, and Hugging Face | "
    "All models are free and open-source"
)
