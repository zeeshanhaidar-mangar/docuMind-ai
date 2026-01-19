import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import PyPDF2
import re
from typing import List, Dict, Any

class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline with models and storage."""
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Storage for chunks and metadata
        self.chunks = []
        self.chunk_embeddings = []
        
        # Load LLM for answer generation
        self.llm = pipeline(
            'text-generation',
            model='distilgpt2',
            max_length=200,
            device=-1  # CPU
        )
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text content from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk) > 50:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks
    
    def process_document(self, file_path: str, filename: str) -> bool:
        """Process a document: extract text, chunk, and embed."""
        try:
            # Extract text based on file type
            if file_path.endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
            elif file_path.endswith('.txt'):
                text = self.extract_text_from_txt(file_path)
            else:
                return False
            
            if not text:
                return False
            
            # Chunk the text
            text_chunks = self.chunk_text(text)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                text_chunks,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store chunks with metadata
            for i, chunk in enumerate(text_chunks):
                self.chunks.append({
                    'text': chunk,
                    'metadata': {
                        'source': filename,
                        'chunk_id': len(self.chunks) + i,
                        'embedding_id': len(self.chunk_embeddings) + i
                    }
                })
                self.chunk_embeddings.append(embeddings[i])
            
            return True
            
        except Exception as e:
            print(f"Error processing document: {e}")
            return False
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filtered_chunks: List[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks for a query."""
        if not self.chunks:
            return []
        
        # Use filtered chunks if provided
        search_chunks = filtered_chunks if filtered_chunks is not None else self.chunks
        
        if not search_chunks:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')
        
        # Create temporary index for filtered chunks if needed
        if filtered_chunks is not None:
            filtered_embeddings = np.array([
                self.chunk_embeddings[c['metadata']['embedding_id']]
                for c in search_chunks
            ]).astype('float32')
            temp_index = faiss.IndexFlatL2(self.embedding_dim)
            temp_index.add(filtered_embeddings)
            search_index = temp_index
        else:
            search_index = self.index
        
        # Search
        k = min(top_k, len(search_chunks))
        distances, indices = search_index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(search_chunks):
                chunk = search_chunks[idx].copy()
                # Convert L2 distance to similarity score (inverse)
                chunk['score'] = 1.0 / (1.0 + distances[0][i])
                results.append(chunk)
        
        return results
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate an answer using retrieved context."""
        if not context_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context
        context = "\n\n".join([
            f"Source {i+1}: {chunk['text'][:300]}"
            for i, chunk in enumerate(context_chunks[:3])
        ])
        
        # Create prompt
        prompt = f"""Context information:
{context}

Question: {query}

Answer based on the context above:"""
        
        try:
            # Generate answer
            response = self.llm(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Extract only the answer part
            if "Answer based on the context above:" in generated_text:
                answer = generated_text.split("Answer based on the context above:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            # Clean up answer
            answer = answer.split('\n')[0].strip()
            
            if not answer or len(answer) < 10:
                answer = f"Based on the documents, {context_chunks[0]['text'][:200]}..."
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Based on the context: {context_chunks[0]['text'][:200]}..."
    
    def get_all_embeddings(self) -> np.ndarray:
        """Get all chunk embeddings as numpy array."""
        if not self.chunk_embeddings:
            return np.array([])
        return np.array(self.chunk_embeddings)
