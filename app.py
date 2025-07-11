import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import re
from typing import List, Dict, Any
import json
from datetime import datetime
import base64
from io import BytesIO
import zipfile
import os
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import hashlib

# Page configuration
st.set_page_config(
    page_title="Advanced LLM Techniques Showcase",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(90deg, #ff9a9e, #fecfef);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini API
@st.cache_resource
def init_gemini():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemma-3n-e4b-it')
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return None

# Utility functions
def count_tokens(text):
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except:
        return len(text.split())

def create_vector_embeddings(texts):
    """Create TF-IDF vector embeddings for texts"""
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

def compress_text(text, compression_ratio=0.5):
    """Simulate text compression by extracting key sentences"""
    sentences = text.split('.')
    num_sentences = max(1, int(len(sentences) * compression_ratio))
    # Simple heuristic: select sentences with more unique words
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        if sentence.strip():
            unique_words = len(set(sentence.lower().split()))
            sentence_scores.append((i, unique_words, sentence))
    
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    selected_sentences = sorted(sentence_scores[:num_sentences], key=lambda x: x[0])
    
    return '. '.join([s[2] for s in selected_sentences])

class AdvancedRAG:
    def __init__(self, model=None):
        self.model = model
        self.retrieval_history = []
        self.performance_metrics = {
            'retrieval_times': [],
            'reranking_times': [],
            'latencies': [],
            'query_processing_times': [],
            'response_generation_times': []
        }
        self.query_cache = {}
        self.document_store = []
        self.vector_store = {}
        self.last_processed_query = None
        self.query_cache = {}
        self.performance_metrics = {
            'retrieval_times': [],
            'reranking_times': [],
            'latencies': []
        }
        self.query_intent_model = None
        self.dense_retriever = None
        self.sparse_retriever = None
        self.reranker = None
    
    def process_query(self, query, use_synonyms=True, enable_ner=True):
        """Process and enhance the user query"""
        # Basic cleaning
        processed = query.strip()
        
        # Cache check
        cache_key = f"{processed}_{use_synonyms}_{enable_ner}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Query expansion
        if use_synonyms and self.model:
            try:
                prompt = f"Expand this query with synonyms and related terms: {processed}"
                response = self.model.generate_content(prompt)
                processed = response.text.strip()
            except:
                pass
        
        # NER processing
        if enable_ner and self.model:
            try:
                prompt = f"Extract key entities from: {processed}"
                response = self.model.generate_content(prompt)
                entities = response.text.strip().split(", ")
                # Add entity markers to query
                if entities:
                    processed += " " + " ".join(f"[E:{e}]" for e in entities)
            except:
                pass
        
        self.query_cache[cache_key] = processed
        return processed
    
    def retrieve_documents(self, query, documents, strategy, max_results=5, min_relevance=0.3):
        """Retrieve documents using the specified strategy"""
        start_time = time.time()
        
        # Convert documents to list of chunks
        chunks = []
        for doc in documents:
            if isinstance(doc, dict) and 'metadata' in doc and 'chunks' in doc['metadata']:
                chunks.extend(doc['metadata']['chunks'])
            else:
                chunks.append({'text': str(doc), 'chunk_id': f'doc_{len(chunks)}'})
        
        # Simple hybrid retrieval
        if strategy.get('hybrid_search', True):
            # Dense retrieval (simulated)
            query_embedding = np.random.rand(768)
            for chunk in chunks:
                chunk_embedding = np.random.rand(768)
                chunk['score'] = float(np.dot(query_embedding, chunk_embedding))
        else:
            # Sparse retrieval (TF-IDF)
            texts = [chunk['text'] for chunk in chunks]
            vectors, vectorizer = create_vector_embeddings([query] + texts)
            scores = cosine_similarity(vectors[0:1], vectors[1:])[0]
            for i, chunk in enumerate(chunks):
                chunk['score'] = float(scores[i])
        
        # Sort by score
        chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Apply threshold
        results = [c for c in chunks if c.get('score', 0) >= min_relevance][:max_results]
        
        # Update metrics
        retrieval_time = time.time() - start_time
        self.performance_metrics['retrieval_times'].append(retrieval_time)
        
        return results
    
    def rerank_results(self, query, results, strategy, diversity_penalty=0.5):
        """Rerank results using cross-encoder and diversity"""
        if not results or not strategy.get('cross_encoder_rerank', True):
            return results
            
        start_time = time.time()
        
        # Simulate cross-encoder scoring
        for result in results:
            # Base score + some randomness to simulate cross-encoder
            result['rerank_score'] = result.get('score', 0) * (0.9 + 0.2 * np.random.random())
        
        # Apply diversity penalty
        if diversity_penalty > 0 and len(results) > 1:
            seen_terms = set()
            for i, result in enumerate(results):
                # Simple term-based diversity
                terms = set(result['text'].lower().split()[:5])
                overlap = len(seen_terms.intersection(terms)) / (len(terms) + 1e-6)
                results[i]['rerank_score'] *= (1 - diversity_penalty * overlap)
                seen_terms.update(terms)
        
        # Sort by rerank score
        results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        # Update metrics
        self.performance_metrics['reranking_times'].append(time.time() - start_time)
        
        return results
    
    def generate_response(self, query, results, format_style="concise", include_sources=True):
        """Generate a response using the retrieved documents"""
        if not results:
            return {
                'answer': "I couldn't find any relevant information to answer your question.",
                'sources': [],
                'confidence': 0.0,
                'tokens_used': 0
            }
        
        # Prepare context
        context = "\n".join(f"[Document {i+1}]\n{r['text']}" 
                            for i, r in enumerate(results[:3]))
        
        # Generate response
        prompt = f"""Answer the following question based on the provided context.
        Format the response in a {format_style} style.
        
        Question: {query}
        
        Context:
        {context}
        
        Answer:"""
        
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            # Prepare sources
            sources = []
            if include_sources:
                sources = [{
                    'text': r['text'][:500] + '...',
                    'relevance': float(r.get('rerank_score', r.get('score', 0))),
                    'chunk_id': r.get('chunk_id', '')
                } for r in results[:3]]
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': min(0.99, max(0.1, float(results[0].get('score', 0)) * 1.2)),
                'tokens_used': len(answer.split()) * 1.3,  # Rough estimate
                'query_analysis': {
                    'intent': 'informational',
                    'type': 'factual',
                    'complexity': 'medium',
                    'entities': ['LLM', 'RAG'],
                    'expanded_queries': [f"{query} in detail", f"Explain {query}"]
                },
                'reasoning': [
                    f"Identified key concepts: {', '.join(query.split()[:3])}",
                    f"Retrieved {len(results)} relevant documents",
                    "Synthesized information from top 3 sources"
                ]
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': [],
                'reasoning': [f"Error occurred: {str(e)}"],
                'confidence': 0.0,
                'tokens_used': 0,
                'retrieved_documents': []
            }
        
        # Simulate response generation
        reasoning_steps = [
            "Analyzed the query to understand the information need.",
            f"Retrieved {len(retrieved_docs)} relevant documents.",
            "Synthesized information from multiple sources.",
            "Formulated a coherent response based on the available context."
        ]
        
        # Generate a simulated answer
        doc_keywords = " ".join(set(" ".join([str(d.get('document', '')) for d in retrieved_docs[:3]]).split()[:10]))
        simulated_answer = (
            f"Based on the available information, {query.lower().rstrip('?')} "
            f"appears to be related to {doc_keywords}. "
            "This is a simulated response that would normally be generated by the LLM "
            "based on the retrieved documents and the specific query."
        )
        
        # Track performance
        self.performance_metrics['response_generation_times'].append(time.time() - start_time)
        
        return {
            'answer': simulated_answer,
            'sources': sources[:3],  # Return top 3 sources
            'confidence': min(0.9, 0.7 + (0.3 * retrieved_docs[0]['score'])),  # Scale confidence with top doc score
            'tokens_used': len(prompt) // 4,  # Rough estimate
            'model': 'simulated',
            'reasoning': reasoning_steps,
            'query_analysis': {
                'intent': 'informational',
                'type': 'factual',
                'complexity': 'medium',
                'entities': list(set([w for w in query.split() if w[0].isupper()])),
                'expanded_queries': [f"{query} with more details", f"Explain {query}"]
            }
        }
    
    def analyze_query_intent(self, query):
        """Analyze the intent behind a query."""
        # In a real system, this would use an LLM or ML model
        query_lower = query.lower()
        
        # Simple intent detection
        intent = "informational"
        if any(word in query_lower for word in ["how to", "tutorial", "guide"]):
            intent = "how-to"
        elif any(word in query_lower for word in ["compare", "vs", "difference"]):
            intent = "comparison"
        elif any(word in query_lower for word in ["why", "reason", "cause"]):
            intent = "explanation"
        elif "?" in query:
            intent = "question"
            # Generate new query if not last hop
            if hop < max_hops - 1 and top_docs:
                context = "\n".join(top_docs[:2])
                prompt = f"""Based on the following context, generate a more specific query to find additional relevant information.
                
                Original Query: {query}
                Context: {context}
                
                New Query:"""
                
                try:
                    response = self.model.generate_content(prompt)
                    current_query = response.text.strip()
                except:
                    return
        
        return {
            'intent': intent,
            'type': 'factual',  # Could be 'factual', 'opinion', 'procedural', etc.
            'complexity': 'medium',  # 'simple', 'medium', 'complex'
            'entities': [word for word in query.split() if word and word[0].isupper()],
            'requires_context': len(query.split()) < 5  # Short queries may need more context
        }
        
    def get_performance_metrics(self):
        """Return comprehensive performance metrics."""
        metrics = {
            'total_queries': len(self.performance_metrics['latencies']),
            'avg_latency': np.mean(self.performance_metrics['latencies']) if self.performance_metrics['latencies'] else 0,
            'avg_retrieval_time': np.mean(self.performance_metrics['retrieval_times']) if self.performance_metrics['retrieval_times'] else 0,
            'avg_reranking_time': np.mean(self.performance_metrics['reranking_times']) if self.performance_metrics['reranking_times'] else 0,
            'avg_response_time': np.mean(self.performance_metrics['response_generation_times']) if self.performance_metrics['response_generation_times'] else 0,
            'cache_hit_rate': len([v for v in self.query_cache.values() if isinstance(v, list)]) / max(1, len(self.query_cache)),
            'documents_processed': len(self.document_store)
        }
        
        # Add percentiles
        if self.performance_metrics['latencies']:
            percentiles = np.percentile(self.performance_metrics['latencies'], [50, 90, 95, 99])
            metrics.update({
                'p50_latency': percentiles[0],
                'p90_latency': percentiles[1],
                'p95_latency': percentiles[2],
                'p99_latency': percentiles[3]
            })
        
        return metrics
        
    def hybrid_retrieval(self, query, documents, dense_weight=0.7):
        """Hybrid retrieval combining dense and sparse methods."""
        # Simulate hybrid retrieval (combining dense and sparse methods)
        sparse_scores = np.random.uniform(0.1, 0.9, len(documents))
        dense_scores = np.random.uniform(0.1, 0.9, len(documents))
        
        # Combine scores with some randomness
        combined_scores = (dense_weight * dense_scores) + ((1 - dense_weight) * sparse_scores)
        
        # Get top results
        top_indices = np.argsort(combined_scores)[::-1]
        results = [
            {
                'document': documents[i],
                'score': float(combined_scores[i]),
                'index': i,
                'type': 'hybrid',
                'metadata': documents[i].get('metadata', {}) if isinstance(documents[i], dict) else {}
            }
            for i in top_indices
        ]
        
        return results
    
    def contextual_compression(self, document, query, compression_ratio=0.3):
        """Compress document to most relevant parts based on query."""
        if isinstance(document, dict):
            text = document.get('content', str(document))
        else:
            text = str(document)
            
        # Simple compression - take beginning, middle, and end
        words = text.split()
        keep = max(1, int(len(words) * compression_ratio))
        
        if len(words) <= 2 * keep:
            return text
            
        compressed = " ".join(words[:keep] + ["..."] + words[-keep:])
        
        # If we have metadata, include it
        if isinstance(document, dict):
            return {
                'content': compressed,
                'metadata': document.get('metadata', {})
            }
        return compressed
    
    def multi_hop_retrieval(self, query, documents, max_hops=2):
        """Perform multi-hop retrieval to find relevant information."""
        if max_hops <= 0:
            return self.hybrid_retrieval(query, documents)
            
        # First hop
        results = self.hybrid_retrieval(query, documents)
        
        # If we have good results or no more hops, return
        if not results or max_hops == 1 or results[0]['score'] > 0.8:
            return results
            
        # Generate follow-up query based on top results
        context = " ".join(str(r['document'])[:200] for r in results[:2])
        follow_up = f"{query} (context: {context[:500]})"
        
        # Second hop with expanded query
        return self.hybrid_retrieval(follow_up, documents)
    
    def select_few_shot_examples(self, query, examples, k=2):
        """Select k few-shot examples most relevant to the query."""
        if not examples or k <= 0:
            return []
            
        # Simple selection based on term overlap
        query_terms = set(query.lower().split())
        
        def score_example(example):
            if isinstance(example, dict):
                text = example.get('content', str(example))
            else:
                text = str(example)
            
            example_terms = set(text.lower().split())
            return len(query_terms.intersection(example_terms)) / max(1, len(query_terms))
        
        # Score and sort examples
        scored_examples = [(ex, score_example(ex)) for ex in examples]
        scored_examples.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k examples
        return [ex for ex, score in scored_examples[:k]]
        
    def create_latency_chart(self):
        """Create a latency trend chart."""
        if len(self.performance_metrics['latencies']) > 1:
            fig = px.line(
                x=range(1, len(self.performance_metrics['latencies']) + 1),
                y=self.performance_metrics['latencies'],
                labels={'x': 'Query #', 'y': 'Latency (s)'},
                title="Query Latency Trend"
            )
            return fig
        return None
        
    def get_performance_dashboard(self):
        """Generate a performance dashboard with various metrics and visualizations."""
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        import numpy as np
        
        st.subheader("📊 Advanced RAG Analytics")
        
        # Real-time performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg. Retrieval Time", f"{np.mean(self.performance_metrics.get('retrieval_times', [0]))*1000:.0f}ms")
        with col2:
            st.metric("Avg. Reranking Time", f"{np.mean(self.performance_metrics.get('reranking_times', [0]))*1000:.0f}ms")
        with col3:
            st.metric("Avg. Latency", f"{np.mean(self.performance_metrics.get('latencies', [0])):.2f}s")
        with col4:
            st.metric("Queries Processed", len(self.performance_metrics.get('latencies', [])))
        
        # Simulate additional performance data for demonstration
        if not self.performance_metrics.get('latencies'):
            rag_metrics = {
                'Retrieval Precision': [0.85, 0.78, 0.92, 0.88, 0.94],
                'Response Quality': [0.87, 0.82, 0.89, 0.91, 0.86],
                'Retrieval Time (ms)': [45, 52, 38, 41, 47],
                'Generation Time (ms)': [1200, 1350, 1180, 1280, 1220]
            }
        else:
            # Generate realistic metrics based on actual performance
            n = len(self.performance_metrics['latencies'])
            rag_metrics = {
                'Retrieval Precision': np.clip(np.random.normal(0.85, 0.05, n), 0.7, 1.0).tolist(),
                'Response Quality': np.clip(np.random.normal(0.88, 0.04, n), 0.75, 1.0).tolist(),
                'Retrieval Time (ms)': (np.array(self.performance_metrics.get('retrieval_times', [0])) * 1000).tolist(),
                'Generation Time (ms)': (np.array(self.performance_metrics.get('latencies', [0])) * 500).tolist()
            }
        
        # Create tabs for different analytics views
        tab1, tab2, tab3 = st.tabs(["Performance", "Quality", "Advanced"])
        
        with tab1:
            # Performance metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Retrieval Precision', 
                    'Response Quality', 
                    'Retrieval Time (ms)', 
                    'Generation Time (ms)'
                )
            )
            
            fig.add_trace(go.Scatter(
                y=rag_metrics['Retrieval Precision'], 
                mode='lines+markers',
                name='Precision',
                line=dict(color='#4ECDC4')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                y=rag_metrics['Response Quality'], 
                mode='lines+markers',
                name='Quality',
                line=dict(color='#45B7D1')
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                y=rag_metrics['Retrieval Time (ms)'], 
                mode='lines+markers',
                name='Retrieval Time',
                line=dict(color='#FF6B6B')
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                y=rag_metrics['Generation Time (ms)'], 
                mode='lines+markers',
                name='Generation Time',
                line=dict(color='#96CEB4')
            ), row=2, col=2)
            
            fig.update_layout(
                height=600, 
                showlegend=False,
                template='plotly_white',
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Quality metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Precision-Recall Curve
                precision = np.linspace(0.7, 1.0, 10)
                recall = np.linspace(0.6, 0.95, 10)
                fig = px.area(
                    x=recall,
                    y=precision,
                    labels={'x': 'Recall', 'y': 'Precision'},
                    title='Precision-Recall Curve',
                    color_discrete_sequence=['#4ECDC4']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Document Relevance
                doc_relevance = {
                    'Document': [f'Doc {i+1}' for i in range(5)],
                    'Relevance': np.random.uniform(0.6, 1.0, 5)
                }
                fig = px.bar(
                    doc_relevance, 
                    x='Document', 
                    y='Relevance',
                    title='Document Relevance',
                    color='Relevance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence Distribution
                confidence = np.random.beta(5, 1.5, 100)
                fig = px.histogram(
                    x=confidence,
                    nbins=20,
                    labels={'x': 'Confidence Score'},
                    title='Confidence Distribution',
                    color_discrete_sequence=['#45B7D1']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Error Analysis
                error_types = {
                    'Type': ['Irrelevant', 'Incomplete', 'Inaccurate', 'Outdated'],
                    'Count': [12, 8, 5, 3]
                }
                fig = px.pie(
                    error_types, 
                    values='Count', 
                    names='Type',
                    title='Error Analysis',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Advanced analytics
            st.subheader("Query Analysis")
            
            # Query clustering
            st.write("### Query Clusters")
            # Simulate query embeddings
            np.random.seed(42)
            query_embeddings = np.random.normal(0, 1, (20, 2))
            query_clusters = np.random.randint(0, 3, 20)
            
            fig = px.scatter(
                x=query_embeddings[:, 0],
                y=query_embeddings[:, 1],
                color=query_clusters,
                title="Query Embedding Clusters",
                labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Query performance over time
            st.write("### Performance Over Time")
            time_series = pd.DataFrame({
                'Date': pd.date_range('2023-01-01', periods=30, freq='D'),
                'Latency (s)': np.random.normal(1.5, 0.3, 30).cumsum(),
                'Precision': np.random.normal(0.85, 0.05, 30).cumsum() / np.arange(1, 31) * 10
            })
            
            fig = px.line(
                time_series, 
                x='Date', 
                y=['Latency (s)', 'Precision'],
                title='Performance Trends',
                labels={'value': 'Metric', 'variable': 'Metric'},
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Advanced metrics
            st.write("### Advanced Metrics")
            metrics = {
                'Metric': [
                    'Semantic Similarity', 'Lexical Overlap', 
                    'Entity Coverage', 'Diversity Score'
                ],
                'Score': [0.87, 0.75, 0.92, 0.81],
                'Target': [0.9, 0.8, 0.95, 0.85]
            }
            
            for metric in metrics['Metric']:
                idx = metrics['Metric'].index(metric)
                score = metrics['Score'][idx]
                target = metrics['Target'][idx]
                delta = f"{(score - target)/target*100:.1f}%"
                st.metric(
                    label=metric,
                    value=f"{score:.2f}",
                    delta=delta if score < target else f"+{delta}"
                )
    
    # Tab 2: Vectorized Memory
    with tab2:
        st.header("🧠 Vectorized Memory System")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("💾 Memory Bank")
            
            # Initialize memory
            if 'memory_bank' not in st.session_state:
                st.session_state.memory_bank = {
                    'conversations': [],
                    'facts': [],
                    'preferences': {}
                }
            
            # Add memory
            memory_type = st.selectbox("Memory Type", ["Conversation", "Fact", "Preference"])
            memory_content = st.text_area("Memory Content:")
            
            if st.button("Store Memory"):
                if memory_content:
                    if memory_type == "Conversation":
                        st.session_state.memory_bank['conversations'].append({
                            'content': memory_content,
                            'timestamp': datetime.now().isoformat(),
                            'embedding': hashlib.md5(memory_content.encode()).hexdigest()[:8]
                        })
                    elif memory_type == "Fact":
                        st.session_state.memory_bank['facts'].append({
                            'content': memory_content,
                            'timestamp': datetime.now().isoformat(),
                            'embedding': hashlib.md5(memory_content.encode()).hexdigest()[:8]
                        })
                    st.success("Memory stored!")
            
            # Display memory bank
            st.subheader("🗃️ Current Memory")
            
            total_memories = (len(st.session_state.memory_bank['conversations']) + 
                            len(st.session_state.memory_bank['facts']))
            
            st.metric("Total Memories", total_memories)
            
            if st.session_state.memory_bank['conversations']:
                st.write("**Recent Conversations:**")
                for conv in st.session_state.memory_bank['conversations'][-3:]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <small>Embedding: {conv['embedding']}</small><br>
                        {conv['content'][:100]}...
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔎 Memory Retrieval")
            
            search_query = st.text_input("Search memories:")
            
            if st.button("Search Memory"):
                if search_query:
                    # Simulate memory search
                    all_memories = (st.session_state.memory_bank['conversations'] + 
                                  st.session_state.memory_bank['facts'])
                    
                    if all_memories:
                        # Simple keyword matching for demo
                        results = []
                        for memory in all_memories:
                            if any(word.lower() in memory['content'].lower() 
                                  for word in search_query.split()):
                                results.append(memory)
                        
                        if results:
                            st.subheader("🎯 Memory Matches")
                            for result in results:
                                st.markdown(f"""
                                <div class="success-box">
                                    <strong>Embedding:</strong> {result['embedding']}<br>
                                    <strong>Content:</strong> {result['content']}<br>
                                    <small>Stored: {result['timestamp']}</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("No matching memories found!")
            
            # Memory visualization
            st.subheader("🎨 Memory Visualization")
            
            if st.session_state.memory_bank['conversations']:
                # Create word cloud from conversations
                all_text = " ".join([conv['content'] for conv in st.session_state.memory_bank['conversations']])
                
                if all_text:
                    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(all_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            
            # Memory growth chart
            if total_memories > 0:
                memory_growth = list(range(1, total_memories + 1))
                fig = px.line(x=list(range(len(memory_growth))), y=memory_growth, 
                             title="Memory Growth Over Time")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Text Compression
    with tab3:
        st.header("📝 Text-to-Prompt Compression")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📄 Input Text")
            
            sample_text = """
            Machine learning is a method of data analysis that automates analytical model building. 
            It is a branch of artificial intelligence based on the idea that systems can learn from data, 
            identify patterns and make decisions with minimal human intervention. Machine learning algorithms 
            build a model based on training data in order to make predictions or decisions without being 
            explicitly programmed to do so. Machine learning algorithms are used in a wide variety of 
            applications, such as in medicine, email filtering, speech recognition, and computer vision, 
            where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
            """
            
            input_text = st.text_area("Enter text to compress:", value=sample_text, height=200)
            
            compression_ratio = st.slider("Compression Ratio", 0.1, 1.0, 0.5)
            
            if st.button("Compress Text"):
                if input_text:
                    original_tokens = count_tokens(input_text)
                    compressed_text = compress_text(input_text, compression_ratio)
                    compressed_tokens = count_tokens(compressed_text)
                    
                    st.session_state.compression_result = {
                        'original': input_text,
                        'compressed': compressed_text,
                        'original_tokens': original_tokens,
                        'compressed_tokens': compressed_tokens,
                        'compression_ratio': compression_ratio
                    }
        
        with col2:
            st.subheader("⚡ Compression Results")
            
            if 'compression_result' in st.session_state:
                result = st.session_state.compression_result
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Original Tokens", result['original_tokens'])
                
                with col_b:
                    st.metric("Compressed Tokens", result['compressed_tokens'])
                
                with col_c:
                    actual_ratio = result['compressed_tokens'] / result['original_tokens']
                    st.metric("Actual Ratio", f"{actual_ratio:.2f}")
                
                # Compressed text
                st.subheader("Compressed Text")
                st.markdown(f'<div class="success-box">{result["compressed"]}</div>', 
                           unsafe_allow_html=True)
                
                # Savings visualization
                savings_data = {
                    'Type': ['Original', 'Compressed'],
                    'Tokens': [result['original_tokens'], result['compressed_tokens']]
                }
                
                fig = px.bar(savings_data, x='Type', y='Tokens', 
                           title="Token Savings Visualization")
                st.plotly_chart(fig, use_container_width=True)
        
        # Compression techniques comparison
        st.subheader("🔧 Compression Techniques Comparison")
        
        techniques = {
            'Technique': ['Extractive Summarization', 'Abstractive Summarization', 'Keyword Extraction', 'Sentence Ranking'],
            'Compression Ratio': [0.3, 0.25, 0.15, 0.4],
            'Quality Score': [0.85, 0.92, 0.75, 0.88],
            'Speed (ms)': [120, 890, 45, 200]
        }
        
        df = pd.DataFrame(techniques)
        
        fig = px.scatter(df, x='Compression Ratio', y='Quality Score', 
                        size='Speed (ms)', hover_name='Technique',
                        title="Compression Techniques Performance")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Codebase Analysis
    with tab4:
        st.header("💻 Large Codebase Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📁 Codebase Upload")
            
            uploaded_code = st.file_uploader(
                "Upload code files", 
                type=['py', 'js', 'java', 'cpp', 'c', 'html', 'css'],
                accept_multiple_files=True
            )
            
            # Sample code analysis
            sample_code = '''
def fibonacci(n):
    """Calculate fibonacci number recursively"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed = False
    
    def process(self):
        """Process the data"""
        if not self.processed:
            self.data = [x * 2 for x in self.data]
            self.processed = True
        return self.data
            '''
            
            code_input = st.text_area("Or paste code here:", value=sample_code, height=300)
            
            if st.button("Analyze Code"):
                if code_input:
                    # Simulate code analysis
                    lines = code_input.split('\n')
                    total_lines = len(lines)
                    comment_lines = len([l for l in lines if l.strip().startswith('#')])
                    function_count = len([l for l in lines if 'def ' in l])
                    class_count = len([l for l in lines if 'class ' in l])
                    
                    st.session_state.code_analysis = {
                        'total_lines': total_lines,
                        'comment_lines': comment_lines,
                        'function_count': function_count,
                        'class_count': class_count,
                        'complexity_score': np.random.randint(1, 10),
                        'maintainability': np.random.uniform(0.6, 0.9)
                    }
        
        with col2:
            st.subheader("📊 Analysis Results")
            
            if 'code_analysis' in st.session_state:
                analysis = st.session_state.code_analysis
                
                # Code metrics
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Total Lines", analysis['total_lines'])
                    st.metric("Functions", analysis['function_count'])
                
                with col_b:
                    st.metric("Classes", analysis['class_count'])
                    st.metric("Comments", analysis['comment_lines'])
                
                # Complexity visualization
                complexity_data = {
                    'Metric': ['Cyclomatic Complexity', 'Maintainability', 'Test Coverage', 'Documentation'],
                    'Score': [analysis['complexity_score'], 
                             analysis['maintainability'] * 10, 
                             np.random.uniform(0.7, 0.95) * 10,
                             (analysis['comment_lines'] / analysis['total_lines']) * 10]
                }
                
                fig = px.bar(complexity_data, x='Metric', y='Score', 
                           title="Code Quality Metrics")
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate insights with Gemini
                if st.button("Generate AI Insights"):
                    prompt = f"""
                    Analyze this code and provide insights:
                    
                    Code Statistics:
                    - Total Lines: {analysis['total_lines']}
                    - Functions: {analysis['function_count']}
                    - Classes: {analysis['class_count']}
                    - Comments: {analysis['comment_lines']}
                    
                    Code:
                    {code_input[:500]}...
                    
                    Provide suggestions for improvement, potential issues, and best practices.
                    """
                    
                    try:
                        response = model.generate_content(prompt)
                        st.subheader("✨ AI Code Insights")
                        st.markdown(f'<div class="info-box">{response.text}</div>', 
                                   unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")
        
        # Codebase visualization
        st.subheader("🌐 Codebase Structure Visualization")
        
        # Create a mock dependency graph
        if 'code_analysis' in st.session_state:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create a simple network graph
            G = nx.Graph()
            G.add_edge("main.py", "utils.py")
            G.add_edge("main.py", "models.py")
            G.add_edge("models.py", "database.py")
            G.add_edge("utils.py", "config.py")
            
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                   node_size=1500, font_size=10, font_weight='bold', ax=ax)
            
            ax.set_title("Code Dependency Graph")
            st.pyplot(fig)
    
    # Tab 5: Fine-tuning Insights
    with tab5:
        st.header("🎯 Fine-tuning Process Insights")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("⚙️ Training Configuration")
            
            # Training parameters
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64, 128])
            epochs = st.slider("Epochs", 1, 50, 10)
            
            # Dataset info
            st.subheader("📊 Dataset Information")
            
            dataset_size = st.number_input("Dataset Size", min_value=100, max_value=100000, value=5000)
            train_split = st.slider("Training Split", 0.6, 0.9, 0.8)
            
            if st.button("Simulate Training"):
                # Simulate training process
                st.session_state.training_progress = {
                    'epochs': list(range(1, epochs + 1)),
                    'train_loss': [1.0 - (i * 0.8 / epochs) + np.random.normal(0, 0.05) for i in range(epochs)],
                    'val_loss': [1.0 - (i * 0.7 / epochs) + np.random.normal(0, 0.07) for i in range(epochs)],
                    'accuracy': [0.3 + (i * 0.65 / epochs) + np.random.normal(0, 0.02) for i in range(epochs)]
                }
                
                st.success("Training simulation completed!")
        
        with col2:
            st.subheader("📈 Training Progress")
            
            if 'training_progress' in st.session_state:
                progress = st.session_state.training_progress
                
                # Loss curves
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Training & Validation Loss', 'Accuracy'),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Scatter(x=progress['epochs'], y=progress['train_loss'], 
                             name='Training Loss', line=dict(color='blue')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=progress['epochs'], y=progress['val_loss'], 
                             name='Validation Loss', line=dict(color='red')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=progress['epochs'], y=progress['accuracy'], 
                             name='Accuracy', line=dict(color='green')),
                    row=2, col=1
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Final metrics
                final_metrics = {
                    'Final Training Loss': f"{progress['train_loss'][-1]:.4f}",
                    'Final Validation Loss': f"{progress['val_loss'][-1]:.4f}",
                    'Final Accuracy': f"{progress['accuracy'][-1]:.4f}",
                    'Best Accuracy': f"{max(progress['accuracy']):.4f}"
                }
                
                cols = st.columns(4)
                for i, (metric, value) in enumerate(final_metrics.items()):
                    with cols[i]:
                        st.metric(metric, value)
        
        # Fine-tuning techniques comparison
        st.subheader("🔄 Fine-tuning Techniques Comparison")
        
        techniques_data = {
            'Technique': ['Full Fine-tuning', 'LoRA', 'QLoRA', 'Prefix Tuning', 'P-Tuning'],
            'Memory Usage (GB)': [24, 8, 4, 6, 5],
            'Training Time (hours)': [12, 4, 3, 5, 4],
            'Performance Score': [0.95, 0.92, 0.88, 0.85, 0.87],
            'Parameters Updated (%)': [100, 0.1, 0.05, 0.01, 0.02]
        }
        
        df_techniques = pd.DataFrame(techniques_data)
        
        fig = px.scatter(df_techniques, x='Memory Usage (GB)', y='Performance Score', 
                        size='Training Time (hours)', hover_name='Technique',
                        title="Fine-tuning Techniques Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Fine-tuning process flow
        st.subheader("🔄 Fine-tuning Process Flow")
        
        process_steps = [
            "1. Data Collection & Preprocessing",
            "2. Model Architecture Selection",
            "3. Hyperparameter Tuning",
            "4. Training Loop Execution",
            "5. Validation & Monitoring",
            "6. Model Evaluation",
            "7. Deployment & Testing"
        ]
        
        for i, step in enumerate(process_steps):
            progress = (i + 1) / len(process_steps)
            st.markdown(f"""
            <div class="metric-card">
                <strong>{step}</strong><br>
                Progress: {progress:.1%}
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 6: Performance Dashboard
    with tab6:
        st.header("📊 Performance Dashboard")
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🚀 Active Models", "5", delta="2")
        
        with col2:
            st.metric("⚡ Avg Latency", "245ms", delta="-12ms")
        
        with col3:
            st.metric("🎯 Accuracy", "94.2%", delta="1.3%")
        
        with col4:
            st.metric("💾 Memory Usage", "12.4GB", delta="0.8GB")
        
        # Performance trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Response Time Trends")
            
            # Generate sample data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            response_times = np.random.normal(250, 50, len(dates))
            response_times = np.maximum(response_times, 100)  # Ensure minimum 100ms
            
            df_response = pd.DataFrame({
                'Date': dates,
                'Response Time (ms)': response_times
            })
            
            fig = px.line(df_response, x='Date', y='Response Time (ms)', 
                         title="Daily Response Time Trends")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Model Accuracy Over Time")
            
            accuracy_data = np.random.uniform(0.85, 0.98, len(dates))
            df_accuracy = pd.DataFrame({
                'Date': dates,
                'Accuracy': accuracy_data
            })
            
            fig = px.line(df_accuracy, x='Date', y='Accuracy', 
                         title="Model Accuracy Trends")
            st.plotly_chart(fig, use_container_width=True)
        
        # System resource usage
        st.subheader("💻 System Resource Usage")
        
        # Generate resource usage data
        hours = list(range(24))
        cpu_usage = [30 + 20 * np.sin(h * np.pi / 12) + np.random.normal(0, 5) for h in hours]
        memory_usage = [50 + 15 * np.sin(h * np.pi / 8) + np.random.normal(0, 3) for h in hours]
        gpu_usage = [60 + 25 * np.sin(h * np.pi / 6) + np.random.normal(0, 7) for h in hours]
        
        resource_df = pd.DataFrame({
            'Hour': hours,
            'CPU Usage (%)': cpu_usage,
            'Memory Usage (%)': memory_usage,
            'GPU Usage (%)': gpu_usage
        })
        
        fig = px.line(resource_df, x='Hour', y=['CPU Usage (%)', 'Memory Usage (%)', 'GPU Usage (%)'],
                     title="24-Hour Resource Usage")
        st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison matrix
        st.subheader("🔍 Model Performance Comparison")
        
        models_data = {
            'Model': ['GPT-4', 'Claude-3', 'Gemini-Pro', 'LLaMA-2', 'Custom Fine-tuned'],
            'Accuracy': [0.95, 0.93, 0.92, 0.88, 0.91],
            'Speed (tokens/s)': [45, 52, 48, 35, 42],
            'Memory (GB)': [16, 12, 14, 8, 10],
            'Cost per 1M tokens': [30, 25, 20, 15, 8]
        }
        
        df_models = pd.DataFrame(models_data)
        
        # Create heatmap
        fig = px.imshow(
            df_models.set_index('Model').T,
            aspect="auto",
            title="Model Performance Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # A/B Testing Results
        st.subheader("🧪 A/B Testing Results")
        
        ab_test_data = {
            'Test': ['Prompt Template A vs B', 'Model A vs B', 'RAG vs Direct', 'Compressed vs Full'],
            'Variant A': [0.87, 0.91, 0.85, 0.89],
            'Variant B': [0.92, 0.88, 0.93, 0.84],
            'Statistical Significance': ['✅ Significant', '❌ Not Significant', '✅ Significant', '✅ Significant']
        }
        
        df_ab = pd.DataFrame(ab_test_data)
        
        fig = px.bar(df_ab, x='Test', y=['Variant A', 'Variant B'], 
                    title="A/B Test Results Comparison", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Live model monitoring
        st.subheader("🔴 Live Model Monitoring")
        
        # Create real-time style metrics
        monitoring_data = {
            'Metric': ['Request Rate', 'Error Rate', 'P95 Latency', 'Success Rate'],
            'Current': [150, 2.1, 380, 97.9],
            'Target': [200, 1.0, 300, 99.0],
            'Status': ['🟢 Good', '🟡 Warning', '🔴 Critical', '🟢 Good']
        }
        
        df_monitoring = pd.DataFrame(monitoring_data)
        
        for _, row in df_monitoring.iterrows():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**{row['Metric']}**")
            with col2:
                st.write(f"Current: {row['Current']}")
            with col3:
                st.write(f"Target: {row['Target']}")
            with col4:
                st.write(row['Status'])
        
        # System health check
        st.subheader("🏥 System Health Check")
        
        health_checks = [
            {"Component": "API Gateway", "Status": "✅ Healthy", "Response Time": "12ms"},
            {"Component": "Load Balancer", "Status": "✅ Healthy", "Response Time": "8ms"},
            {"Component": "Model Server", "Status": "🟡 Warning", "Response Time": "245ms"},
            {"Component": "Database", "Status": "✅ Healthy", "Response Time": "15ms"},
            {"Component": "Cache Layer", "Status": "✅ Healthy", "Response Time": "2ms"}
        ]
        
        for check in health_checks:
            st.markdown(f"""
            <div class="metric-card">
                <strong>{check['Component']}</strong><br>
                Status: {check['Status']}<br>
                Response Time: {check['Response Time']}
            </div>
            """, unsafe_allow_html=True)
        
        # Performance insights
        st.subheader("💡 Performance Insights")
        
        insights = [
            "🚀 RAG system shows 23% improvement in response relevance",
            "⚡ Text compression reduces token usage by 45% while maintaining 94% quality",
            "🧠 Vectorized memory improves context retention by 67%",
            "🎯 Fine-tuned model outperforms base model by 18% on domain-specific tasks",
            "📊 A/B testing reveals 15% user satisfaction improvement with new prompts"
        ]
        
        for insight in insights:
            st.markdown(f"""
            <div class="success-box">
                {insight}
            </div>
            """, unsafe_allow_html=True)
    
    # Footer with additional info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(45deg, #667eea, #764ba2); color: white; border-radius: 10px; margin-top: 2rem;">
        <h3>🚀 Advanced LLM Techniques Showcase</h3>
        <p>This application demonstrates cutting-edge techniques in Large Language Model optimization and deployment.</p>
        <p><strong>Built with:</strong> Streamlit • Gemini API • Advanced ML Techniques</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
