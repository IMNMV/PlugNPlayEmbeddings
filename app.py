import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import networkx as nx
import ollama
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class PaperAnalyzer:
    def __init__(self):
        self.client = ollama.Client(host='http://localhost:11434')

    def chunk_text(self, text, chunk_size):
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) 
                for i in range(0, len(words), chunk_size)]

    def chunk_and_embed(self, text, chunk_size=8000):
        """Chunk long text and return averaged embedding"""
        # Split into chunks if needed
        if len(text.split()) > chunk_size:
            chunks = self.chunk_text(text, chunk_size)
        else:
            chunks = [text]
            
        # Get embeddings for all chunks
        chunk_embeddings = []
        for chunk in chunks:
            emb = self.get_embedding(chunk, task_type='clustering')
            chunk_embeddings.append(emb)
        
        # Return averaged embedding
        return np.mean(chunk_embeddings, axis=0)
        
    def parse_document(self, uploaded_file, chunk_size=None):
        """Parse any text document into chunks with metadata"""
        content = uploaded_file.getvalue().decode('utf-8')
        
        # Determine file type and name
        filename = uploaded_file.name
        file_type = filename.split('.')[-1]
        
        # Initialize parsing settings
        MAX_TOKENS = 8192
        WORDS_PER_TOKEN = 1.5
        MAX_WORDS = int(MAX_TOKENS / WORDS_PER_TOKEN)
        
        # Use provided chunk size or default
        if chunk_size:
            CHUNK_SIZE = min(chunk_size, MAX_WORDS)  # Ensure we don't exceed max
        else:
            CHUNK_SIZE = int(MAX_WORDS * 0.8)  # 80% of max to be safe
            
        OVERLAP = int(CHUNK_SIZE * 0.1)  # 10% overlap between chunks

        # Split content into words
        words = content.split()
        total_words = len(words)

        parsed_chunks = []
        for i in range(0, total_words, CHUNK_SIZE - OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = ' '.join(chunk_words)
            
            # Verify chunk isn't too long
            if len(chunk_text.split()) > MAX_WORDS:
                st.warning(f"Chunk {len(parsed_chunks) + 1} exceeds maximum token length and will be truncated")
                chunk_words = chunk_words[:MAX_WORDS]
                chunk_text = ' '.join(chunk_words)
            
            # Create chunk identifier
            chunk_num = len(parsed_chunks) + 1
            chunk_id = f"{filename}_chunk_{chunk_num}"
            
            # Calculate chunk position metadata
            start_pos = i
            end_pos = min(i + CHUNK_SIZE, total_words)
            
            parsed_chunks.append({
                'source': chunk_id,
                'content': chunk_text,
                'metadata': {
                    'filename': filename,
                    'file_type': file_type,
                    'chunk_number': chunk_num,
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'word_count': len(chunk_words)
                }
            })
        
        return parsed_chunks
    
    def extract_topics(self, documents, n_topics=5):
        """Extract topics from document chunks using embeddings"""
        from sklearn.decomposition import PCA
        
        # Get embeddings for all chunks
        embeddings = []
        for doc in documents:
            emb = self.chunk_and_embed(doc['content'])
            embeddings.append(emb)
        
        embeddings_array = np.array(embeddings)
        
        # Apply PCA to embeddings
        pca = PCA(n_components=n_topics, random_state=42)
        topic_embeddings = pca.fit_transform(embeddings_array)
        
        # Get top chunks for each topic
        topics = []
        for topic_idx in range(n_topics):
            # Get normalized topic scores (convert to 0-1 range)
            topic_scores = topic_embeddings[:, topic_idx]
            topic_scores = (topic_scores - topic_scores.min()) / (topic_scores.max() - topic_scores.min())
            
            # Get top chunks
            top_chunks_idx = topic_scores.argsort()[-5:][::-1]  # Get top 5 chunks
            topics.append({
                'id': topic_idx,
                'chunks': [documents[idx]['content'][:200] + "..." for idx in top_chunks_idx],
                'chunk_ids': [documents[idx]['source'] for idx in top_chunks_idx],
                'strengths': topic_scores[top_chunks_idx]
            })
        
        # Normalize topic distribution matrix
        topic_dist = (topic_embeddings - topic_embeddings.min()) / (topic_embeddings.max() - topic_embeddings.min())
        
        return topics, topic_dist
    
    def get_embedding(self, text, task_type='search_document'):
        """Get embedding with task-specific prefix"""
        try:
            prefixed_text = f"{task_type}: {text}"
            response = self.client.embeddings(
                model='nomic-embed-text',
                prompt=prefixed_text
            )
            
            # Convert response to numeric array
            if isinstance(response, dict):
                emb = response.get('embedding', response.get('embeddings', []))
            elif isinstance(response, list):
                emb = response[0]
            else:
                emb = response

            # Ensure we have a numeric array
            if hasattr(emb, 'embedding'):
                return np.array(emb.embedding)
            return np.array(emb)
                
        except Exception as e:
            st.error(f"Embedding error: {str(e)}")
            print(f"Response type: {type(response)}")
            print(f"Response content: {response}")
            raise

    def analyze_query(self, query_text, papers, similarity_threshold=0.5):
        """Search with query prefix"""
        query_embedding = self.get_embedding(query_text, task_type='search_query')
        
        related_papers = []
        for paper in papers:
            paper_embedding = self.chunk_and_embed(paper['content'])
            similarity = self.calculate_similarity(query_embedding, paper_embedding)
            
            if similarity > similarity_threshold:
                preview = ' '.join(paper['content'].split()[:50]) + "..."
                related_papers.append({
                    'source': paper['source'],
                    'similarity': similarity,
                    'content': paper['content'],
                    'metadata': paper.get('metadata', {}),
                    'cosine': similarity  
                })
        
        return sorted(related_papers, key=lambda x: x['similarity'], reverse=True)
        
    def cluster_papers(self, papers, n_clusters=5):
        """Cluster papers using averaged embeddings"""
        embeddings = []
        for paper in papers:
            paper_embedding = self.chunk_and_embed(paper['content'])
            embeddings.append(paper_embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(embeddings_array)
            
        return clusters, embeddings_array
            
        

    def compare_documents(self, doc1_chunks, doc2_chunks):
        """Compare two documents by analyzing their chunks in embedding space"""
        # Get embeddings for all chunks from both documents
        doc1_embeddings = []
        doc2_embeddings = []
        
        # Get embeddings for all chunks
        for chunk in doc1_chunks:
            emb = self.chunk_and_embed(chunk['content'])
            doc1_embeddings.append(emb)
        
        for chunk in doc2_chunks:
            emb = self.chunk_and_embed(chunk['content'])
            doc2_embeddings.append(emb)
        
        # Combine embeddings for TSNE
        all_embeddings = np.vstack([doc1_embeddings, doc2_embeddings])
        
        # Calculate appropriate perplexity (should be smaller than n_samples)
        n_samples = len(doc1_chunks) + len(doc2_chunks)
        perplexity = min(30, n_samples - 1)  # Default is 30, but must be < n_samples
        
        # Create TSNE visualization with adjusted perplexity
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # Split back into document groups
        doc1_coords = embeddings_2d[:len(doc1_embeddings)]
        doc2_coords = embeddings_2d[len(doc1_embeddings):]
        
        # Calculate cross-document similarities
        similarities = np.zeros((len(doc1_chunks), len(doc2_chunks)))
        for i, emb1 in enumerate(doc1_embeddings):
            for j, emb2 in enumerate(doc2_embeddings):
                similarities[i,j] = self.calculate_similarity(emb1, emb2)
        
        # Create visualization data
        viz_data = {
            'tsne_coords': embeddings_2d,
            'doc1_coords': doc1_coords,
            'doc2_coords': doc2_coords,
            'cross_similarities': similarities,
            'metadata': {
                'doc1_chunks': len(doc1_chunks),
                'doc2_chunks': len(doc2_chunks),
                'perplexity_used': perplexity
            }
        }
        
        return viz_data
    
    def visualize_comparison(self, viz_data, doc1_chunks, doc2_chunks):
        """Create visualizations for document comparison"""
        # TSNE scatter plot
        fig_scatter = go.Figure()
        
        # Add points for document 1
        doc1_coords = viz_data['doc1_coords']
        doc1_hover_text = [f"Doc1 Chunk {i+1}<br>{chunk['content'][:100]}..." 
                        for i, chunk in enumerate(doc1_chunks)]
        fig_scatter.add_trace(go.Scatter(
            x=doc1_coords[:, 0],
            y=doc1_coords[:, 1],
            mode='markers+text',
            name='Document 1',
            text=[f"D1-{i+1}" for i in range(len(doc1_coords))],
            textposition="top center",
            hovertext=doc1_hover_text,
            hoverinfo='text',
            marker=dict(size=10, color='blue', opacity=0.6)
        ))
        
        # Add points for document 2
        doc2_coords = viz_data['doc2_coords']
        doc2_hover_text = [f"Doc2 Chunk {i+1}<br>{chunk['content'][:100]}..." 
                        for i, chunk in enumerate(doc2_chunks)]
        fig_scatter.add_trace(go.Scatter(
            x=doc2_coords[:, 0],
            y=doc2_coords[:, 1],
            mode='markers+text',
            name='Document 2',
            text=[f"D2-{i+1}" for i in range(len(doc2_coords))],
            textposition="top center",
            hovertext=doc2_hover_text,
            hoverinfo='text',
            marker=dict(size=10, color='red', opacity=0.6)
        ))
        
        fig_scatter.update_layout(
            title=f"Document Chunks in Embedding Space (perplexity={viz_data['metadata']['perplexity_used']})",
            xaxis_title="TSNE Dimension 1",
            yaxis_title="TSNE Dimension 2",
            showlegend=True,
            hovermode='closest'
        )
        
        # Create cross-similarity heatmap
        fig_heatmap = px.imshow(
            viz_data['cross_similarities'],
            labels=dict(x="Document 2 Chunks", y="Document 1 Chunks", color="Similarity"),
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig_heatmap.update_layout(title="Cross-document Chunk Similarities")
        
        return fig_scatter, fig_heatmap


    def create_similarity_network(self, papers, similarity_threshold=0.7):
        """Create network using averaged embeddings"""
        try:
            embeddings = []
            progress_bar = st.progress(0)
            
            for i, paper in enumerate(papers):
                progress_bar.progress(i/len(papers))
                embedding = self.chunk_and_embed(paper['content'])
                embeddings.append(embedding)
            
            progress_bar.progress(1.0)
            
            # Calculate similarities including self-similarity
            n = len(papers)
            similarities = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    # Remove the if i != j check to include self-similarities
                    similarities[i,j] = self.calculate_similarity(
                        embeddings[i], embeddings[j]
                    )
            
            # Create network
            G = nx.Graph()
            
            for i, paper in enumerate(papers):
                G.add_node(i, title=paper['source'])
            
            for i in range(n):
                for j in range(i+1, n):  # Keep this check for network edges
                    if similarities[i,j] > similarity_threshold:
                        G.add_edge(i, j, weight=similarities[i,j])
                        
            return G, similarities, embeddings
            
        except Exception as e:
            st.error(f"Network error: {str(e)}")
            raise

    def calculate_similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def visualize_network(self, G, clusters=None):
        """Network visualization with optional clustering"""
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                color=clusters if clusters is not None else None
            ),
            text=[G.nodes[node]['title'] for node in G.nodes()],
            textposition="top center"
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title="Paper Network (colored by cluster)" if clusters is not None else "Paper Network",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40)
                    ))
        
        return fig

def main():
    st.title("Textual Analysis")
    analyzer = PaperAnalyzer()
    
    # Global settings in sidebar
    st.sidebar.header("Analysis Settings")
    chunk_size = st.sidebar.slider(
        "Chunk Size (words)", 
        min_value=100,
        max_value=5000,
        value=4000,
        step=100
    )
    
    n_topics = st.sidebar.slider(
        "Number of Topics",
        min_value=2,
        max_value=10,
        value=5
    )
    
    # Choose analysis type
    analysis_type = st.radio("Select Analysis Type:", 
                            ["Single Document Analysis", "Document Comparison"])
    
    if analysis_type == "Single Document Analysis":
        uploaded_file = st.file_uploader("Upload document", type=["txt", "md", "csv", "json"], key="single_doc")
        
        if uploaded_file is not None:
            documents = analyzer.parse_document(uploaded_file, chunk_size=chunk_size)
            
            # Search functionality
            st.subheader("Semantic Search")
            query = st.text_input("Enter research question or topic:")
            search_threshold = st.slider("Search Similarity Threshold", 0.0, 1.0, 0.3)
            
            if query:
                with st.spinner("Searching document..."):
                    matches = analyzer.analyze_query(query, documents, search_threshold)
                    if matches:
                        for match in matches:
                            with st.expander(f"{match['source']} (Similarity: {match['cosine']:.3f})"):
                                st.write(match['content'])
                                st.write("**Metadata:**")
                                st.json(match['metadata'])
                    else:
                        st.warning("No matches found. Try lowering the threshold.")
            
            # Topic Modeling
            st.subheader("Topic Analysis")
            if st.button("Extract Topics", key="extract_topics"):
                with st.spinner("Analyzing topics..."):
                    topics, topic_dist = analyzer.extract_topics(documents, n_topics=n_topics)
                    
                    # Display topics
                    for topic in topics:
                        with st.expander(f"Topic {topic['id'] + 1}"):
                            for chunk_idx, (chunk, chunk_id, strength) in enumerate(zip(
                                topic['chunks'], topic['chunk_ids'], topic['strengths'])):
                                st.write(f"**{chunk_id}** (Strength: {strength:.3f})")
                                st.write(chunk)
                                st.write("---")
                    
                    # Show topic distribution heatmap
                    st.write("### Topic Distribution Across Chunks")
                    topic_df = pd.DataFrame(
                        topic_dist,
                        index=[d['source'] for d in documents],
                        columns=[f"Topic {i+1}" for i in range(n_topics)]
                    )
                    fig_topics = px.imshow(
                        topic_df,
                        labels=dict(x="Topics", y="Chunks", color="Strength"),
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_topics)
            
            # Add visualizations
            st.subheader("Document Structure Visualizations")
            
            # Network visualization
            similarity_threshold = st.slider(
                "Network Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="single_threshold"
            )
            
            with st.spinner("Creating visualizations..."):
                # Network visualization
                G, similarities, _ = analyzer.create_similarity_network(
                    documents,
                    similarity_threshold
                )
                st.write("### Network Visualization")
                fig_network = analyzer.visualize_network(G)
                st.plotly_chart(fig_network)
                
                # TSNE and similarity matrices (only if multiple chunks)
                if len(documents) > 1:
                    viz_data = analyzer.compare_documents(documents[:len(documents)//2], documents[len(documents)//2:])
                    fig_scatter, fig_heatmap = analyzer.visualize_comparison(viz_data, 
                                                                          documents[:len(documents)//2], 
                                                                          documents[len(documents)//2:])
                    
                    st.write("### TSNE Visualization")
                    st.plotly_chart(fig_scatter)
                    
                    st.write("### Similarity Matrices")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Network Similarities")
                        df_sim = pd.DataFrame(
                            similarities,
                            index=[d['source'] for d in documents],
                            columns=[d['source'] for d in documents]
                        )
                        fig_net_heatmap = px.imshow(
                            df_sim,
                            labels=dict(x="Chunks", y="Chunks", color="Similarity"),
                            color_continuous_scale="RdBu"
                        )
                        st.plotly_chart(fig_net_heatmap)
                    
                    with col2:
                        st.write("TSNE-based Similarities")
                        st.plotly_chart(fig_heatmap)
                else:
                    st.warning("Need at least 2 chunks for TSNE visualization and similarity matrices")
                
    else:  # Document Comparison
        col1, col2 = st.columns(2)
        with col1:
            file1 = st.file_uploader("Upload first document", type=["txt", "md", "csv", "json"], key="doc1")
        with col2:
            file2 = st.file_uploader("Upload second document", type=["txt", "md", "csv", "json"], key="doc2")
            
        if file1 is not None and file2 is not None:
            doc1_chunks = analyzer.parse_document(file1, chunk_size=chunk_size)
            doc2_chunks = analyzer.parse_document(file2, chunk_size=chunk_size)
            
            # Topic Modeling for combined documents
            st.subheader("Combined Topic Analysis")
            if st.button("Extract Topics", key="extract_topics_combined"):
                with st.spinner("Analyzing topics..."):
                    combined_docs = doc1_chunks + doc2_chunks
                    topics, topic_dist = analyzer.extract_topics(combined_docs, n_topics=n_topics)
                    
                    # Display topics
                    for topic in topics:
                        with st.expander(f"Topic {topic['id'] + 1}"):
                            for chunk_idx, (chunk, chunk_id, strength) in enumerate(zip(
                                topic['chunks'], topic['chunk_ids'], topic['strengths'])):
                                st.write(f"**{chunk_id}** (Strength: {strength:.3f})")
                                st.write(chunk)
                                st.write("---")
                    
                    # Show topic distribution heatmap
                    st.write("### Topic Distribution Across All Chunks")
                    topic_df = pd.DataFrame(
                        topic_dist,
                        index=[d['source'] for d in combined_docs],
                        columns=[f"Topic {i+1}" for i in range(n_topics)]
                    )
                    fig_topics = px.imshow(
                        topic_df,
                        labels=dict(x="Topics", y="Chunks", color="Strength"),
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_topics)
            
            if st.button("Compare Documents", key="compare_btn"):
                with st.spinner("Analyzing documents..."):
                    # TSNE visualization
                    viz_data = analyzer.compare_documents(doc1_chunks, doc2_chunks)
                    fig_scatter, fig_heatmap = analyzer.visualize_comparison(viz_data, doc1_chunks, doc2_chunks)
                    
                    st.write("### TSNE Visualization")
                    st.plotly_chart(fig_scatter)
                    
                    # Network visualization
                    all_chunks = doc1_chunks + doc2_chunks
                    similarity_threshold = 0.7
                    G, similarities, _ = analyzer.create_similarity_network(
                        all_chunks,
                        similarity_threshold
                    )
                    
                    st.write("### Network Visualization")
                    fig_network = analyzer.visualize_network(G)
                    st.plotly_chart(fig_network)
                    
                    # Show both similarity matrices
                    st.write("### Similarity Matrices")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Cross-document Similarities")
                        st.plotly_chart(fig_heatmap)
                    
                    with col2:
                        st.write("Network Similarities")
                        df_sim = pd.DataFrame(
                            similarities,
                            index=[d['source'] for d in all_chunks],
                            columns=[d['source'] for d in all_chunks]
                        )
                        fig_net_heatmap = px.imshow(
                            df_sim,
                            labels=dict(x="Chunks", y="Chunks", color="Similarity"),
                            color_continuous_scale="RdBu"
                        )
                        st.plotly_chart(fig_net_heatmap)
                    
                    # Basic statistics and most similar chunks
                    st.write("### Document Statistics")
                    st.write(f"Document 1: {viz_data['metadata']['doc1_chunks']} chunks")
                    st.write(f"Document 2: {viz_data['metadata']['doc2_chunks']} chunks")
                    
                    similarities = viz_data['cross_similarities']
                    max_sim_idx = np.unravel_index(np.argmax(similarities), similarities.shape)
                    
                    st.write("### Most Similar Chunks")
                    with st.expander("View chunk contents"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Document 1 Chunk:")
                            st.write(doc1_chunks[max_sim_idx[0]]['content'])
                        with col2:
                            st.write("Document 2 Chunk:")
                            st.write(doc2_chunks[max_sim_idx[1]]['content'])

if __name__ == "__main__":
    main()
