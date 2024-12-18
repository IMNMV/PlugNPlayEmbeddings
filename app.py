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
        
    def parse_document(self, uploaded_file):
        """Parse any text document into chunks with metadata"""
        content = uploaded_file.getvalue().decode('utf-8')
        
        # Determine file type and name
        filename = uploaded_file.name
        file_type = filename.split('.')[-1]
        
        # Initialize parsing settings - nomic-embed-text has 8192 token context window
        # Conservative estimate: assume average token is 1.5 words. This may need adjusting
        MAX_TOKENS = 8192
        WORDS_PER_TOKEN = 1.5
        MAX_WORDS = int(MAX_TOKENS / WORDS_PER_TOKEN)  # ~5461 words
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
        
    def cluster_papers(self, papers, n_clusters=5, validate=True):
        """Cluster papers using averaged embeddings with validation metrics"""
        embeddings = []
        for paper in papers:
            paper_embedding = self.chunk_and_embed(paper['content'])
            embeddings.append(paper_embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # If validate, calculate elbow curve first
        if validate:
            inertias = []
            silhouette_scores = []
            # Ensure k_range is valid for silhouette score (2 to n_samples-1)
            k_range = range(2, min(len(papers)-1, 10) + 1)  
            
            # Only proceed if we have enough samples for at least 2 clusters
            if len(papers) > 2:
                for k in k_range:
                    kmeans = KMeans(n_clusters=k)
                    labels = kmeans.fit_predict(embeddings_array)
                    inertias.append(kmeans.inertia_)
                    try:
                        score = silhouette_score(embeddings_array, labels)
                        silhouette_scores.append(score)
                    except ValueError:
                        silhouette_scores.append(0)
            else:
                st.warning("Not enough documents for meaningful clustering validation")
                return kmeans.fit_predict(embeddings_array), embeddings_array, 0
                    
            # Store validation metrics
            self.clustering_metrics = {
                'k_range': list(k_range),
                'inertias': inertias,
                'silhouette_scores': silhouette_scores
            }
        
        # Perform final clustering with specified n_clusters
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(embeddings_array)
        
        # Calculate silhouette score for final clustering
        if len(set(clusters)) > 1:
            final_silhouette = silhouette_score(embeddings_array, clusters)
        else:
            final_silhouette = 0
            
        return clusters, embeddings_array, final_silhouette

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
    uploaded_file = st.file_uploader("Upload text document", type=["txt", "md", "csv", "json"])
    
    if uploaded_file is not None:
        documents = analyzer.parse_document(uploaded_file)
        
        # Search functionality
        st.subheader("Semantic Search")
        query = st.text_input("Enter research question or topic:")
        search_threshold = st.slider("Search Similarity Threshold", 0.0, 1.0, 0.3)
        
        if query:
            with st.spinner("Searching papers..."):
                matches = analyzer.analyze_query(query, documents, search_threshold)

                if matches:
                    st.write("### Matching Documents")
                    for match in matches:
                        with st.expander(f"{match['source']} (Cosine Similarity: {match['cosine']:.3f})"):
                            st.write(f"**Similarity Score:** {match['similarity']:.3f}")
                            
                            # Add buttons to toggle between preview and full content
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button('Show Preview', key=f"preview_{match['source']}"): 
                                    preview = ' '.join(match['content'].split()[:50]) + "..."
                                    st.write("**Preview:**")
                                    st.write(preview)
                            
                            with col2:
                                if st.button('Show Full Content', key=f"full_{match['source']}"):
                                    st.write("**Full Content:**")
                                    st.write(match['content'])
                                    
                            # Also display chunk metadata if available
                            if 'metadata' in match:
                                st.write("**Chunk Metadata:**")
                                st.json(match['metadata'])
                else:
                    st.warning("No matches found. Try lowering the threshold.")
        
        # Clustering
        st.subheader("Paper Clustering")
        n_clusters = st.slider("Number of clusters", 2, 10, 5)

        with st.spinner("Analyzing paper relationships..."):
            clusters, embeddings, silhouette = analyzer.cluster_papers(documents, n_clusters, validate=True)

            
            # Display validation metrics
            if hasattr(analyzer, 'clustering_metrics'):
                st.subheader("Clustering Validation")
                
                # Elbow curve
                fig_elbow = px.line(
                    x=analyzer.clustering_metrics['k_range'], 
                    y=analyzer.clustering_metrics['inertias'],
                    labels={'x': 'Number of Clusters', 'y': 'Inertia'},
                    title='Elbow Curve'
                )
                st.plotly_chart(fig_elbow)
                
                # Silhouette scores
                fig_silhouette = px.line(
                    x=analyzer.clustering_metrics['k_range'], 
                    y=analyzer.clustering_metrics['silhouette_scores'],
                    labels={'x': 'Number of Clusters', 'y': 'Silhouette Score'},
                    title='Silhouette Scores by Number of Clusters'
                )
                st.plotly_chart(fig_silhouette)
                
                # Current clustering silhouette score
                st.write(f"Current clustering (k={n_clusters}) silhouette score: {silhouette:.3f}")

            
            # Network visualization with clusters
            similarity_threshold = st.slider(
                "Network Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05
            )
            
            G, similarities, _ = analyzer.create_similarity_network(
                documents,
                similarity_threshold
            )
            
            st.subheader("Clustered Paper Network")
            fig = analyzer.visualize_network(G, clusters)
            st.plotly_chart(fig)
            
            # Similarity matrix heatmap
            st.subheader("Paper Similarity Matrix")
            df_sim = pd.DataFrame(
                similarities,
                index=[d['source'] for d in documents],
                columns=[d['source'] for d in documents]
            )
            fig_heatmap = px.imshow(
                df_sim,
                labels=dict(x="Paper", y="Paper", color="Similarity"),
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig_heatmap)
            
            # Show cluster assignments
            st.subheader("Cluster Assignments")
            cluster_df = pd.DataFrame({
            'Document': [d['source'] for d in documents],
            'Cluster': clusters
        })
            st.dataframe(cluster_df)

if __name__ == "__main__":
    main()