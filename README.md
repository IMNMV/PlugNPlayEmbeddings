# PlugNPlayEmbeddings

Text Analysis with Embeddings - This application provides semantic search and clustering analysis for text documents using Nomic embeddings and Ollama.

## Prerequisites
* Python 3.9+
* Ollama installed (https://ollama.com)
* Nomic-embed-text model pulled in Ollama: https://ollama.com/library/nomic-embed-text

## Setup Instructions

1. Install Ollama:
   * Visit https://ollama.com
   * Follow installation instructions for your OS

2. Pull the Nomic embeddings model using the command line interface:
   ```bash
   ollama pull nomic-embed-text

   # Confirm it worked correctly
   curl http://localhost:11434/api/embeddings -d '{
    "model": "nomic-embed-text",
    "prompt": "test"
    }'
   ```
  

3. Create and activate virtual environment:
   
    For macOS/Linux:
      ```bash
      # Create virtual environment
      python -m venv plugnplay_env
         
      # Activate virtual environment
      source plugnplay_env/bin/activate
      ```
     
    For Windows:
      ```bash
      # Create virtual environment
      python -m venv plugnplay_env
        
      # Activate virtual environment
      .\plugnplay_env\Scripts\activate
      ```

4. Install Dependencies and run:
   ```bash
   # Install requirements
   pip install -r requirements.txt

   # Run the application
   streamlit run app.py
   ```

5. Deactivate when done
   ```bash
   deactivate
   ```


# Features

Document chunking with overlap
Semantic search
K-means clustering with validation metrics
Network visualization of document similarities
Similarity matrix visualization

# Usage

Upload a text document (.txt, .md, .csv, or .json)
Use the semantic search to find relevant sections
Explore document clusters and relationships
View similarity metrics between document sections

# Interpretation

1. **Document Chunking**
- Your text gets split into chunks of ~4,000 words
- Each chunk overlaps by 10% with the next chunk to maintain context
- You'll see chunks labeled as "yourfile_chunk_1", "yourfile_chunk_2", etc.
- Each chunk shows metadata about its position in the original text

2. **Semantic Search**
- Type your query in the search box
- Adjust the similarity threshold slider (higher = stricter matching)
- Results show:
  * Similarity score (0-1, higher means better match)
  * Preview or full content options
  * Which chunk it came from

3. **Clustering Validation**
- Elbow Curve: Look for the "bend" - it suggests optimal number of clusters
- Silhouette Score: Ranges from -1 to 1
  * Closer to 1 = better defined clusters
  * Closer to 0 = overlapping clusters
  * Below 0 = likely too many clusters

4. **Network Visualization**
- Each dot = a chunk of your text
- Lines between dots = similar content
- Colors = cluster assignments
- Thicker lines = stronger similarities
- Adjust similarity threshold to see different connection strengths

5. **Similarity Matrix**
- Shows how every chunk relates to every other chunk
- Darker colors = more similar
- Diagonal is always darkest (chunk compared to itself)
- Look for dark blocks to find groups of related content

6. **Cluster Assignments Table**
- Lists which chunk belongs to which cluster
- Use this to cross-reference with network visualization
- Helps track how your document was grouped

Remember: Higher similarity scores (>0.7) indicate strong relationships, while lower scores (<0.3) suggest weak or coincidental similarities.

