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
   ollama pull nomic-embed-text:latest
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

# Note
Ensure Ollama is running locally (default: http://localhost:11434) before starting the application.



