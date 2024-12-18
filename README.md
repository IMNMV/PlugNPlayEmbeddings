# PlugNPlayEmbeddings
# Text Analysis with Embeddings

This application provides semantic search and clustering analysis for text documents using Nomic embeddings and Ollama.

## Prerequisites
- Python 3.9+
- Ollama installed (https://ollama.ai](https://ollama.com)
- Nomic-embed-text model pulled in Ollama: https://ollama.com/library/nomic-embed-text

## Setup Instructions

1. Install Ollama:
   - Visit [https://ollama.ai](https://ollama.com](https://ollama.com)
   - Follow installation instructions for your OS

2. Pull the Nomic embeddings model using the command line interface:
   ```bash
   ollama pull nomic-embed-text:latest
   pip install -r requirements.txt
   streamlit run app.py

#Features

Document chunking with overlap
Semantic search
K-means clustering with validation metrics
Network visualization of document similarities
Similarity matrix visualization

#Usage

Upload a text document (.txt, .md, .csv, or .json)
Use the semantic search to find relevant sections
Explore document clusters and relationships
View similarity metrics between document sections

#Note
Ensure Ollama is running locally (default: http://localhost:11434) before starting the application.
